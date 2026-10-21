import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    SEED, CHECKPOINT_DIR, IMAGE_SIZE, S1_BATCH_SIZE, S1_EPOCHS,
    S1_LR, S1_WEIGHT_DECAY, S1_TEMPERATURE, K_SHOTS, VAL_EPISODES, FINAL_TEST_EPISODES,
    ADAPT_STEPS, ADAPT_LR, BACKBONE_LR_FACTOR, ADAPT_MOMENTUM, STEP_SETTINGS
)
from dataset import train_transform, SupConDataset, class_split, episode_path, load_episodes
from models import SupConModel, ResNet18Encoder, CosineClassifier, SupConLoss
from utils import (
    set_seed, seed_worker, compute_model_macs, episode_gmacs,
    freeze_batchnorm_stats, mean_confidence_interval
)

def _pooled(encoder, x):
    """512-d global-average-pooled feature of the ResNet-18 feature map."""
    return F.adaptive_avg_pool2d(encoder(x), (1, 1)).flatten(1)

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def evaluate_no_tta(encoder, episodes, way, shot, device):
    """Validation after every epoch: prototype-initialised cosine classifier, no adaptation."""
    classifier = CosineClassifier(512, way).to(device)
    classifier.eval()
    encoder.eval()
    accs = []
    for support, query, labels in episodes:
        support, query, labels = support.to(device), query.to(device), labels.to(device)
        with torch.inference_mode():
            feats = F.normalize(_pooled(encoder, support), dim=1)
            classifier.weight.copy_(F.normalize(feats.view(way, shot, -1).mean(dim=1), dim=1))
            preds = classifier(_pooled(encoder, query)).argmax(dim=1)
        accs.append((preds == labels).float().mean().item() * 100.0)
    del classifier
    return mean_confidence_interval(accs)

def evaluate_test(encoder_state_dict, episodes, way, shot, steps, device):
    """Final test. For every episode the encoder is reset to the trained checkpoint and a cosine
    classifier is initialised from the l2-normalised support prototypes. With steps > 0, classifier
    and encoder are adapted on the support set only (SGD with momentum, encoder learning rate
    = BACKBONE_LR_FACTOR x classifier learning rate, BatchNorm statistics frozen)."""
    encoder = ResNet18Encoder().to(device)
    support_labels = torch.arange(way, device=device).unsqueeze(1).repeat(1, shot).view(-1)
    accs = []
    for support, query, labels in episodes:
        support, query, labels = support.to(device), query.to(device), labels.to(device)
        encoder.load_state_dict(encoder_state_dict, strict=True)
        encoder.eval()
        classifier = CosineClassifier(512, way).to(device)
        classifier.eval()
        with torch.inference_mode():
            feats = F.normalize(_pooled(encoder, support), dim=1)
            classifier.weight.copy_(F.normalize(feats.view(way, shot, -1).mean(dim=1), dim=1))

        if steps > 0:
            encoder.train()
            encoder.apply(freeze_batchnorm_stats)
            optimizer = torch.optim.SGD([
                {"params": classifier.parameters(), "lr": ADAPT_LR},
                {"params": encoder.parameters(), "lr": ADAPT_LR * BACKBONE_LR_FACTOR},
            ], momentum=ADAPT_MOMENTUM, foreach=False)
            for _ in range(steps):
                encoder.train()
                encoder.apply(freeze_batchnorm_stats)
                loss = F.cross_entropy(classifier(_pooled(encoder, support)), support_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            del optimizer

        encoder.eval()
        classifier.eval()
        with torch.inference_mode():
            preds = classifier(_pooled(encoder, query)).argmax(dim=1)
        accs.append((preds == labels).float().mean().item() * 100.0)
        del classifier
    del encoder
    mean, ci = mean_confidence_interval(accs)
    return mean, ci, accs

def execute_dataset_pipeline(config, gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    use_cuda = device.type == "cuda"

    process_seed = SEED
    set_seed(process_seed)
    name = config["name"]

    # Setup Dataset-Specific Folders: checkpoints/<dataset_name>/logs/
    dataset_dir = os.path.join(CHECKPOINT_DIR, name)
    logs_dir = os.path.join(dataset_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(logs_dir, f"{name}_execution_log_{timestamp}.txt")

    # Local logger writing strictly to dataset's own log file
    def log_msg(message):
        current_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{current_ts}] {message}"
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
        print(formatted, flush=True)

    train_classes, val_classes, test_classes = class_split(config)
    log_msg(f"[{name}] Base classes:       {train_classes}")
    log_msg(f"[{name}] Validation classes: {val_classes}")
    log_msg(f"[{name}] Novel classes:      {test_classes}")

    # Fixed, leak-free episodes (built once by make_episodes.py)
    val_episodes, test_episodes = {}, {}
    for shot in K_SHOTS:
        for split, store, n in (("val", val_episodes, VAL_EPISODES), ("test", test_episodes, FINAL_TEST_EPISODES)):
            path = episode_path(name, split, shot)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found. Run 'python make_episodes.py' first.")
            store[shot] = load_episodes(path)[:n]
        log_msg(f"[{name}] {shot}-shot: {len(val_episodes[shot])} validation / {len(test_episodes[shot])} test episodes")

    # STAGE 1: SUPCON PRE-TRAINING WITH PER-EPOCH VALIDATION
    log_msg(f"[{name}] Starting Stage 1 SupCon pre-training "
            f"(tau={S1_TEMPERATURE}, LR={S1_LR}, WD={S1_WEIGHT_DECAY}, {S1_EPOCHS} epochs).")
    set_seed(process_seed)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    s1_dataset = SupConDataset(config["root"], train_classes, train_transform)
    if len(s1_dataset) == 0:
        log_msg(f"[ERROR] [{name}] Has 0 samples at path: {config['root']}. Please verify your data location!")
        return None

    g = torch.Generator()
    g.manual_seed(process_seed)
    loader = DataLoader(
        s1_dataset, batch_size=S1_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )

    model = SupConModel().to(device)
    criterion = SupConLoss(S1_TEMPERATURE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=S1_LR, weight_decay=S1_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=S1_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    s1_macs_per_sample = compute_model_macs(model, dummy_input) + 128

    best_val = {s: -float("inf") for s in K_SHOTS}
    best_epoch = {s: 0 for s in K_SHOTS}
    best_encoder = {s: None for s in K_SHOTS}
    val_time_s = 0.0
    train_start = time.perf_counter()

    for epoch in range(S1_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, labels in loader:
            images = torch.stack(images, dim=0).to(device, non_blocking=True).view(-1, *images[0].shape[1:])
            labels = labels.to(device, non_blocking=True)
            bsz = labels.size(0)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                feats = model(images)
                feats = feats.view(2, bsz, -1).permute(1, 0, 2)
                loss = criterion(feats, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        log_msg(f"[{name}] Epoch [{epoch+1:03d}/{S1_EPOCHS}] | Avg Loss: {epoch_loss / len(loader):.4f}")

        # Validation after every epoch; the best checkpoint is kept separately for 1-shot and 5-shot
        v_start = time.perf_counter()
        for shot in K_SHOTS:
            val_acc, val_ci = evaluate_no_tta(model.encoder, val_episodes[shot], config["val_way"], shot, device)
            log_msg(f"[{name}] Epoch {epoch+1:03d} Validation | {shot}-Shot = {val_acc:.4f}% +/- {val_ci:.4f}%")
            if np.isfinite(val_acc) and val_acc > best_val[shot]:
                best_val[shot], best_epoch[shot] = val_acc, epoch + 1
                best_encoder[shot] = copy.deepcopy(model.encoder.state_dict())
                torch.save({"shot": shot, "best_epoch": epoch + 1, "best_val_accuracy": val_acc,
                            "encoder_state_dict": best_encoder[shot]},
                           os.path.join(dataset_dir, f"{name}_SupCon_Stage1_best_val_{shot}shot.pth"))
        val_time_s += time.perf_counter() - v_start

    _sync(device)
    train_time_s = time.perf_counter() - train_start
    batches = len(loader)
    for shot in K_SHOTS:
        if best_encoder[shot] is None:
            best_encoder[shot] = copy.deepcopy(model.encoder.state_dict())
            best_epoch[shot] = S1_EPOCHS
        log_msg(f"[{name}] Best {shot}-shot checkpoint: epoch {best_epoch[shot]} (validation {best_val[shot]:.2f}%)")

    del model, optimizer, scheduler, loader, s1_dataset
    if use_cuda:
        torch.cuda.empty_cache()

    # Training cost: the complete 100-epoch budget, including the per-epoch validation
    encoder_probe = ResNet18Encoder().to(device)
    enc_macs = compute_model_macs(encoder_probe, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    del encoder_probe
    train_gmacs = s1_macs_per_sample * 3.0 * (S1_BATCH_SIZE * 2) * batches * S1_EPOCHS / 1e9
    val_gmacs = sum(VAL_EPISODES * episode_gmacs(enc_macs, config["val_way"], s, 0) for s in K_SHOTS) * S1_EPOCHS

    # STAGE 2: FINAL TEST WITHOUT (0 steps) AND WITH TEST-TIME ADAPTATION (ADAPT_STEPS)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    way = config["test_way"]
    rows, episode_rows = [], []

    for shot in K_SHOTS:
        row = {"dataset": name, "train_way": config["train_way"], "test_way": way, "shot": shot,
               "best_epoch": best_epoch[shot], "best_val": best_val[shot],
               "params_m": (sum(p.numel() for p in ResNet18Encoder().parameters()) + 512 * way) / 1e6,
               "train_gmacs_total": train_gmacs + val_gmacs, "train_time_s": train_time_s,
               "validation_time_s": val_time_s}
        per_step = {}
        for steps in STEP_SETTINGS:
            set_seed(process_seed)
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            start = time.perf_counter()
            mean, ci, accs = evaluate_test(best_encoder[shot], test_episodes[shot], way, shot, steps, device)
            _sync(device)
            row[f"infer_gmacs_tta{steps}"] = episode_gmacs(enc_macs, way, shot, steps)
            row[f"infer_time_s_tta{steps}"] = time.perf_counter() - start
            row[f"memory_mb_tta{steps}"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if use_cuda else 0.0
            row[f"acc_tta{steps}"] = mean
            row[f"ci_tta{steps}"] = ci
            per_step[steps] = accs
            log_msg(f"[RESULT] {name} | {shot}-Shot | adaptation steps = {steps} | "
                    f"Test = {mean:.2f}% +/- {ci:.2f}% | GMACs/episode = {row[f'infer_gmacs_tta{steps}']:.2f}")
        rows.append(row)
        for i, (a0, a1) in enumerate(zip(per_step[0], per_step[ADAPT_STEPS]), start=1):
            episode_rows.append({"dataset": name, "shot": shot, "episode": i,
                                 "acc_no_tta": a0, f"acc_tta{ADAPT_STEPS}": a1})

    return {"rows": rows, "episodes": episode_rows}
