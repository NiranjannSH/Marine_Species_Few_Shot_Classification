import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    SEED, CHECKPOINT_DIR, IMAGE_SIZE, S1_BATCH_SIZE, S1_EPOCHS, 
    S1_LR, S1_WEIGHT_DECAY, S1_TEMPERATURE, K_SHOTS, N_QUERY, 
    ADAPT_STEPS, ADAPT_LR, ADAPT_BACKBONE, FINAL_TEST_EPISODES, SAVE_STAGE_2_MODELS
)
from dataset import train_transform, SupConDataset, FewShotDataset
from models import SupConModel, ResNet18Encoder, CosineClassifier, SupConLoss
from utils import (
    set_seed, seed_worker, log_to_file, compute_model_macs, 
    freeze_batchnorm_stats, mean_confidence_interval
)

def execute_dataset_pipeline(config, gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    process_seed = SEED + (gpu_id if gpu_id else 0)
    set_seed(process_seed)

    dataset = FewShotDataset(config, split_seed=SEED, sampler_seed=process_seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_to_file(f"[{config['name']}] Train Classes: {dataset.train_classes}")
    log_to_file(f"[{config['name']}] Val Classes:   {dataset.val_classes}")
    log_to_file(f"[{config['name']}] Test Classes:  {dataset.test_classes}")

    # STAGE 1: CONTRASTIVE PRE-TRAINING
    log_to_file(f"[{config['name']}] Starting Stage 1 Contrastive Pre-Training.")
    s1_dataset = SupConDataset(config["root"], dataset.train_classes, train_transform)

    if len(s1_dataset) == 0:
        log_to_file(f"[ERROR] [{config['name']}] Has 0 samples at path: {config['root']}. Please verify your data location!")
        return []

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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    s1_macs_per_sample = compute_model_macs(model, dummy_input) + 128
    total_train_gmacs = (s1_macs_per_sample * 3.0 * (S1_BATCH_SIZE * 2) * len(loader) * S1_EPOCHS) / 1e9

    train_start = time.perf_counter()
    total_batches = len(loader)

    for epoch in range(S1_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, labels in loader:
            images = torch.stack(images, dim=0).to(device, non_blocking=True).view(-1, *images[0].shape[1:])
            labels = labels.to(device, non_blocking=True)
            bsz = labels.size(0)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                feats = model(images)
                feats = feats.view(2, bsz, -1).permute(1, 0, 2)
                loss = criterion(feats, labels)

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_epoch_loss = epoch_loss / total_batches
        log_to_file(f"[{config['name']}] Epoch [{epoch+1:03d}/{S1_EPOCHS}] | Avg Loss: {avg_epoch_loss:.4f}")

    train_time_s = time.perf_counter() - train_start
    
    encoder_state_dict = copy.deepcopy(model.encoder.state_dict())
    stage1_save_path = os.path.join(CHECKPOINT_DIR, f"{config['name']}_SupCon_Stage1.pth")
    torch.save(model.state_dict(), stage1_save_path)

    del model, optimizer, loader, s1_dataset
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()

    # STAGE 2: EPISODIC TEST-TIME ADAPTATION
    dataset_results = []
    way = config["test_way"]

    for shot in K_SHOTS:
        log_to_file(f"[{config['name']}] Starting Test-Time Adaptation loop ({shot}-Shot)...")
            
        inference_start = time.perf_counter()
        episode_encoder = ResNet18Encoder().to(device)
        classifier = CosineClassifier(512, way).to(device)
        ft_criterion = nn.CrossEntropyLoss()
        support_labels = torch.arange(way, device=device).unsqueeze(1).repeat(1, shot).view(-1)

        params_m = (sum(p.numel() for p in episode_encoder.parameters()) + sum(p.numel() for p in classifier.parameters())) / 1e6

        dummy_single_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        encoder_macs_per_sample = compute_model_macs(episode_encoder, dummy_single_image)
        classifier_macs_per_sample = 512 * way
        
        support_count = way * shot
        query_count = way * N_QUERY
        norm_macs_s2 = 512
        
        initial_support_macs = (encoder_macs_per_sample + norm_macs_s2) * support_count
        adaptation_macs = ((encoder_macs_per_sample + classifier_macs_per_sample) * support_count * 3.0) * ADAPT_STEPS
        query_macs = (encoder_macs_per_sample + classifier_macs_per_sample) * query_count
        
        total_episode_inference_macs = initial_support_macs + adaptation_macs + query_macs
        inference_gmacs = total_episode_inference_macs / 1e9

        chunk_accuracies = []

        for ep in range(FINAL_TEST_EPISODES):
            support, query, labels = dataset.sample_episode("test", shot, N_QUERY)
            support, query, labels = support.to(device), query.to(device), labels.to(device)

            episode_encoder.load_state_dict(encoder_state_dict, strict=True)
            episode_encoder.eval()

            with torch.no_grad():
                raw_support = episode_encoder(support)
                feats_support = F.adaptive_avg_pool2d(raw_support, (1, 1)).flatten(1)
                feats_support = F.normalize(feats_support, dim=1)
                prototypes = feats_support.view(way, shot, -1).mean(dim=1)
                classifier.weight.copy_(F.normalize(prototypes, dim=1))

            if ADAPT_STEPS > 0:
                if ADAPT_BACKBONE:
                    episode_encoder.train()
                    episode_encoder.apply(freeze_batchnorm_stats)
                    ft_optim = torch.optim.SGD([
                        {"params": classifier.parameters(), "lr": ADAPT_LR},
                        {"params": episode_encoder.parameters(), "lr": ADAPT_LR * 0.1},
                    ], momentum=0.9)
                else:
                    ft_optim = torch.optim.SGD(classifier.parameters(), lr=ADAPT_LR, momentum=0.9)

                for _ in range(ADAPT_STEPS):
                    if ADAPT_BACKBONE:
                        episode_encoder.train()
                        episode_encoder.apply(freeze_batchnorm_stats)
                        raw_supp_adapt = episode_encoder(support)
                        supp_adapt_pooled = F.adaptive_avg_pool2d(raw_supp_adapt, (1, 1)).flatten(1)
                        support_logits = classifier(supp_adapt_pooled)
                        loss = ft_criterion(support_logits, support_labels)
                    else:
                        support_logits = classifier(feats_support.detach())
                        loss = ft_criterion(support_logits, support_labels)

                    ft_optim.zero_grad(set_to_none=True)
                    loss.backward()
                    ft_optim.step()

            if SAVE_STAGE_2_MODELS and ep % 100 == 0:
                tta_dir = os.path.join(CHECKPOINT_DIR, f"{config['name']}_{shot}shot_TTA_Models")
                os.makedirs(tta_dir, exist_ok=True)
                torch.save({
                    "episode": ep,
                    "encoder_state_dict": episode_encoder.state_dict(),
                    "classifier_state_dict": classifier.state_dict()
                }, os.path.join(tta_dir, f"episode_{ep:04d}.pth"))

            episode_encoder.eval()
            classifier.eval()
            with torch.no_grad():
                raw_query = episode_encoder(query)
                query_pooled = F.adaptive_avg_pool2d(raw_query, (1, 1)).flatten(1)
                preds = classifier(query_pooled).argmax(dim=1)
                chunk_accuracies.append((preds == labels).float().mean().item() * 100.0)

        final_mean, final_ci = mean_confidence_interval(chunk_accuracies)
        inference_time_s = time.perf_counter() - inference_start
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

        dataset_results.append({
            "dataset": config["name"],
            "train_way": config["train_way"],
            "test_way": config["test_way"],
            "shot": shot,
            "params_m": params_m,
            "infer_gmacs": inference_gmacs,
            "total_train_gmacs": total_train_gmacs,
            "train_time_s": train_time_s,
            "inference_time_s": inference_time_s,
            "memory_mb": peak_memory_mb,
            "final_acc": final_mean,
            "final_ci": final_ci,
        })
        log_to_file(f"[RESULT] {config['name']} | {shot}-Shot: {final_mean:.4f}% +/- {final_ci:.4f}%")

    return dataset_results
