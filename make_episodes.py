"""
make_episodes.py -- builds the fixed, leak-free validation and test episodes.
Run once, after prepare_data.py and before main.py.

For every dataset:
  * class split: sorted class folders shuffled with random.Random(42) (dataset.class_split);
  * every image is passed through the evaluation transform (resize to 84x84, ImageNet normalisation);
  * pixel-identical images are kept only once per class, and images that also occur in another
    class (in any split) are removed from the validation and test pools;
  * 600 episodes per shot setting are sampled WITHOUT replacement, so no identical image tensor
    appears more than once within an episode (support and query included);
  * validation: 2-way, random.Random(42 + dataset_index * 100 + shot);
    test: N-way, random.Random(888 + shot); 5 query images per class;
  * every saved file is reloaded and verified.

    python make_episodes.py            # build and verify
    python make_episodes.py --force    # rebuild even if the files already exist
"""
import argparse
import hashlib
import json
import os
import random

import torch
from PIL import Image

from config import DATASETS_CONFIG, EPISODE_DIR, IMAGE_EXTS, IMAGE_SIZE, K_SHOTS, N_QUERY, \
    VAL_EPISODES, FINAL_TEST_EPISODES
from dataset import eval_transform, class_split, episode_path

VAL_SEED = lambda ds_idx, shot: 42 + ds_idx * 100 + shot
TEST_SEED = lambda ds_idx, shot: 888 + shot


def tensor_key(t):
    return hashlib.sha1(t.contiguous().numpy().tobytes()).hexdigest()


def scan_class(root, class_name, keep_tensors):
    """Sorted image files of one class -> list of (relative path, pixel key, tensor or None)."""
    class_dir = os.path.join(root, class_name)
    items = []
    for f in sorted(os.listdir(class_dir)):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        try:
            with Image.open(os.path.join(class_dir, f)) as img:
                t = eval_transform(img.convert("RGB"))
        except Exception:
            continue
        items.append((f"{class_name}/{f}", tensor_key(t), t if keep_tensors else None))
    return items


def build_pools(root, train_classes, val_classes, test_classes):
    """De-duplicated validation/test pools; also returns the pixel keys of the base classes."""
    scanned = {}
    for c in train_classes:
        scanned[c] = scan_class(root, c, keep_tensors=False)
    for c in val_classes + test_classes:
        scanned[c] = scan_class(root, c, keep_tensors=True)

    owners = {}
    for c, items in scanned.items():
        for _, k, _ in items:
            owners.setdefault(k, set()).add(c)

    pools = {}
    for c in val_classes + test_classes:
        seen, pool = set(), []
        for rel, k, t in scanned[c]:
            if k in seen or len(owners[k]) > 1:   # duplicate within the class, or shared with another class
                seen.add(k)
                continue
            seen.add(k)
            pool.append((rel, k, t))
        pools[c] = pool
    train_keys = {k for c in train_classes for _, k, _ in scanned[c]}
    return pools, train_keys


def sample_episodes(rng, classes, pools, way, shot, n_episodes):
    need = shot + N_QUERY
    short = {c: len(pools[c]) for c in classes if len(pools[c]) < need}
    if short:
        raise RuntimeError(f"Classes with fewer than {need} unique images: {short}. "
                           "Refusing to sample with replacement.")
    episodes = []
    for ep_num in range(1, n_episodes + 1):
        selected = rng.sample(classes, way)
        support, query, labels, s_files, q_files = [], [], [], [], []
        for label, c in enumerate(selected):
            picks = [pools[c][i] for i in rng.sample(range(len(pools[c])), need)]
            support += [t for _, _, t in picks[:shot]]
            query += [t for _, _, t in picks[shot:]]
            s_files += [rel for rel, _, _ in picks[:shot]]
            q_files += [rel for rel, _, _ in picks[shot:]]
            labels += [label] * N_QUERY
        episodes.append({"episode_id": ep_num, "support": torch.stack(support), "query": torch.stack(query),
                         "labels": torch.tensor(labels, dtype=torch.long), "classes": selected,
                         "support_files": s_files, "query_files": q_files})
    return episodes


def verify_file(path, way, shot, allowed_classes, train_keys, n_episodes):
    bank = torch.load(path, map_location="cpu", weights_only=False)
    problems = []
    if len(bank) != n_episodes:
        problems.append(f"{len(bank)} episodes, expected {n_episodes}")
    for ep in bank:
        s, q = ep["support"], ep["query"]
        if tuple(s.shape) != (way * shot, 3, IMAGE_SIZE, IMAGE_SIZE) or \
                tuple(q.shape) != (way * N_QUERY, 3, IMAGE_SIZE, IMAGE_SIZE):
            problems.append(f"episode {ep['episode_id']}: wrong tensor shapes")
        if len(set(ep["classes"])) != way or not set(ep["classes"]) <= set(allowed_classes):
            problems.append(f"episode {ep['episode_id']}: classes {ep['classes']} are not {way} split classes")
        keys = [tensor_key(t) for t in s] + [tensor_key(t) for t in q]
        if len(set(keys)) != len(keys):
            problems.append(f"episode {ep['episode_id']}: an identical image tensor appears twice")
        if set(keys) & train_keys:
            problems.append(f"episode {ep['episode_id']}: image identical to a base-class image")
        if len(problems) > 20:
            break
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild episode files that already exist")
    args = ap.parse_args()
    os.makedirs(EPISODE_DIR, exist_ok=True)

    for ds_idx, config in enumerate(DATASETS_CONFIG):
        name = config["name"]
        train_classes, val_classes, test_classes = class_split(config)
        print(f"[{name}] base: {train_classes}\n[{name}] validation: {val_classes}\n[{name}] novel: {test_classes}")
        pools, train_keys = build_pools(config["root"], train_classes, val_classes, test_classes)

        for split, classes, way, seed_rule, n_episodes in (
            ("val", val_classes, config["val_way"], VAL_SEED, VAL_EPISODES),
            ("test", test_classes, config["test_way"], TEST_SEED, FINAL_TEST_EPISODES),
        ):
            for shot in K_SHOTS:
                path = episode_path(name, split, shot)
                if os.path.isfile(path) and not args.force:
                    print(f"[{name}] {split} {shot}-shot: exists, skipped ({path})")
                    continue
                seed = seed_rule(ds_idx, shot)
                episodes = sample_episodes(random.Random(seed), classes, pools, way, shot, n_episodes)
                torch.save(episodes, path)
                with open(path.replace(".pt", "_files.json"), "w", encoding="utf-8") as f:
                    json.dump([{"episode_id": e["episode_id"], "classes": e["classes"],
                                "support_files": e["support_files"], "query_files": e["query_files"]}
                               for e in episodes], f)
                problems = verify_file(path, way, shot, classes, train_keys, n_episodes)
                if problems:
                    raise RuntimeError(f"Verification failed for {path}:\n  " + "\n  ".join(problems[:20]))
                print(f"[{name}] {split} {shot}-shot: {way}-way, seed {seed}, {n_episodes} episodes -> {path} (verified)")


if __name__ == "__main__":
    main()
