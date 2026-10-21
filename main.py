import csv
import multiprocessing as mp
import torch

from config import DATASETS_CONFIG, ADAPT_STEPS, ADAPT_LR, S1_TEMPERATURE
from trainer import execute_dataset_pipeline
from utils import log_to_file, save_table_to_file, CSV_FILE_PATH, EPISODE_CSV_PATH

def process_runner_wrapper(config, gpu_queue, shared_return_dict):
    gpu_id = gpu_queue.get()
    try:
        shared_return_dict[config["name"]] = execute_dataset_pipeline(config, gpu_id)
    except Exception as e:
        import traceback
        error_msg = f"[CRITICAL ERROR] Pipeline failed for {config['name']}: {str(e)}\n{traceback.format_exc()}"
        log_to_file(error_msg)
        shared_return_dict[config["name"]] = None
    finally:
        gpu_queue.put(gpu_id)

def main():
    mp.set_start_method("spawn", force=True)
    gpu_count = torch.cuda.device_count()

    log_to_file(f"System Check: Detected {gpu_count} GPUs. Commencing Dynamically Balanced Process Allocations...")

    manager = mp.Manager()
    shared_return_dict = manager.dict()

    gpu_queue = manager.Queue()
    for g in range(max(1, gpu_count)):
        gpu_queue.put(g if gpu_count > 0 else None)

    processes = []
    for config in DATASETS_CONFIG:
        p = mp.Process(
            target=process_runner_wrapper,
            args=(config, gpu_queue, shared_return_dict)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    rows, episode_rows = [], []
    for config in DATASETS_CONFIG:
        result = shared_return_dict.get(config["name"])
        if result:
            rows.extend(result["rows"])
            episode_rows.extend(result["episodes"])

    if not rows:
        log_to_file("All computational threads returned empty results arrays.")
        return

    rows = sorted(rows, key=lambda x: (x["dataset"], x["shot"]))
    s = ADAPT_STEPS
    head = (f"{'Dataset':<16}{'Shot':<6}{'BestEp':<8}{'Val(%)':<9}{'Params(M)':<11}{'TotalTrain(GMACs)':<19}"
            f"{'Train(s)':<10}{'GMACs noTTA':<13}{'GMACs TTA':<12}{'Time(s) noTTA':<15}{'Time(s) TTA':<13}"
            f"{'Mem(MB) TTA':<13}{'Acc noTTA (%)':<17}{'Acc TTA (%)':<15}")
    lines = ["=" * len(head),
             f"PROPOSED (SupCon, tau {S1_TEMPERATURE}) | without TTA (0 steps) and with TTA (LR {ADAPT_LR}, {s} steps)",
             "=" * len(head), head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['dataset']:<16}{r['shot']:<6}{r['best_epoch']:<8}{r['best_val']:<9.2f}{r['params_m']:<11.4f}"
            f"{r['train_gmacs_total']:<19.2f}{r['train_time_s']:<10.1f}{r['infer_gmacs_tta0']:<13.2f}"
            f"{r[f'infer_gmacs_tta{s}']:<12.2f}{r['infer_time_s_tta0']:<15.2f}{r[f'infer_time_s_tta{s}']:<13.2f}"
            f"{r[f'memory_mb_tta{s}']:<13.1f}{r['acc_tta0']:.2f} ± {r['ci_tta0']:<9.2f}"
            f"{r[f'acc_tta{s}']:.2f} ± {r[f'ci_tta{s}']:.2f}")
    lines += ["=" * len(head),
              "TotalTrain = 100 epochs of Stage-1 training (backward = 2 x forward) + per-epoch validation; "
              "one run per dataset, shared by 1-shot and 5-shot.",
              "GMACs = per test episode; Time(s) and Mem(MB) = all 600 test episodes; Acc = mean ± 95% CI."]
    table = "\n".join(lines) + "\n"

    log_to_file("\n" + table)
    save_table_to_file(table)
    with open(CSV_FILE_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(EPISODE_CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

if __name__ == "__main__":
    main()
