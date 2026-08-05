import multiprocessing as mp
import torch

from config import DATASETS_CONFIG
from trainer import execute_dataset_pipeline
from utils import log_to_file, save_table_to_file

def process_runner_wrapper(config, gpu_queue, shared_return_dict):
    gpu_id = gpu_queue.get() 
    try:
        results = execute_dataset_pipeline(config, gpu_id)
        shared_return_dict[config["name"]] = results
    except Exception as e:
        import traceback
        error_msg = f"[CRITICAL ERROR] Pipeline failed for {config['name']}: {str(e)}\n{traceback.format_exc()}"
        log_to_file(error_msg)
        shared_return_dict[config["name"]] = []
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

    summary_results = []
    for config in DATASETS_CONFIG:
        summary_results.extend(shared_return_dict.get(config["name"], []))

    valid_rows = summary_results
    if valid_rows:
        valid_rows = sorted(valid_rows, key=lambda x: (x['dataset'], x['shot']))
        
        table = f"\n========================================================================================================================================================\n"
        table += f"{'Dataset':<17}{'Tr-Way':<9}{'Ts-Way':<9}{'Shot':<8}{'Params(M)':<13}{'Inference(GMACs)':<21}{'Total Train(GMACs)':<23}{'Train(s)':<12}{'Infer(s)':<12}{'Mem(MB)':<10}{'Acc(%)':<10}{'CI(%)':<8}\n"
        table += f"--------------------------------------------------------------------------------------------------------------------------------------------------------\n"
        for r in valid_rows:
            table += f"{r['dataset']:<17}{r['train_way']:<9}{r['test_way']:<9}{r['shot']:<8}{r['params_m']:<13.4f}{r['infer_gmacs']:<21.4f}{r['total_train_gmacs']:<23.4f}{r['train_time_s']:<12.4f}{r['inference_time_s']:<12.4f}{r['memory_mb']:<10.2f}{r['final_acc']:<10.4f}{r['final_ci']:<8.4f}\n"
        table += f"========================================================================================================================================================\n"
        
        log_to_file(table)
        save_table_to_file(table)
    else:
        log_to_file("All computational threads returned empty results arrays.")

if __name__ == "__main__":
    main()