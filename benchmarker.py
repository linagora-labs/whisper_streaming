import os
import argparse
from tqdm import tqdm

LANGUAGE = "fr"
MIN_CHUNK_SIZE = 2
BUFFER_TRIMMING_SEC = 15
# MIN_CHUNK_SIZE = 0.6
# BUFFER_TRIMMING_SEC = 6
# GPU_SUPPORTED_PRECISIONS = ["int8","float32", "float16", "int8-float16"]
GPU_SUPPORTED_PRECISIONS = ["int8", "float32"]


CONFIG_FILE = "benchmark_configs.txt"

def get_possible_params_faster_whisper(device, small_test):
    if device == "cpu":
        return {'precisions': ["int8", "float32"],
                'vads': ["", "vad"],
                'methods': ["greedy", "beam-search"],
                } if not small_test else {'precisions': ["int8"], 'vads': ["", "vad"],
                'methods': ["greedy"]}
    return {'precisions': GPU_SUPPORTED_PRECISIONS,
                'vads': ["", "vad"],
                'methods': ["greedy", "beam-search"],
                } if not small_test else {'precisions': ["int8"], 'vads': ["", "vad"],
                'methods': ["greedy"]}


def get_possible_params_whisper_timestamped(device, small_test):
    if device == "cpu":
        return {'precisions': ["float32"],
                'vads': ["", "vad", "vad auditok"],
                'methods': ["greedy", "beam-search"],
                } if not small_test else {'precisions': ["float32"], 'vads': ["", "vad"],
                'methods': ["greedy"]}
    return {'precisions': ["float32"],
                'vads': ["", "vad", "vad auditok"],
                'methods': ["greedy", "beam-search"],
            } if not small_test else {'precisions': ["float32"], 'vads': ["", "vad"],
                'methods': ["greedy"]}

def is_params_valid_faster(device, precision, vad, method, subfolders=False):
    if device == "cpu":
        if precision=="float32" and (method=="beam-search" or vad==""):
            return False
        elif precision=="float16":
            return False
        elif method=="beam-search" and vad=="":
            return False
        return True
    else:
        if (precision=="float16" or precision=="int8-float16") and (method=="beam-search"):
            return False
        elif precision=="float32" and method=="beam-search":
            return False
        if subfolders:
            if (precision=="float16") and not (method=="beam-search" and precision=="int8" and vad==""):
                return False
    return True

def is_params_valid_whisper_timestamped(device, precision, vad, method, subfolders=False):
    if device == "cpu":
        if precision=="float16":
            return False
        elif method=="beam-search" and vad=="":
            return False
        return True
    else:
        if precision=="float16" and (method=="beam-search" or vad):
            return False
        if subfolders:
            if (precision=="float16") and not (method=="beam-search" and precision=="float32" and vad==""):
                return False
    return True

def generate_test(device, file="benchmark_configs.txt", subfolders=False, small_test=False):
    with open(file, "w") as f:
        backends = ["faster-whisper", "whisper-timestamped-openai"]#, "whisper-timestamped-transformers"]
        for backend in backends:
            if backend == "faster-whisper":
                possible_params = get_possible_params_faster_whisper(device, small_test)
            else:
                possible_params = get_possible_params_whisper_timestamped(device, small_test)
            for precision in possible_params['precisions']:
                for vad in possible_params['vads']:
                    for method in possible_params['methods']:
                        if vad == "vad auditok":
                            if method == "beam-search":
                                continue
                        test_id = f'{precision}_{method}'
                        if vad!="":
                            test_id += f'_{vad.replace(" ", "-")}'
                        if (backend == "faster-whisper" and is_params_valid_faster(device,precision, vad, method, subfolders)) or (backend.startswith("whisper-timestamped") and is_params_valid_whisper_timestamped(device, precision, vad, method, subfolders)):
                            f.write(f'{backend}_{test_id}\n')
                            if device=='cpu' and not small_test and ((backend.startswith("whisper-timestamped") and precision=="float32") or (backend=="faster-whisper" and precision=="int8")) and method=="greedy" and vad=="":
                                f.write(f'{backend}_{test_id}_2t\n')
                                f.write(f'{backend}_{test_id}_8t\n')
                                # f.write(f'{backend}_{test_id}_16t\n')
                            # if device=="cuda" and not small_test and ((backend.startswith("whisper-timestamped") and precision=="float32") or (backend=="faster-whisper" and precision=="int8")) and method=="greedy" and vad=="vad":
                            #     f.write(f'{backend}_{test_id}_previous-text\n')
                            if not subfolders:
                                if method == "greedy" and ((precision == "int8" and backend == "faster-whisper") or (backend.startswith("whisper-timestamped") and precision=="float32")):
                                    if small_test and vad=="":
                                        continue
                                    f.write(f'{backend}_{test_id}_silence\n')
                                    if vad=="":
                                        f.write(f'{backend}_{test_id}_bts-7\n')
                                        f.write(f'{backend}_{test_id}_mcs-0.6\n')
                                        f.write(f'{backend}_{test_id}_bts-7_mcs-0.6\n')
                            else:
                                if method == "greedy" and ((precision == "int8" and backend == "faster-whisper") or (backend.startswith("whisper-timestamped") and precision=="float32")) and vad=="":
                                    f.write(f'{backend}_{test_id}_offline\n')
                                    f.write(f'{backend}_{test_id}_medium_offline\n')
                                    f.write(f'{backend}_{test_id}_tiny_offline\n')
                                    f.write(f'{backend}_{test_id}_large-v1_offline\n')
                                    f.write(f'{backend}_{test_id}_bts-7_mcs-0.6\n')
                                    f.write(f'{backend}_{test_id}_previous-text\n')
                                    f.write(f'{backend}_{test_id}_bts-7_mcs-0.6_previous-text\n')
        suffixe = ""
        if subfolders:
            suffixe = "_offline"
        f.write(f'whisper-timestamped-openai_float32_greedy_model-bofenghuang/whisper-large-v3-french{suffixe}\n')
        f.write(f'whisper-timestamped-openai_float32_greedy_model-bofenghuang/whisper-large-v3-french-distil-dec16{suffixe}\n')
        f.write(f'whisper-timestamped-openai_float32_greedy_model-bofenghuang/whisper-large-v3-french-distil-dec8{suffixe}\n')
        f.write(f'whisper-timestamped-openai_float32_greedy_model-bofenghuang/whisper-large-v3-french-distil-dec4{suffixe}\n')
        f.write(f'whisper-timestamped-openai_float32_greedy_model-bofenghuang/whisper-large-v3-french-distil-dec2{suffixe}\n')


def run_commands(hardware, device, data, model_size, subfolder, args):
    benchmark_folder = f'{data.split("/")[-1]}_{model_size.split("-")[0]}{"_wer" if subfolder else ""}'
    output_path = os.path.join(benchmark_folder, hardware, device if device != "cuda" else "gpu")
    os.makedirs(output_path, exist_ok=True)
    pbar = tqdm(total=sum(1 for line in open(CONFIG_FILE, "r") if not line.startswith("#")))
    with open(CONFIG_FILE, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line.startswith("#"):
                params = line.split("_")
                backend = params[0]
                if backend.startswith('whisper'):
                    backend = '_'.join(backend.split("-", 1))
                sub_path = os.path.join(output_path, backend, '_'.join(params[1:]).replace('/','-'))
                if os.path.exists(os.path.join(sub_path, "result.json")) and not args.force_command:
                    print(f'Skipping {sub_path}')
                    pbar.update(1)
                    continue
                os.makedirs(sub_path, exist_ok=True)
                command = ""
                if device == "cpu":
                    command = f'/usr/bin/time -o {sub_path}/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" '
                if "silence" in params:
                    command += f'python whisper_online_full_options.py {data_silence} '
                else:
                    command += f'python whisper_online_full_options.py {data} '
                model = model_size
                if "medium" in params:
                    model="medium"
                elif "large-v1" in params:
                    model="large-v1"
                elif "tiny" in params:
                    model="tiny"
                elif len([x for x in params if x.startswith('model-')])>0:
                    model = [x for x in params if x.startswith('model-')][0].split("-",1)[1]
                tmp = [i for i in params if i.startswith('mcs')]
                min_chunk_size = tmp[0].split("-")[1] if tmp else MIN_CHUNK_SIZE
                tmp = [i for i in params if i.startswith('bts')]
                buffer_trimming_sec = tmp[0].split("-")[1] if tmp else BUFFER_TRIMMING_SEC
                command += f'--language {LANGUAGE} --model {model} --min-chunk-size {min_chunk_size} --buffer_trimming_sec {buffer_trimming_sec} --task transcribe --device {device} --backend {backend} --compute_type {params[1].replace("-", "_")} --method {params[2]} --output_path {sub_path}'
                if subfolder:
                    command += f' --subfolders'
                tmp = [i for i in params if i.startswith('vad')]
                if tmp:
                    command += f' --{tmp[0].replace("-", " ")}'
                if "previous-text" in params:
                    command += f' --previous_text'
                if "offline" in params:
                    command += f' --offline'
                tmp = [i for i in params if i[-1]=="t" and len(i)<=3 and i[0].isdigit()]
                if tmp:
                    command += f' --cpu_threads {tmp[0][:-1]}'
                print("Running:\n",command)
                os.system(command)
                pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', type=str, default='koios')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data', type=str, default='../data-fr/normal')
    parser.add_argument('--data_silence', type=str, default='../data-fr/silence')
    parser.add_argument('--subfolders', action="store_true", default=False)
    parser.add_argument('--model_size', type=str, default='large-v3')
    parser.add_argument('--force_command', action="store_true", default=False)
    parser.add_argument('--small_test', action="store_true", default=False)
    args = parser.parse_args()
    hardware = args.hardware
    device = args.device
    data = args.data
    model_size = args.model_size
    data_silence = args.data_silence
    subfolder = args.subfolders


    if hardware == "koios":
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']="1"
        os.environ['PYTHONPATH']="${PYTHONPATH}:/home/abert/abert/speech-army-knife"
        # os.system('export PYTHONPATH="${PYTHONPATH}:/home/abert/abert/whisper-timestamped"')
    elif hardware == "biggerboi":
        os.environ['PYTHONPATH']="${PYTHONPATH}:/home/abert/abert/speech-army-knife"      # don't work
        os.environ['PYTHONPATH']="${PYTHONPATH}:/home/abert/abert/whisper-timestamped"
    elif hardware == "lenovo":
        os.environ['PYTHONPATH']="${PYTHONPATH}:/mnt/c/Users/berta/Documents/Linagora/speech-army-knife"
        os.environ['PYTHONPATH']="${PYTHONPATH}:/mnt/c/Users/berta/Documents/Linagora/whisper-timestamped"
    else:
        i = input("Hardware not recognized, continue? (y/n) ")
        if i.lower() != "y":
            raise ValueError("Hardware not recognized")
    with open("log.txt", "w") as f:
        f.write(f'')
    if not os.path.exists(CONFIG_FILE):
        generate_test(device, CONFIG_FILE, subfolder, small_test=args.small_test)
    run_commands(hardware, device, data, model_size, subfolder, args)