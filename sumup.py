import json
import argparse
import os
from tqdm import tqdm

# plot data
import matplotlib.pyplot as plt
import numpy as np
from evaluate_wer import process_wer

COLORS_DICT = {
    0: 'blue',
    1: 'red',
    2: 'green',
    3: 'yellow',
    4: 'orange',
    5: 'purple',
    6: 'brown',
    7: 'pink',
    8: 'gray',
    9: 'olive',
    10: 'cyan',
}

def search_row(data, hardware, device, backend, compute_type, vad, method):
    for row in data:
        if row['hardware'] == hardware and row['device'] == device and row['backend'] == backend and row['compute_type'] == compute_type and row['vad'] == vad and row['method'] == method:
            return row
    return None

def search_rows(data, hardware=None, device=None, backend=None, compute_type=None, \
    vad=None, method=None, condition_on_previous_text=None, buffer_trimming=None, min_chunk_size=None, \
    data_type=None, cpu_threads=None, model_size=None, offline=None):
    rows = []
    for row in data:
        if (hardware is None or row['hardware'] == hardware) and (model_size is None or row['model_size']==model_size) and \
            (offline is None or row['offline']==offline) and (cpu_threads is None or row['cpu_threads']==cpu_threads) and \
            (data_type is None or row['data_type']==data_type) and (condition_on_previous_text is None or row['condition_on_previous_text']==condition_on_previous_text) \
            and (device is None or row['device'] == device) and (backend is None or row['backend'] == backend) and \
            ((compute_type is None or row['compute_type'] == compute_type) or (compute_type=="best" and row['compute_type']=="int8" and row['backend']=="faster") or \
            (compute_type=="best" and row['compute_type']=="float32" and row['backend']=="timestamped")) and (vad is None or row['vad'] == vad) and \
            (method is None or row['method'] == method) and (buffer_trimming is None or row['buffer_trimming'] == buffer_trimming) and \
            (min_chunk_size is None or row['min_chunk_size'] == min_chunk_size):
            rows.append(row)
                
    return rows

def search_rows_by_key(data, key, value):
    rows = []
    for row in data:
        if row[key] == value:
            rows.append(row)
    return rows

def plot_processting_times_of_files(row, output_path='plots'):
    os.makedirs(output_path, exist_ok=True)
    #define fig size
    plt.rcParams["figure.figsize"] = (12,8)
    fig, ax = plt.subplots()
    plot_mean= []
    plot_std = []
    data_names = row['data'].keys()
    for j in data_names:
        lv = row['data'][j]
        plot_mean.append(np.mean(lv['segment_processing_time']))
        plot_std.append(np.std(lv['segment_processing_time']))
    # print(data_names)
    data_names = [x.split('/')[-1] for x in data_names]
    data_names = [x[0:14] for x in data_names]

    # data_names = [x.split('.')[0] for x in data_names]
    # print(data_names)
    #plot mean and std for each file
    ax.bar(data_names, plot_mean)
    ax.set_xlabel('files')
    plt.xticks(rotation=40)
    ax.set_ylabel('processing time [s]')
    ax.set_title('processing time')
    plt.savefig(os.path.join(output_path,'processing_time_per_file.png'))
    plt.close()

def get_values(row, key='segment_latency', mode='max'):
    data_names = row['data'].keys()
    values= []
    duration = []
    for j in data_names:
        lv = row['data'][j]
        if lv.get(key, None) is None:
            continue
        if isinstance(lv[key], list):
            values.extend(lv[key])
            duration.extend(lv['segment_duration'])
        else:
            values.append(lv[key])
            duration.append(lv['segment_duration'])
    if mode=="max":
        return max(values)
    elif mode=="min":
        return min(values)
    elif mode=="rtf":
        return [x/y for x,y in zip(values, duration)]
    return values

def plot(data, wer=False):
    if wer:
        print("Plotting WER")
        plot_param(data, title="WER depending on precision on 1080TI (GPU) for faster-whisper", ylabel="WER", key='wer_score', output_path='plots/wer/faster', hardware="koios", device="gpu", backend="faster", method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", compute_type=None)
        plot_param(data, title="WER depending on precision on 1080TI (GPU)", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", backend=None, method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", compute_type=None)
        plot_param(data, title="WER depending on method on 1080TI (GPU)", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", backend=None, method=None, vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", compute_type="best")
        plot_param(data, title="WER depending on method with VAD on 1080TI (GPU)", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", backend=None, method=None, vad='VAD', condition_on_previous_text="NoCondition", data_type="speech", compute_type="best")
        plot_param(data, title="WER depending on VAD on 1080TI (GPU)", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", backend=None, method="greedy", vad=None, condition_on_previous_text="NoCondition", data_type="speech", compute_type="best")
        plot_param(data, title="WER depending on Previous text on 1080TI (GPU)", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", backend=None, method="greedy", vad="VAD", condition_on_previous_text=None, data_type="speech", compute_type="best")
        plot_param(data, title="WER streaming vs offline", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", vad="NoVAD",method="greedy", condition_on_previous_text="NoCondition", data_type="speech", model_size="large-v3", offline=None,  compute_type="best")
        plot_param(data, title="WER model size streaming", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", vad="NoVAD",method="greedy", condition_on_previous_text="NoCondition", data_type="speech", model_size=None, offline="streaming", compute_type="best")
        plot_param(data, title="WER model offline", ylabel="WER", key='wer_score', output_path='plots/wer/', hardware="koios", device="gpu", vad="NoVAD",method="greedy", condition_on_previous_text="NoCondition", data_type="speech", model_size=None, offline="offline", compute_type="best")

    
    
    else:
        print("Plotting latencies")
        data_gpu = search_rows_by_key(data, 'device', 'gpu')
        # plot_processting_times_per_params(data_gpu, "GPU processing times", output_path='plots/gpu')

        plot_param(data_gpu, title="Latency depending on precision on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/koios/faster', hardware="koios", device="gpu", backend="faster", method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="Latency depending on precision on 1080TI (GPU) for whisper-timestamped", key='segment_latency',output_path='plots/gpu/koios/timestamped', hardware="koios", device="gpu", backend="timestamped", method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="Latency depending on VAD on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu/koios', hardware="koios", device="gpu", method="greedy", compute_type="best", condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="RTF depending on VAD on 1080TI (GPU)", key='segment_processing_time', ylabel="Processing time/duration", plot_data_mode='rtf', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, method="greedy", condition_on_previous_text="NoCondition", data_type="speech", compute_type="best")
        plot_param(data_gpu, title="Latency depending on VAD on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/koios/faster', hardware="koios", device="gpu", backend="faster", method="greedy", condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="Latency depending on VAD for silence data on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu/koios', hardware="koios", device="gpu", method="greedy", compute_type="best", condition_on_previous_text="NoCondition", data_type="silence")
        plot_param(data_gpu, title="RTF depending on VAD for silence data on 1080TI (GPU)", key='segment_processing_time', ylabel="Processing time/duration", plot_data_mode='rtf', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, method="greedy", condition_on_previous_text="NoCondition", data_type="silence", compute_type="best")
        plot_param(data_gpu, title="Latency depending on VAD for silence data on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/koios/faster', hardware="koios", device="gpu", backend="faster", method="greedy", condition_on_previous_text="NoCondition", data_type="silence")
        plot_param(data_gpu, title="Latency depending on method on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu/koios', hardware="koios", device="gpu", vad='VAD', compute_type="best", condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="Latency depending on method on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/koios/faster', hardware="koios", device="gpu", backend="faster", compute_type="best", condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="RTF depending on method on 1080TI (GPU)", key='segment_processing_time', ylabel="Processing time/duration", plot_data_mode='rtf', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, method=None, vad="VAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="best")
        plot_param(data_gpu, title="Latency depending on VAD on 1080TI (GPU) for whisper-timestamped", key='segment_latency',output_path='plots/gpu/koios/timestamped', hardware="koios", device="gpu", compute_type="float32",backend="timestamped", method="greedy", vad=None, condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="Latency depending on VAD for silence data on 1080TI (GPU) for whisper-timestamped", key='segment_latency',output_path='plots/gpu/koios/timestamped', hardware="koios", device="gpu", compute_type="float32",backend="timestamped", method="greedy", vad=None, condition_on_previous_text="NoCondition", data_type="silence")
        plot_param(data_gpu, title="Latency depending Previous text on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, method="greedy", compute_type="best", vad="NoVAD", condition_on_previous_text=None, data_type="speech")
        plot_param(data_gpu, title="Latency depending on model on 1080TI (GPU) for whisper-timestamped", key='segment_latency',output_path='plots/gpu/koios/timestamped', hardware="koios", device="gpu", compute_type="float32",backend="timestamped", method="greedy", vad="NoVAD", condition_on_previous_text="NoCondition", data_type="speech", model_size=None)
        plot_param(data_gpu, title="Latency depending on model on 1080TI (GPU)", key='segment_latency',output_path='plots/gpu/koios/', hardware="koios", device="gpu", compute_type="best",backend=None, method="greedy", vad="NoVAD", condition_on_previous_text="NoCondition", data_type="speech", model_size=None)
    
        plot_param(data_gpu, title="VRAM usage depending on precision on 1080TI (GPU) for faster-whisper", key='max_vram',output_path='plots/gpu/koios/faster', hardware="koios", device="gpu", backend="faster", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="VRAM usage depending on precision on 1080TI (GPU) for whisper-timestamped", key='max_vram',output_path='plots/gpu/koios/timestamped', hardware="koios", device="gpu", backend="timestamped", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', condition_on_previous_text="NoCondition", data_type="speech")
        
        plot_param(data_gpu, title="Last confirmed word latency depending on segments (GPU)", key='segment_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")
        plot_param(data_gpu, title="First confirmed word latency depending on segments (GPU)", key='segment_start_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")
        plot_param(data_gpu, title="Last buffered word latency depending on segments (GPU)", key='segment_buffer_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")
        plot_param(data_gpu, title="First buffered word latency depending on segments (GPU)", key='segment_start_buffer_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/koios', hardware="koios", device="gpu", backend=None, buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")


        data_cpu = search_rows_by_key(data, 'device', 'cpu')
        plot_param(data_cpu, title="Latency depending on precision on CPU for faster-whisper", key='segment_latency', output_path='plots/cpu/biggerboi/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
        plot_param(data_cpu, title="Latency depending on precision on CPU for whisper-timestamped", key='segment_latency',output_path='plots/cpu/biggerboi/timestamped', hardware="biggerboi", device="cpu", backend="timestamped", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
        plot_param(data_cpu, title="Latency depending on number of threads on CPU for faster-whisper", key='segment_latency', output_path='plots/cpu/biggerboi/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads=None, compute_type="int8")
        plot_param(data_cpu, title="Latency depending on number of threads on CPU for whisper-timestamped", key='segment_latency',output_path='plots/cpu/biggerboi/timestamped', hardware="biggerboi", device="cpu", backend="timestamped", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads=None, compute_type="float32")
        plot_param(data_cpu, title="Latency depending on VAD on CPU", key='segment_latency', output_path='plots/cpu/biggerboi', hardware="biggerboi", device="cpu", method="greedy", compute_type="best", condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t")
        plot_param(data_cpu, title="Latency depending on VAD on CPU for faster-whisper", key='segment_latency', output_path='plots/cpu/biggerboi/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t")

        plot_param(data_cpu, title="RAM usage depending on precision on CPU for faster-whisper", key='max_vram',output_path='plots/cpu/biggerboi/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", vad='NoVAD', ylabel="RAM usage [MB]", plot_data_mode='max', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
        plot_param(data_cpu, title="RAM usage depending on precision on CPU for whisper-timestamped", key='max_vram',output_path='plots/cpu/biggerboi/timestamped', hardware="biggerboi", device="cpu", backend="timestamped", method="greedy", vad='NoVAD', ylabel="RAM usage [MB]", plot_data_mode='max', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")

        plot_param(data_cpu, title="Latency depending on number of threads on CPU", key='segment_latency', output_path='plots/cpu/lenovo', hardware="lenovo", device="cpu", backend=None, method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads=None, compute_type="best")
        plot_param(data_cpu, title="Latency depending on hardware and backend", key='segment_latency', output_path='plots/cpu/', hardware=None, device="cpu", backend=None, method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t", compute_type="best")

        
        combined_data = data
        plot_param(combined_data, title="Latency depending on device and backend", key='segment_latency', output_path='plots/', hardware=None, device=None, backend=None, method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t", compute_type="best")
        plot_param(combined_data, title="RTF depending on device and backend", key='segment_processing_time', ylabel="Processing time/duration", plot_data_mode='rtf', output_path='plots/', hardware=None, device=None, backend=None, method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t", compute_type="best")
        # plot_param(combined_data, title="Memory usage depending on device and backend", key='max_vram', ylabel="RAM/VRAM usage [MB]", plot_data_mode='max', output_path='plots/', hardware=None, device=None, backend=None, method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t", compute_type="best")

        plot_param(data_gpu, title="Latency depending on hardware", key='segment_latency', output_path='plots/gpu/', hardware=None, device="gpu", backend=None, method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech", compute_type="best", offline="streaming", model_size="large")
        plot_param(data_gpu, title="Latency depending on precision on 4090 Laptop (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/lenovo/faster', hardware="lenovo", device="gpu", backend="faster", method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech")
        plot_param(data_gpu, title="VRAM usage depending on precision on 4090 Laptop (GPU) for faster-whisper", key='max_vram',output_path='plots/gpu/lenovo/faster', hardware="lenovo", device="gpu", backend="faster", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', condition_on_previous_text="NoCondition", data_type="speech")
        
        plot_param(data_gpu, title="Latency depending on method on 4090", key='segment_latency', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend=None, method=None, vad="VAD", condition_on_previous_text="NoCondition", data_type="speech", compute_type="best", offline="streaming", model_size="large")
        plot_param(data_gpu, title="Latency depending on method and VAD on 4090 Laptop (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/lenovo/faster', hardware="lenovo", device="gpu", backend="faster", method=None, vad=None, condition_on_previous_text="NoCondition", data_type="speech", compute_type="best")

        plot_param(data_gpu, title="Latency depending on VAD on 4090", key='segment_latency', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend=None, method="greedy", vad=None, condition_on_previous_text="NoCondition", data_type="speech", compute_type="best", offline="streaming", model_size="large")
        plot_param(data_gpu, title="Latency depending on VAD on 4090 on silence data", key='segment_latency', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend=None, method="greedy", vad=None, condition_on_previous_text="NoCondition", data_type="silence", compute_type="best", offline="streaming", model_size="large")
        
        plot_param(data_gpu, title="Last confirmed word latency depending on segments (GPU)", key='segment_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend="faster", buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")
        plot_param(data_gpu, title="First confirmed word latency depending on segments (GPU)", key='segment_start_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend="faster", buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")
        plot_param(data_gpu, title="Last buffered word latency depending on segments (GPU)", key='segment_buffer_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend="faster", buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")
        plot_param(data_gpu, title="First buffered word latency depending on segments (GPU)", key='segment_start_buffer_latency', ylabel="Latency [s]", plot_data_mode='all', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend="faster", buffer_trimming=None, min_chunk_size=None, method="greedy", vad="NoVAD",condition_on_previous_text="NoCondition", data_type="speech", compute_type="int8")

        # plot_param(data_gpu, title="Latency depending Previous text on 4090", key='segment_latency', output_path='plots/gpu/lenovo', hardware="lenovo", device="gpu", backend=None, method="greedy", vad="NoVAD", condition_on_previous_text=None, compute_type="best", data_type="speech")


def plot_param(data, title="Latency", key='segment_latency', output_path='plots', ylabel='Latency [s]', plot_data_mode='all', \
    hardware=None, device=None, backend=None, compute_type=None, method=None, vad=None, condition_on_previous_text=None, data_type=None, \
    cpu_threads=None, offline="streaming", model_size="large-v3", buffer_trimming="15", min_chunk_size="2"):
    os.makedirs(output_path, exist_ok=True)
    description = [f"hardware: {hardware}" if hardware is not None else ""]
    description.append(f"device: {device}" if device is not None else "")
    description.append(f"backend: {backend}" if backend is not None else "")
    description.append(f"compute_type: {compute_type}" if compute_type is not None else "")
    description.append(f"method: {method}" if method is not None else "")
    description.append(f"vad: {vad}" if vad is not None else "")
    description.append(f"data_type: {data_type}" if data_type is not None else "")
    if device == "cpu":
        description.append(f"cpu_threads: {cpu_threads}" if cpu_threads is not None else "")
    if condition_on_previous_text is not None:
        description.append(f"condition_on_previous_text: False" if condition_on_previous_text=="NoCondition" else "condition_on_previous_text: True")
    description.append(f"{offline}" if offline is not None else "")
    description.append(f"model_size: {model_size}" if model_size is not None else "")
    description.append(f"buffer_trimming: {buffer_trimming}" if buffer_trimming is not None else "")
    description.append(f"min_chunk_size: {min_chunk_size}" if min_chunk_size is not None else "")
    description = [x for x in description if x != ""]
    if len(description) > 3:
        description[3] = "\n" + description[3]
    if len(description) > 6:
        description[6] = "\n" + description[6]
    description = ', '.join(description)
    # plt.rcParams["figure.figsize"] = (12,10)
    data = search_rows(data, hardware=hardware, device=device, backend=backend, compute_type=compute_type, min_chunk_size=min_chunk_size, \
        method=method, vad=vad, condition_on_previous_text=condition_on_previous_text, buffer_trimming=buffer_trimming, \
        data_type=data_type, cpu_threads=cpu_threads, offline=offline, model_size=model_size)
    if len(data) == 0:
        print(f"No data for '{title}' with : {description}".replace("\n", ""))
        return
    fig, ax = plt.subplots()    
    plot_values= []
    plot_names = []
    longest_name = ""
    for row in data:
        m = get_values(row, key=key, mode=plot_data_mode)
        plot_values.append(m)
        name=""
        name += f"{row['hardware']}_" if hardware is None else ""
        name += f"{row['device']}_" if device is None else ""
        name += f"{row['backend']}_" if backend is None else "" 
        name += f"{row['compute_type']}_" if compute_type is None else ""
        name += f"{row['method']}_" if method is None else ""
        name += f"{row['vad']}_" if vad is None else ""
        name += f"{row['condition_on_previous_text']}_" if condition_on_previous_text is None else ""
        name += f"{row['data_type']}_" if data_type is None else ""
        model = row['model_size'].replace('model-','').replace('whisper-','')
        if len(model) > 20:
            model = model.split("-",1)[1]
        name += f"{model}_" if model_size is None else ""
        name += f"{row['offline']}_" if offline is None else ""
        name += f"buffer-trimming:{row['buffer_trimming']}_" if buffer_trimming is None else ""
        name += f"min-chunk-size:{row['min_chunk_size']}_" if min_chunk_size is None else ""
        if device == "cpu":
            name += f"{row['cpu_threads']}_" if cpu_threads is None else ""
        name = name[:-1]
        if len(name) > len(longest_name):
            longest_name = name
        if len(name) >= 40:
            name = name.split("_")
            mid = len(name)//2
            name = "_".join(name[:mid]) + "_\n" + "_".join(name[mid:])
        plot_names.append(name)
    if plot_data_mode == 'max':
        ax.bar(plot_names, plot_values)
        plt.ylim([0, max(plot_values)*1.1])
        ax.set_xticks([y for y in range(len(plot_names))], labels=plot_names)
        # write values in bar
        for i, v in enumerate(plot_values):
            ax.text(i, v, f"{v:.0f}", color='black', ha='center', va='bottom')
    elif plot_data_mode == 'all' or plot_data_mode == 'rtf':
        ax.violinplot(plot_values, showmedians=True, quantiles=[[0.25, 0.75] for i in range(len(plot_values))], showextrema=True)
        ax.set_xticks([y + 1 for y in range(len(plot_names))], labels=plot_names)
    if key=="wer_score":
        ax.set_ylim([0, 100])
    ax.set_xlabel(description)
    if len(longest_name) >= 40:
        plt.xticks(rotation=60)
    else:
        plt.xticks(rotation=25)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # plt.ylim(bottom=0)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(output_path,f'{title}.png'), bbox_inches='tight')
    plt.close()

def get_ram_value(path, verbose=False):
    try:
        with open(os.path.join(path, "ram.txt"), 'r') as f:
            line = f.readline()
            line = line.split(": ")[1]
            line = line.split(" ")[0]
            ram_value = int(line) / 1000
    except Exception as e:
        if verbose:
            print(f"Error reading ram.txt ({path}): {e}")
        ram_value = None
    return ram_value

def load_data(data_path, truth_path):
    hardwares  = os.listdir(data_path)
    data = []
    pbar = tqdm(total=len(hardwares))
    for hardware in hardwares:
        devices = os.listdir(os.path.join(data_path, hardware))
        for device in devices:
            backends = os.listdir(os.path.join(data_path, hardware, device))
            for backend in backends:
                execs = os.listdir(os.path.join(data_path, hardware, device, backend))
                execs = [x for x in execs if not os.path.isfile(os.path.join(data_path, hardware, device, backend, x))]
                # params = [x.split('.')[0] for x in execs]
                params = [x.split('_') for x in execs]
                compute_types = [x[0] for x in params]
                data_types = ["silence" if "silence" in x else "speech" for x in params]
                vads = []
                for x in params:
                    added=False
                    for j in x:
                        if j.startswith('vad'):
                            vads.append(j.upper())
                            added=True
                    if not added:
                        vads.append("NoVAD")
                methods = ["beam-search" if ("beam-search" in x or "beam" in x) else "greedy" for x in params]
                condition_on_previous_text = ["ConditionOnPreviousText" if "previous-text" in x else "NoCondition" for x in params]
                threads = [list({'2t', '4t', '8t', '16t'} & set(x)) for x in params]
                threads = [x[0] if x else "4t" for x in threads]
                bts = [list({'bts-7', 'bts-5', 'bts-6', 'bts-10', 'bts-12', 'bts-15'} & set(x)) for x in params]
                bts = [x[0].split("-")[1] if x else "15" for x in bts]
                mcs = [list({'mcs-0.6', 'mcs-0', 'mcs-2', 'mcs-1', 'mcs-0.8', 'mcs-1.5'} & set(x)) for x in params]
                mcs = [x[0].split("-")[1] if x else "2" for x in mcs]
                model_sizes = []
                for x in params:
                    added=False
                    for j in x:
                        if j.startswith('large') or j.startswith('model') or j=="medium" or j=="small" or j=="tiny":
                            model_sizes.append(j)
                            added=True
                    if not added:
                        model_sizes.append("large-v3")
                offlines = ["offline" if "offline" in x else "streaming" for x in params]
                for i, exec in enumerate(execs):
                    if os.path.exists(os.path.join(data_path, hardware, device, backend, exec, 'result.json')):
                        with open(os.path.join(data_path, hardware, device, backend, exec, 'result.json'), 'r') as f:
                            raw_data = json.load(f)
                            if device == "cpu":
                                for j in raw_data.keys():
                                    raw_data[j]['max_vram'] = get_ram_value(os.path.join(data_path, hardware, device, backend, exec.split('.')[0]))
                            for j in raw_data.keys():
                                file_id = os.path.basename(j).split('.')[0]
                                wer = process_wer(os.path.join(truth_path, file_id + '.txt'), os.path.join(data_path, hardware, device, backend, exec.split('.')[0], 'transcripts', file_id), exec.split('.')[0]+"_"+file_id)
                                raw_data[j]['wer_score'] = wer['wer'] if wer else None
                            data.append({'hardware': hardware,'device': device, "offline": offlines[i], 'buffer_trimming': bts[i], 'min_chunk_size': mcs[i] ,"model_size": model_sizes[i], \
                                'cpu_threads': threads[i], 'data_type':data_types[i], 'condition_on_previous_text': condition_on_previous_text[i], \
                                'backend': "faster" if backend=="faster-whisper" or backend=="faster" else "timestamped", \
                                'compute_type': compute_types[i], 'vad': vads[i], 'method': methods[i], 'data': raw_data})
                    else:
                        print(f"Missing result.json for {os.path.join(data_path, hardware, device, backend, exec.split('.')[0])}")
        pbar.update(1)
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../results/wstart/normal_large')
    # parser.add_argument('--data_path', type=str, default='normal_large')

    parser.add_argument('--ground_truth', type=str, default='../ground_truths')
    parser.add_argument('--wer', action="store_true", default=False, help="Plot WER instead of latencies.")
    args = parser.parse_args()

    data_path = args.data_path

    data = load_data(data_path, args.ground_truth)
    os.makedirs('plots', exist_ok=True)
    plot(data, wer=args.wer)

