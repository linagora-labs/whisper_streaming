import json
import argparse
import os

# plot data
import matplotlib.pyplot as plt
import numpy as np


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

def search_rows(data, hardware=None, device=None, backend=None, compute_type=None, vad=None, method=None, condition_on_previous_text=None, data_type=None, cpu_threads=None):
    rows = []
    for row in data:
        if (hardware is None or row['hardware'] == hardware) and (cpu_threads is None or row['cpu_threads']==cpu_threads) and (data_type is None or row['data_type']==data_type) and (condition_on_previous_text is None or row['condition_on_previous_text']==condition_on_previous_text) and (device is None or row['device'] == device) and (backend is None or row['backend'] == backend) and (compute_type is None or row['compute_type'] == compute_type or (compute_type=="best" and row['compute_type']=="int8" and row['backend']=="faster") or (compute_type=="best" and row['compute_type']=="float32" and row['backend']=="timestamped")) and (vad is None or row['vad'] == vad) and (method is None or row['method'] == method):
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
    for j in data_names:
        lv = row['data'][j]
        if isinstance(lv[key], list):
            values.extend(lv[key])
        else:
            values.append(lv[key])
    if mode=="max":
        return max(values)
    elif mode=="min":
        return min(values)
    return values

def plot(data):
    data_gpu = search_rows_by_key(data, 'device', 'gpu')
    # plot_processting_times_per_params(data_gpu, "GPU processing times", output_path='plots/gpu')

    plot_param(data_gpu, title="Latency depending on precision on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/faster', hardware="koios", device="gpu", backend="faster", method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="Latency depending on precision on 1080TI (GPU) for whisper-timestamped", key='segment_latency',output_path='plots/gpu/timestamped', hardware="koios", device="gpu", backend="timestamped", method="greedy", vad='NoVAD', condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="VRAM usage depending on precision on 1080TI (GPU) for faster-whisper", key='max_vram',output_path='plots/gpu/faster', hardware="koios", device="gpu", backend="faster", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="VRAM usage depending on precision on 1080TI (GPU) for whisper-timestamped", key='max_vram',output_path='plots/gpu/timestamped', hardware="koios", device="gpu", backend="timestamped", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="Latency depending on VAD on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu', hardware="koios", device="gpu", method="greedy", compute_type="best", condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="Latency depending on VAD on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/faster', hardware="koios", device="gpu", backend="faster", method="greedy", condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="Latency depending on VAD for silence data on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu', hardware="koios", device="gpu", method="greedy", compute_type="best", condition_on_previous_text="NoCondition", data_type="silence")
    plot_param(data_gpu, title="Latency depending on VAD for silence data on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/faster', hardware="koios", device="gpu", backend="faster", method="greedy", condition_on_previous_text="NoCondition", data_type="silence")
    plot_param(data_gpu, title="Latency depending on method on 1080TI (GPU)", key='segment_latency', output_path='plots/gpu', hardware="koios", device="gpu", vad='VAD', compute_type="best", condition_on_previous_text="NoCondition", data_type="speech")
    plot_param(data_gpu, title="Latency depending on method on 1080TI (GPU) for faster-whisper", key='segment_latency', output_path='plots/gpu/faster', hardware="koios", device="gpu", backend="faster", compute_type="best", condition_on_previous_text="NoCondition", data_type="speech")
    

    data_cpu = search_rows_by_key(data, 'device', 'cpu')
    plot_param(data_cpu, title="Latency depending on precision on CPU for faster-whisper", key='segment_latency', output_path='plots/cpu/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
    plot_param(data_cpu, title="Latency depending on precision on CPU for whisper-timestamped", key='segment_latency',output_path='plots/cpu/timestamped', hardware="biggerboi", device="cpu", backend="timestamped", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
    plot_param(data_cpu, title="Latency depending on number of threads on CPU for faster-whisper", key='segment_latency', output_path='plots/cpu/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads=None, compute_type="int8")
    plot_param(data_cpu, title="Latency depending on number of threads on CPU for whisper-timestamped", key='segment_latency',output_path='plots/cpu/timestamped', hardware="biggerboi", device="cpu", backend="timestamped", method="greedy", vad='NoVAD', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads=None, compute_type="float32")
    plot_param(data_cpu, title="RAM usage depending on precision on CPU for faster-whisper", key='max_vram',output_path='plots/cpu/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
    plot_param(data_cpu, title="RAM usage depending on precision on CPU for whisper-timestamped", key='max_vram',output_path='plots/cpu/timestamped', hardware="biggerboi", device="cpu", backend="timestamped", method="greedy", vad='NoVAD', ylabel="VRAM usage [MB]", plot_data_mode='max', data_type="speech", condition_on_previous_text="NoCondition", cpu_threads="4t")
    plot_param(data_cpu, title="Latency depending on VAD on CPU", key='segment_latency', output_path='plots/cpu', hardware="biggerboi", device="cpu", method="greedy", compute_type="best", condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t")
    plot_param(data_cpu, title="Latency depending on VAD on CPU for faster-whisper", key='segment_latency', output_path='plots/cpu/faster', hardware="biggerboi", device="cpu", backend="faster", method="greedy", condition_on_previous_text="NoCondition", data_type="speech", cpu_threads="4t")


def plot_param(data, title="Latency", key='segment_latency', output_path='plots', ylabel='Latency [s]', plot_data_mode='all', hardware=None, device=None, backend=None, compute_type=None, method=None, vad=None, condition_on_previous_text=None, data_type=None, cpu_threads=None):
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
    description = [x for x in description if x != ""]
    if len(description) > 4:
        description[4] = "\n" + description[4]
    description = ', '.join(description)
    # plt.rcParams["figure.figsize"] = (12,10)
    data = search_rows(data, hardware=hardware, device=device, backend=backend, compute_type=compute_type, method=method, vad=vad, condition_on_previous_text=condition_on_previous_text, data_type=data_type, cpu_threads=cpu_threads)
    if len(data) == 0:
        print(f"No data for {description}")
        return
    fig, ax = plt.subplots()    
    plot_values= []
    plot_names = []
    for row in data:
        m = get_values(row, key=key, mode=plot_data_mode)
        plot_values.append(m)
        name = f"{row['backend']}_" if backend is None else "" 
        name += f"{row['compute_type']}_" if compute_type is None else ""
        name += f"{row['method']}_" if method is None else ""
        name += f"{row['vad']}_" if vad is None else ""
        name += f"{row['condition_on_previous_text']}_" if condition_on_previous_text is None else ""
        name += f"{row['data_type']}_" if data_type is None else ""
        if device == "cpu":
            name += f"{row['cpu_threads']}_" if cpu_threads is None else ""
        name = name[:-1]
        plot_names.append(name)
    if plot_data_mode == 'all':
        ax.violinplot(plot_values, showmedians=True, quantiles=[[0.25, 0.75] for i in range(len(plot_values))], showextrema=False)
        ax.set_xticks([y + 1 for y in range(len(plot_names))], labels=plot_names)

    elif plot_data_mode == 'max':
        ax.bar(plot_names, plot_values)
        plt.ylim([0, max(plot_values)*1.1])
        ax.set_xticks([y for y in range(len(plot_names))], labels=plot_names)
        # write values in bar
        for i, v in enumerate(plot_values):
            ax.text(i, v, f"{v:.0f}", color='black', ha='center', va='bottom')
    ax.set_xlabel(description)
    plt.xticks(rotation=25)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # plt.ylim(bottom=0)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(output_path,f'{title}.png'), bbox_inches='tight')
    plt.close()

def get_ram_value(path):
    with open(os.path.join(path, "ram.txt"), 'r') as f:
        line = f.readline()
        line = line.split(": ")[1]
        line = line.split(" ")[0]
        ram_value = int(line) / 1000
    return ram_value

def load_data(data_path):
    hardwares  = os.listdir(data_path)
    data = []
    for hardware in hardwares:
        devices = os.listdir(os.path.join(data_path, hardware))
        for device in devices:
            backends = os.listdir(os.path.join(data_path, hardware, device))
            for backend in backends:
                execs = os.listdir(os.path.join(data_path, hardware, device, backend))
                execs = [x for x in execs if not os.path.isfile(os.path.join(data_path, hardware, device, backend, x))]
                params = [x.split('.')[0] for x in execs]
                params = [x.split('_') for x in params]
                compute_types = [x[0] for x in params]
                data_types = ["silence" if "silence" in x else "speech" for x in params]
                vads = ["VAD" if "vad" in x else "NoVAD" for x in params]
                methods = ["beam_search" if "beam" in x  else "greedy" for x in params]
                condition_on_previous_text = ["ConditionOnPreviousText" if "previous-text" in x else "NoCondition" for x in params]
                threads = [list({'2t', '4t', '8t', '16t'} & set(x)) for x in params]
                for i, exec in enumerate(execs):
                    if os.path.exists(os.path.join(data_path, hardware, device, backend, exec.split('.')[0], 'result.json')):
                        with open(os.path.join(data_path, hardware, device, backend, exec.split('.')[0], 'result.json'), 'r') as f:
                            raw_data = json.load(f)
                            if device == "cpu":
                                for j in raw_data.keys():
                                    raw_data[j]['max_vram'] = get_ram_value(os.path.join(data_path, hardware, device, backend, exec.split('.')[0]))
                            data.append({'hardware': hardware,'device': device, 'cpu_threads': threads[i][0] if threads[i] else '4t', 'data_type':data_types[i], 'condition_on_previous_text': condition_on_previous_text[i], 'backend': backend, 'compute_type': compute_types[i], 'vad': vads[i], 'method': methods[i], 
                                            'data': raw_data})
                    else:
                        print(f"Missing result.json for {os.path.join(data_path, hardware, device, backend, exec.split('.')[0])}")
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../faster_n_openai/normal_large')
    args = parser.parse_args()

    data_path = args.data_path

    data = load_data(data_path)
    
    os.makedirs('plots', exist_ok=True)
    plot(data)

