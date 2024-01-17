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

def search_rows(data, hardware=None, device=None, backend=None, compute_type=None, vad=None, method=None):
    rows = []
    for row in data:
        if (hardware is None or row['hardware'] == hardware) and (device is None or row['device'] == device) and (backend is None or row['backend'] == backend) and (compute_type is None or row['compute_type'] == compute_type) and (vad is None or row['vad'] == vad) and (method is None or row['method'] == method):
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
        plot_mean.append(np.mean(lv['segments_processing_times']))
        plot_std.append(np.std(lv['segments_processing_times']))
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
    plt.savefig(os.path.join(output_path,'processing_times_per_file.png'))
    plt.close()

def get_stats(row):
    data_names = row['data'].keys()
    plot_mean= []
    plot_std = []
    for j in data_names:
        lv = row['data'][j]
        plot_mean.append(np.sum(lv['latencies']))
        plot_std.append(np.sum(lv['segments_duration']))
    return np.concatenate(plot_mean), 0


def plot(data):
    data_gpu = search_rows_by_key(data, 'device', 'gpu')
    # plot_processting_times_per_params(data_gpu, "GPU processing times", output_path='plots/gpu')
    # data_cpu = search_rows(data, 'device', 'cpu')
    # plot_processting_times_per_params(data_cpu, "CPU processing times" ,output_path='plots/cpu')
    plot_param(data_gpu, param="vad", title="VAD GPU processing times", output_path='plots/gpu')

def plot_param(data, param="vad", title="Processing times", output_path='plots'):
    os.makedirs(output_path, exist_ok=True)
    # plt.rcParams["figure.figsize"] = (12,10)
    data = search_rows(data, compute_type="fp16", hardware="koios", device="gpu")
    fig, ax = plt.subplots()    
    plot_mean= []
    plot_std = []
    plot_names = []
    plot_labels = []
    for row in data:
        m, std = get_stats(row)
        plot_mean.append(m)
        plot_std.append(std)
        plot_names.append(f"{row['backend']}_{row['compute_type']}_\n{row['method']}_{'VAD' if row['vad'] else 'NoVAD'}")
        plot_labels.append('VAD' if row[param] else 'NoVAD')
    for i, label in enumerate(list(set(plot_labels))):
        lplot_mean= []
        lplot_names = []
        lplot_colors = []
        for j in range(len(plot_labels)):
            if plot_labels[j] == label:
                lplot_mean.append(plot_mean[j])
                lplot_names.append(plot_names[j])
                lplot_colors.append(COLORS_DICT[i])
        ax.bar(lplot_names, lplot_mean, color=lplot_colors, label=label)
    ax.set_xlabel('params')
    plt.xticks(rotation=25)
    ax.set_ylabel('processing time [s]')
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(output_path,f'{title}_{param}.png'), bbox_inches='tight')
    plt.close()

def plot_processting_times_per_params(data, title="Processing times", output_path='plots'):
    os.makedirs(output_path, exist_ok=True)
    fig, ax = plt.subplots(nrows=2, ncols=2)    
    plot_mean= []
    plot_std = []
    plot_names = []
    for row in data:
        m, std = get_stats(row)
        plot_mean.append(m)
        plot_std.append(std)
        plot_names.append(f"{row['hardware']}_{row['compute_type']}_{row['method']}_{row['vad']}")
    #plot mean and std for each file
    ax.bar(plot_names, plot_mean)
    ax.set_xlabel('params')
    plt.xticks(rotation=40)
    ax.set_ylabel('processing time [s]')
    ax.set_title(title)
    plt.savefig(os.path.join(output_path,f'{title}.png'))
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    args = parser.parse_args()

    data_path = args.data_path

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
                vads = [True if len(x)>1 and x[1]=="vad" else False for x in params]
                methods = ["greedy" if len(x)>2 or (len(x)>1 and x[1]=="greedy") else "beam_search" for x in params]
                for i, exec in enumerate(execs):
                    with open(os.path.join(data_path, hardware, device, backend, exec.split('.')[0], 'processing_times.json'), 'r') as f:
                        raw_data = json.load(f)
                        data.append({'hardware': hardware,'device': device, 'backend': backend, 'compute_type': compute_types[i], 'vad': vads[i], 'method': methods[i], 
                                        'data': raw_data})
    os.makedirs('plots', exist_ok=True)
    plot_processting_times_of_files(search_row(data, 'koios', 'gpu', 'faster', 'int8', False, 'beam_search'))
    plot(data)

