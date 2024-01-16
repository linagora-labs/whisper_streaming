import json
import argparse
import os

# plot data
import matplotlib.pyplot as plt
import numpy as np


def search_row(data, hardware, device, backend, compute_type, vad, method):
    for row in data:
        if row['hardware'] == hardware and row['device'] == device and row['backend'] == backend and row['compute_type'] == compute_type and row['vad'] == vad and row['method'] == method:
            return row
    return None

def search_rows(data, hardware, device, backend, compute_type, vad, method):
    rows = []
    for row in data:
        if row['hardware'] == hardware and row['device'] == device and row['backend'] == backend and row['compute_type'] == compute_type and row['vad'] == vad and row['method'] == method:
            rows.append(row)
    return rows

def plot_processting_times_of_files(row):
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
    plt.savefig('processing_times_per_file.png')
    plt.close()

def get_stats(row):
    data_names = row['data'].keys()
    plot_mean= []
    plot_std = []
    for j in data_names:
        lv = row['data'][j]
        plot_mean.append(np.mean(lv['segments_processing_times']))
        plot_std.append(np.std(lv['segments_processing_times']))
    return np.mean(plot_mean), np.std(plot_std)

def plot_processting_times_per_params(data):
    fig, ax = plt.subplots()
    plot_mean= []
    plot_std = []
    plot_names = []
    for row in data:
        m, std = get_stats(row)
        plot_mean.append(m)
        plot_std.append(std)
        plot_names.append(row['device']+'_'+row['compute_type'] + '_' + row['method'] + '_' + str(row['vad']))
    #plot mean and std for each file
    ax.bar(plot_names, plot_mean)
    ax.set_xlabel('params')
    plt.xticks(rotation=40)
    ax.set_ylabel('processing time [s]')
    ax.set_title('processing time')
    plt.savefig('processing_times_per_params.png')
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
                params = [x.split('.')[0] for x in execs]
                params = [x.split('_') for x in params]
                compute_types = [x[0] for x in params]
                vads = [True if len(x)>1 and x[1]=="vad" else False for x in params]
                methods = [x if len(x)>2 or (len(x)>1 and x[1]=="greedy") else "beam_search" for x in params]
                for i, exec in enumerate(execs):
                    with open(os.path.join(data_path, hardware, device, backend, exec.split('.')[0], 'processing_times.json'), 'r') as f:
                        raw_data = json.load(f)
                        data.append({'hardware': hardware,'device': device, 'backend': backend, 'compute_type': compute_types[i], 'vad': vads[i], 'method': methods[i], 
                                        'data': raw_data})

    plot_processting_times_of_files(search_row(data, 'koios', 'gpu', 'faster', 'int8', False, 'beam_search'))
    plot_processting_times_per_params(data)

