import os
import argparse
from linastt.utils.wer import compute_wer, plot_wer
import numpy as np

def load_data(data_path, ground_truth_source_file, ground_truth_folder):
    hardwares  = os.listdir(data_path)
    files = {}
    for hardware in hardwares:
        devices = os.listdir(os.path.join(data_path, hardware))
        for device in devices:
            backends = os.listdir(os.path.join(data_path, hardware, device))
            for backend in backends:
                execs = os.listdir(os.path.join(data_path, hardware, device, backend))
                execs = [x for x in execs if not os.path.isfile(os.path.join(data_path, hardware, device, backend, x))]
                for exec in execs:
                    transcripts_folder_path = os.path.join(data_path, hardware, device, backend, exec, 'transcripts')
                    transcript_files = os.listdir(transcripts_folder_path)
                    for k in transcript_files:
                        k=k.split('.')[0]
                        if files.get(k) is None:
                            files[k] = {}
                        files[k][backend+'_'+exec] = os.path.join(transcripts_folder_path, k)
    ground_truth_file_paths = os.listdir(ground_truth_folder)
    for i in files.keys():
        if i+".txt" not in ground_truth_file_paths:
            print(i)
            get_truth_from_source(ground_truth_source_file, ground_truth_folder, files.keys())
            break
    ground_truth_files = os.listdir(ground_truth_folder)
    ground_truth_files = [x.split('.')[0] for x in ground_truth_files]
    for i in ground_truth_files:
        if files.get(i) is None:
            continue
        files[i]['ground_truth'] = os.path.join(ground_truth_folder, i + '.txt')
    return files


def load_prediction(file_path):
    with open(file_path+".txt", 'r') as f:
        line = f.readline()
        line = line.strip()
        if line.startswith("(None, None, '')"):
            print('Empty prediction')
            return ''
        else:
            line = line.strip()
            pred = line.split(' ', 3)[3][1:]
    return pred

def load_truth(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        txt = ' '.join(lines)
    return txt

def process_wer(name, ref_file, pred_file, verbose=False):
    pred = load_prediction(pred_file)
    ref = load_truth(ref_file)
    # compute wer between ref and pred
    wer_score = compute_wer([ref], [pred], normalization="fr")
    if verbose:
        print(f"{name} WER: {wer_score['wer']:.2f}")
    return wer_score
    

def get_truth_from_source(source, target, files):
    truth_files = []
    with open(source, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            id = line.split(' ')[0]
            id = id.split('_')[0]
            if id not in files:
                continue
            else:
                if id in truth_files:
                    with open(os.path.join(target, id+'.txt'), 'a') as f:
                        f.write(' ' + line.split(' ', 1)[1])
                else:
                    with open(os.path.join(target, id+'.txt'), 'w') as f:
                        f.write(line.split(' ', 1)[1])
                    truth_files.append(id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../normal_wer_')
    parser.add_argument('--source_truth_folder', type=str, default="/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/ACSYNT/text")
    parser.add_argument('--truth_folder', type=str, default="../ground_truths")
    args = parser.parse_args()

    data_path = args.data_path
    source_truth_folder = args.source_truth_folder
    truth_folder = args.truth_folder

    os.makedirs(truth_folder, exist_ok=True)

    data = load_data(data_path, source_truth_folder, truth_folder)
    os.makedirs('wer', exist_ok=True)

    config_to_test = ['faster_int8_greedy_offline']

    wer_list = []
    for test in config_to_test:
        for i in data.keys():
            wer_list.append(process_wer(i, data[i]['ground_truth'], data[i][test]))

    wer_score_list = [x['wer'] for x in wer_list]
    print(f"Number of files: {len(wer_score_list)}")
    print(f"Mean WER: {sum(wer_score_list)/len(wer_score_list):.2f}")
    print(f"Std WER: {np.std(wer_score_list):.2f}")
    print(f"Median WER: {sorted(wer_score_list)[len(wer_score_list)//2]:.2f}")
    print(f"Min WER: {min(wer_score_list):.2f}")
    print(f"Max WER: {max(wer_score_list):.2f}")
    
    # plot_wer(wer_list)

