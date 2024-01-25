import os
import argparse

ground_truth_source_file = "/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/ACSYNT/text"
ground_truth_folder = "../ground_truths"
os.makedirs(ground_truth_folder, exist_ok=True)

def load_data(data_path):
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
            get_truth_from_source(ground_truth_source_file, files.keys())
            break
    ground_truth_files = os.listdir(ground_truth_folder)
    ground_truth_files = [x.split('.')[0] for x in ground_truth_files]
    for i in ground_truth_files:
        if files.get(i) is None:
            continue
        with open(os.path.join(ground_truth_folder, i+'.txt'), 'r') as f:
            lines = f.readlines()
            txt = ' '.join(lines)
        files[i]['ground_truth'] = txt
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

def process_wer(ref, hyp):
    pred=load_prediction(hyp)
    

def get_truth_from_source(source, files):
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
                    with open(os.path.join(ground_truth_folder, id+'.txt'), 'a') as f:
                        f.write(' ' + line.split(' ', 1)[1])
                else:
                    with open(os.path.join(ground_truth_folder, id+'.txt'), 'w') as f:
                        f.write(line.split(' ', 1)[1])
                    truth_files.append(id)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    args = parser.parse_args()

    data_path = args.data_path

    data = load_data(data_path)
    os.makedirs('wer', exist_ok=True)
    for i in data.keys():
        for j in data[i].keys():
            if j == 'ground_truth':
                continue
            process_wer(data[i]['ground_truth'], data[i][j])

