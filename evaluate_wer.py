import os
import argparse

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
    ground_truths_file = "/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/ACSYNT/text"
    with open(ground_truths_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            id = line.split(' ')[0]
            id = id.split('_')[0]
            if files.get(id) is None:
                continue
            if files[id].get('ground_truth') is None:
                files[id]['ground_truth'] = line.split(' ', 1)[1]
            else:
                files[id]['ground_truth'] += ' ' + line.split(' ', 1)[1]
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

