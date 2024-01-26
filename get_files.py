from linastt.utils.dataset import kaldi_folder_to_dataset
from linastt.utils.audio import load_audio, save_audio
import os

TCOF_PATH = '/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/TCOF_Adultes'
TCOF_FILES_PATH_TO_GET = ['/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/TCOF/tcof/12/Corpus/Adultes/Conversations/voyage_con_15/voyage_con_15.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/TCOF/tcof/12/Corpus/Adultes/Conversations/voyage_hab_14/voyage_hab_14.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/TCOF/tcof/12/Corpus/Adultes/Conversations/voyages_ric_06/voyages_ric_06.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/TCOF/tcof/12/Corpus/Adultes/Conversations/voyage_mel_15/voyage_mel_15.wav']
TCOF_TARGET = '/home/abert/abert/data-fr/normal/tcof'

ETAPE_PATH = '/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/ETAPE'
ETAPE_FILES_PATH_TO_GET = ['/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/ETAPE/elda-etape-rev00/ETAPE/DATA/TRAIN/FLAC/LCP_PileEtFace_2010-11-21_060400.flac',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/ETAPE/elda-etape-rev00/ETAPE/DATA/TRAIN/FLAC/LCP_PileEtFace_2011-02-13_055700.flac',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/ETAPE/elda-etape-rev00/ETAPE/DATA/TRAIN/FLAC/LCP_TopQuestions_2010-10-26_213800.flac',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/ETAPE/elda-etape-rev00/ETAPE/DATA/TRAIN/FLAC/LCP_TopQuestions_2010-11-02_213800.flac']
ETAPE_TARGET = '/home/abert/abert/data-fr/normal/etape'

CFPP_PATH = '/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/CFPP2000'
CFPP_FILES_PATH_TO_GET = ['/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/CFPP2000/CFPP2000/record-21/Yvette_Audin_F_70_7e.mp3',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/CFPP2000/CFPP2000/record-36/Younes_Belkacem_H_59_Mo.mp3',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/CFPP2000/CFPP2000/record-7/Rosier_Bernard_H_60_Rosier_Micheline_58.mp3',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/OTHERS/CFPP2000/CFPP2000/record-35/Youcef_Zerari_H_29_Abdel_Hachim_H_25_SO.mp3']
CFPP_TARGET = '/home/abert/abert/data-fr/normal/cfpp'

def get_files_from_segments(kaldi_path, files_path_to_get, output_dir):
    # remove things in output_dir
    os.makedirs(output_dir, exist_ok=True)
    for x in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, x))
    meta, dataset = kaldi_folder_to_dataset(kaldi_path, do_cache=False)
    dataset = [x for x in dataset if x['path'] in files_path_to_get]
    dataset = sorted(dataset, key=lambda x: (x['path'], x['end']))
    for x in files_path_to_get:
        sub_dataset = [j for j in dataset if j['path']==x]
        text = []
        end_time = -1
        for y in sub_dataset:
            text.append(y['text'])
            if y['end'] > 180:
                end_time = y['end']
                break
        annot = ' '.join(text)
        annot = annot.replace('  ', ' ')
        base_file_name = os.path.basename(x)
        with open(os.path.join(output_dir, base_file_name.replace('.wav', '.txt')), 'w') as f:
            f.write(annot)
        audio = load_audio(x, start=0, end=end_time, sample_rate=16000)
        save_audio(os.path.join(output_dir, base_file_name), audio, 16000)
        print('Done', base_file_name)


if __name__=="__main__":
    get_files_from_segments(TCOF_PATH, TCOF_FILES_PATH_TO_GET, TCOF_TARGET)
    get_files_from_segments(ETAPE_PATH, ETAPE_FILES_PATH_TO_GET, ETAPE_TARGET)
    get_files_from_segments(CFPP_PATH, CFPP_FILES_PATH_TO_GET, CFPP_TARGET)
