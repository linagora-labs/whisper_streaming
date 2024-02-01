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

ACSYNT_PATH = '/media/nas/CORPUS_PENDING/kaldi/Corpus_FR/normalized/ACSYNT'
ACSYNT_FILES_PATH_TO_GET = ['/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC23_ROM/ROME1.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC23_ROM/ROME5.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC23_ROM/ROME12.wav', 
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC22_RIO/RIOE1.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC22_RIO/RIOE5.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC22_RIO/RIOE12.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC18_NES/NESE1.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC18_NES/NESE2.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC18_NES/NESE5.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC17_MAG/MAGE1.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC17_MAG/MAGE2.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC16_LAU/LAUE9.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC15_LAD/LADE10.wav', 
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC15_LAD/LADE4.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC13_JAS/JASE4.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC11_GOL/GOLE7.wav',
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC11_GOL/GOLE10.wav', 
'/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR//OTHERS/ACSYNT/FinalAcsynt/meeting/LOC4_BOA/BOAE4.wav']
ACSYNT_TARGET = '/home/abert/abert/data-fr/normal/acsynt'


def get_files_from_segments(kaldi_path, files_path_to_get, output_dir, time_limit=180, force=False, search_above=False):
    # remove things in output_dir
    os.makedirs(output_dir, exist_ok=True)
    if force:
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
            if time_limit > 0 and y['end'] > time_limit:
                end_time = y['end']
                break
        annot = ' '.join(text)
        annot = annot.replace('  ', ' ')
        base_file_name = os.path.basename(x)
        annot_file = os.path.join(output_dir, base_file_name.replace('.wav', '.txt'))
        annot_file = os.path.join(output_dir, annot_file.replace('.flac', '.txt'))
        annot_file = os.path.join(output_dir, annot_file.replace('.mp3', '.txt'))
        with open(annot_file, 'w') as f:
            f.write(annot)
        audio = load_audio(x, start=0, end=end_time if time_limit > 0 else None, sample_rate=16000)
        if search_above:
            if os.path.exists(os.path.join(output_dir, "../", base_file_name)):
                print('Already exists', base_file_name)
                continue
        save_audio(os.path.join(output_dir, base_file_name), audio, 16000)
        print('Done', base_file_name)


if __name__=="__main__":
    get_files_from_segments(TCOF_PATH, TCOF_FILES_PATH_TO_GET, TCOF_TARGET)
    get_files_from_segments(ETAPE_PATH, ETAPE_FILES_PATH_TO_GET, ETAPE_TARGET)
    get_files_from_segments(CFPP_PATH, CFPP_FILES_PATH_TO_GET, CFPP_TARGET)
    get_files_from_segments(ACSYNT_PATH, ACSYNT_FILES_PATH_TO_GET, ACSYNT_TARGET, time_limit=-1, search_above=True)
