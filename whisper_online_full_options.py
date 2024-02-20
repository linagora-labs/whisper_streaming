import logging
logging.basicConfig(filename="log.txt", filemode="a")
logger = logging.getLogger(__name__)

import whisper_online
import argparse
import os
import csv
import json
import torch
import time
import sys
import numpy as np
import gc

from tqdm import tqdm
from linastt.utils.monitoring import tic, toc, vram_peak, ram_peak 

def export_processing_times(args, processing_times):

    os.makedirs(args.output_path,exist_ok=True)

    with open(os.path.join(args.output_path,"result.json"), 'w') as fp:
        json.dump(processing_times, fp, indent=4) 

    
    with open(os.path.join(args.output_path,"result.txt"),"w") as f:
        f.write(f"Processing time statistics\n")
        f.write(f"Global statistics:\n")
        f.write(f"Number of files: {len(processing_times)}\n\n")

        all_processing_times = []
        f.write(f"All segements statistics:\n")
        for i in processing_times:
            all_processing_times += processing_times[i]['segment_processing_time']
        f.write(f"\tNumber of segements: {len(all_processing_times)}\n")
        f.write(f"\tTotal time: {np.sum(all_processing_times):.2f}\n")
        f.write(f"\tMean: {np.mean(all_processing_times):.2f}\n")
        f.write(f"\tMax: {np.max(all_processing_times):.2f}\n")
        f.write(f"\tMin: {np.min(all_processing_times):.2f}\n")
        f.write(f"\tStd: {np.std(all_processing_times):.2f}\n")
        f.write(f"\tMedian: {np.median(all_processing_times):.2f}\n\n")
        f.write(f"Processing time statistics per file:\n")
        for i in processing_times:
            f.write(f"\t{i}: {len(processing_times[i]['segment_duration'])} processing_times values\n")
            f.write(f"\t\tTotal time: {np.sum(processing_times[i]['segment_processing_time']):.2f}\n")
            f.write(f"\t\tMean: {np.mean(processing_times[i]['segment_processing_time']):.2f}\n")
            f.write(f"\t\tMax: {np.max(processing_times[i]['segment_processing_time']):.2f}\n")
            f.write(f"\t\tMin: {np.min(processing_times[i]['segment_processing_time']):.2f}\n")
            f.write(f"\t\tStd: {np.std(processing_times[i]['segment_processing_time']):.2f}\n")
            f.write(f"\t\tMedian: {np.median(processing_times[i]['segment_processing_time']):.2f}\n")

def export_params(args):
    with open(os.path.join(args.output_path,"params.txt"),"w") as f:
        f.write(f"Parameters\n")
        f.write(f"Audio path: {args.audio_path}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Language: {args.lan}\n")
        f.write(f"Backend: {args.backend}\n")
        f.write(f"Task: {args.task}\n")  
        f.write(f"Device: {args.device}\n")
        if args.device == "cuda":
            f.write(f"GPU: {torch.cuda.get_device_name()}\n")
        else:
            f.write(f"CPU threads: {args.cpu_threads}\n")
        f.write(f"Offline: {args.offline}\n")
        f.write(f"Comp unaware: {args.comp_unaware}\n")

        f.write(f"Buffer trimming: {args.buffer_trimming}\n")
        f.write(f"Buffer trimming sec: {args.buffer_trimming_sec}\n")
        f.write(f"Min chunk size: {args.min_chunk_size}\n")
    
        f.write(f"Output path: {args.output_path}\n")

        f.write(f"VAD: {args.vad}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Previous text: {args.previous_text}\n")
        f.write(f"Compute type: {args.compute_type}\n")
        f.write(f"Verbose: {args.verbose}\n")

def export_transcipt(transcripts, file=None):
    if isinstance(file, str):
        f = open(file, "w")
    else:
        f = file
    for i in transcripts:
        if i[0] is not None:
            f.write(f"{i[0]:1.3f} {i[1]:1.3f} {i[2]}\n")
    if isinstance(file, str):
        f.close()

def output_streaming(out, out_time=0, commit=False, buffered_time=None):
    if out[0] is not None:
        if commit:
            print(f"{buffered_time:6.2f} {out_time:6.2f} {out[0]:6.2f} {out[1]:6.2f} {out[2]:<100}", flush=True)
        else:
            print(f"{out_time:6.2f} {0:6.2f} {out[0]:6.2f} {out[1]:6.2f} {out[2]}", end="\r", flush=True)
    else:
        if commit:
            print(f"{out_time:6.2f} {'':<120}", flush=True)
        else:
            print(f"{out_time:6.2f}", end="\r", flush=True)


def process_file(audio_path, args, online, processing_times):
    # if os.path.exists(os.path.join(args.output_path,"transcripts",os.path.basename(audio_path).replace(".mp3",".txt").replace(".wav",".txt").replace(".flac",".txt"))):
    #     logger.info(f"{audio_path} already processed")
    #     return processing_times

    min_chunk = args.min_chunk_size
    SAMPLING_RATE = 16000
    BENCHMARK_MODE = True

    duration = len(whisper_online.load_audio(audio_path))/SAMPLING_RATE
    logger.info("")
    logger.info(f"Processing {audio_path} (duration is {duration:.2f}s)")

    beg = args.start_at
    start = time.time() - beg
    os.makedirs(os.path.join(args.output_path,"transcripts"),exist_ok=True)
    
    processing_times[audio_path] = {'max_vram': -1,'segment_duration' : [], 'segment_timestamps': [], 'segment_processing_time': []}
    transcripts = []
    if args.offline: ## offline mode processing (for testing/debugging)
        start_time = time.time()
        a = whisper_online.load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
            end_time = time.time()
        except AssertionError:
            logger.info("assertion error")
            pass
        else:
            whisper_online.output_transcript(o, start)
            transcripts.append(o)
        processing_times[audio_path]['segment_duration'].append(duration)
        processing_times[audio_path]['segment_timestamps'].append((0,duration))
        processing_times[audio_path]['segment_processing_time'].append(end_time-start_time)
        logger.info(f"Finished processing {audio_path} in {end_time-start_time:.2f}s")
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        if BENCHMARK_MODE:
            pbar = tqdm(total=round(duration,3))
        while True:
            start_time = time.time()
            a = whisper_online.load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
                end_time = time.time()
            except AssertionError:
                logger.info("assertion error")
                pass
            else:
                whisper_online.output_transcript(o, start, now=end)
                transcripts.append(o)
            logger.debug(f"## last processed {end:.2f}s")
            processing_times[audio_path]['segment_duration'].append(end-beg)
            processing_times[audio_path]['segment_timestamps'].append((beg,end))
            processing_times[audio_path]['segment_processing_time'].append(end_time-start_time)
            if BENCHMARK_MODE:
                pbar.n = round(end,3)
                pbar.refresh()
            if end >= duration:
                break
            beg = end
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration
    
    else: # online = simultaneous mode
        processing_times[audio_path]['segment_latency'] = []
        processing_times[audio_path]['segment_start_latency'] = []
        processing_times[audio_path]['segment_start_buffer_latency'] = []
        processing_times[audio_path]['segment_buffer_latency'] = []
        end = 0
        
        la3 = []
        lu3 = []
        lu2 = []
        buffered_time = 0
        # from playsound import playsound
        # playsound(os.path.abspath(audio_path), False)
        if BENCHMARK_MODE:
            pbar = tqdm(total=round(duration,3))
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            logger.debug(f"Processing {beg:.2f} to {end:.2f}")
            start_time = time.time()
            a = whisper_online.load_audio_chunk(audio_path, beg, end)
            
            online.insert_audio_chunk(a)
            processing_times[audio_path]['segment_duration'].append(len(online.audio_buffer)/online.SAMPLING_RATE)
            processing_times[audio_path]['segment_timestamps'].append((online.buffer_time_offset,online.buffer_time_offset+len(online.audio_buffer)/online.SAMPLING_RATE))
            try:
                committed, buffer = online.process_iter()
                end_time = time.time()
            except AssertionError:
                logger.info("assertion error")
                pass
            else:
                # whisper_online.output_transcript(committed, now=end_time-start, end="                                    \n")
                # whisper_online.output_transcript(buffer, now=end_time-start, end="\r")
                if not BENCHMARK_MODE:
                    if committed[0] is not None:
                        output_streaming(committed, out_time=end_time-start, commit=True, buffered_time=buffered_time)
                    output_streaming(buffer, out_time=end_time-start)
                buffered_time = end_time-start
                transcripts.append(committed)
            now = time.time() - start
            processing_times[audio_path]['segment_processing_time'].append(end_time-start_time)
            if committed[0] is not None:
                processing_times[audio_path]['segment_latency'].append(now - committed[1])
                processing_times[audio_path]['segment_start_latency'].append(now - committed[0])
            if buffer[0] is not None:
                processing_times[audio_path]['segment_start_buffer_latency'].append(now - buffer[0])
                processing_times[audio_path]['segment_buffer_latency'].append(now - buffer[1])
            if buffer[0] is not None:
                if now - buffer[0]>3:
                    la3.append(now - buffer[0])
                elif now - buffer[0]<2:
                    lu2.append(now-buffer[0])
                else:
                    lu3.append(now-buffer[0])
            logger.debug(f"The latency is {now-end:.2f}s and output is '{committed[2]}'")
            if BENCHMARK_MODE:
                pbar.n = min(round(end,3), pbar.total)
                pbar.refresh()
            beg = end
            if end >= duration:
                break
            
        now = None

    if args.device == "cuda":
        processing_times[audio_path]['max_vram'] = vram_peak()
        try:
            logger.info(f'Number of GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        except KeyError:
            pass
        logger.info(f"GPU used: {torch.cuda.get_device_name()}")
    o = online.finish()
    transcripts.append(o)
    # logging.getLogger(__name__).setLevel(level=logging.INFO)
    if not BENCHMARK_MODE:
        output_streaming(o, out_time=end_time-start, commit=True, buffered_time=buffered_time)
        print(f"Above 3sec: {len(la3)}")
        print(f"Between 2 and 3sec: {len(lu3)}")
        print(f"Below 2sec: {len(lu2)}")
    export_transcipt(transcripts, os.path.join(args.output_path,"transcripts",os.path.basename(audio_path).replace(".mp3",".txt").replace(".wav",".txt").replace(".flac",".txt")))
    return processing_times

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename (or folder) of 16kHz mono channel wav, on which live streaming is simulated.")
    # parser.add_argument('--folder', action="store_true", help="If set, audio_path is a folder with wav files, not a single file.")
    whisper_online.add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"],help='Device used.')
    parser.add_argument('--compute_type', type=str, default="int8", choices=["int8", "float16", "float32", "int8_float16"], help='Computation type (int8, float16...).')
    parser.add_argument('--output_path', type=str, default="./", help='Output folder of the script.')
    parser.add_argument('--method', type=str, default="beam-search", choices=["beam-search", "greedy"],help='Greedy or beam search decoding.')
    parser.add_argument('--verbose', default=1, help='Verbose mode (2=DEBUG, 1=INFO, 0=ERROR).')
    parser.add_argument('--cpu_threads', default=4, help='When running on CPU, number of threads to use.')
    parser.add_argument('--previous_text', action="store_true", default=False, help='Condition on previous text (default False).')
    parser.add_argument('--subfolders', action="store_true", default=False, help='Search for audios in subfolders (default False).')
    args = parser.parse_args()
    if args.verbose==2:
        logging.getLogger(__name__).setLevel(level=logging.DEBUG)
        # logging.getLogger('numba').setLevel(logging.WARNING)
        # logging.getLogger('faster_whisper').setLevel(logging.WARNING)
    elif args.verbose==1:
        logging.getLogger(__name__).setLevel(level=logging.INFO)
    else:
        logging.getLogger(__name__).setLevel(level=logging.ERROR)
        # logging.basicConfig(filename="log.txt", filemode="a", level=logging.ERROR)  
   
    
    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)
    return args

def init_processor(args):
    size = args.model
    language = args.lan

    t = time.time()
    logger.info(f"Loading Whisper {size} model for {language}...")
    model_kwargs = {'device': args.device, 'cpu_threads': int(args.cpu_threads), 'compute_type': args.compute_type}
    if args.backend == "faster-whisper":
        asr_cls = whisper_online.FasterWhisperASR
    else:
        asr_cls = whisper_online.WhisperTimestampedASR
        if args.backend == "whisper_timestamped-transformers":
            model_kwargs['backend'] = "transformers"
        else:
            model_kwargs['backend'] = "openai-whisper"
    asr = asr_cls(modelsize=size, lan=language, model_kwargs=model_kwargs)

    if args.method != "greedy":
        asr.transcribe_kargs['beam_size'] = 5
        asr.transcribe_kargs['best_of'] = 5
        asr.transcribe_kargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language


    e = time.time()
    logger.info(f"Loading finished. It took {e-t:.2f} seconds.")
    
    if args.vad:
        logger.info(f"setting VAD filter {args.vad}")
        asr.use_vad(args.vad if args.vad!=True else None)
    
    if args.buffer_trimming == "sentence":
        tokenizer = whisper_online.create_tokenizer(tgt_language)
    else:
        tokenizer = None
    online_processor = whisper_online.OnlineASRProcessor(asr,tokenizer,logfile=logger,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    return online_processor

def get_file_list(args):
    SUBFOLDERS = args.subfolders
    audios_path = []
    if os.path.isdir(args.audio_path): 
        paths = os.listdir(args.audio_path)
        paths = [os.path.join(args.audio_path, f) for f in paths]
        if SUBFOLDERS:
            sub_folders = [f for f in paths if os.path.isdir(f)]
            audios_path = paths
            for sub_folder in sub_folders:
                audios_path += [os.path.join(sub_folder, f) for f in os.listdir(sub_folder)]
        else:
            audios_path = paths
        audios_path.sort()
        audios_path = [f for f in audios_path if f.endswith(".wav") or f.endswith(".mp3") or f.endswith(".flac")]
        logger.info(f"Processing files in {args.audio_path} ({len(audios_path)} files)")
    else:
        audios_path = [args.audio_path]
    return audios_path

if __name__ == "__main__":

    args = init_args()

    audios_path = get_file_list(args)

    processing_times = {}
    for audio_path in tqdm(audios_path, total=len(audios_path)):
        online_processor = init_processor(args)
        # load the audio into the LRU cache before we start the timer
        a = whisper_online.load_audio_chunk(audios_path[0],0,1)

        # warm up the ASR, because the very first transcribe takes much more time than the other
        online_processor.asr.transcribe(a)
        processing_times = process_file(audio_path, args, online_processor, processing_times)
        online_processor = None
        gc.collect()
                
    export_processing_times(args, processing_times)
    export_params(args)




