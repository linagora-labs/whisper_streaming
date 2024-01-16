from whisper_online import * 

import argparse
import os
import csv
import json


from tqdm import tqdm
from pathlib import Path
from ast import literal_eval

def export_processing_times(args, processing_timess):

    os.makedirs(args.latency_path,exist_ok=True)

    with open(os.path.join(args.latency_path,"processing_times.json"), 'w') as fp:
        json.dump(processing_times, fp, indent=4) 

    
    with open(os.path.join(args.latency_path,"processing_times.txt"),"w") as f:
        f.write(f"Processing time statistics\n")
        f.write(f"Global statistics:\n")
        f.write(f"Number of files: {len(processing_times)}\n\n")

        all_processing_times = []
        f.write(f"All segements statistics:\n")
        for i in processing_times:
            all_processing_times += processing_times[i]['segments_processing_times']
        f.write(f"\tNumber of segements: {len(all_processing_times)}\n")
        f.write(f"\tTotal time: {np.sum(all_processing_times):.2f}\n")
        f.write(f"\tMean: {np.mean(all_processing_times):.2f}\n")
        f.write(f"\tMax: {np.max(all_processing_times):.2f}\n")
        f.write(f"\tMin: {np.min(all_processing_times):.2f}\n")
        f.write(f"\tStd: {np.std(all_processing_times):.2f}\n")
        f.write(f"\tMedian: {np.median(all_processing_times):.2f}\n\n")
        f.write(f"Processing time statistics per file:\n")
        for i in processing_times:
            f.write(f"\t{i}: {len(processing_times[i]['segments_duration'])} processing_times values\n")
            f.write(f"\t\tTotal time: {np.sum(processing_times[i]['segments_processing_times']):.2f}\n")
            f.write(f"\t\tMean: {np.mean(processing_times[i]['segments_processing_times']):.2f}\n")
            f.write(f"\t\tMax: {np.max(processing_times[i]['segments_processing_times']):.2f}\n")
            f.write(f"\t\tMin: {np.min(processing_times[i]['segments_processing_times']):.2f}\n")
            f.write(f"\t\tStd: {np.std(processing_times[i]['segments_processing_times']):.2f}\n")
            f.write(f"\t\tMedian: {np.median(processing_times[i]['segments_processing_times']):.2f}\n")
        

def export_params(args):
    with open(os.path.join(args.latency_path,"params.txt"),"w") as f:
        f.write(f"Parameters\n")
        f.write(f"Audio path: {args.audio_path}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Language: {args.lan}\n")
        f.write(f"Backend: {args.backend}\n")
        f.write(f"VAD: {args.vad}\n")
        f.write(f"Buffer trimming: {args.buffer_trimming}\n")
        f.write(f"Buffer trimming sec: {args.buffer_trimming_sec}\n")
        f.write(f"Min chunk size: {args.min_chunk_size}\n")
        f.write(f"Offline: {args.offline}\n")
        f.write(f"Comp unaware: {args.comp_unaware}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Latency path: {args.latency_path}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename (or folder) of 16kHz mono channel wav, on which live streaming is simulated.")
    # parser.add_argument('--folder', action="store_true", help="If set, audio_path is a folder with wav files, not a single file.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"],help='Device used.')
    parser.add_argument('--compute_type', type=str, default="int8", choices=["int8", "fp16", "fp32", "int8_float16"], help='Computation type (int8, fp16...).')
    parser.add_argument('--latency_path', type=str, default="latency", help='Where to store the processing_times.')
    parser.add_argument('--method', type=str, default="beam_search", choices=["beam_search", "greedy"],help='Greedy or beam search decoding.')
    parser.add_argument('--verbose', default=1, help='Verbose mode.')
    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    # logfile = sys.stderr

    if args.verbose==2:
        logging.basicConfig(filename="log.txt", filemode="w", level=logging.DEBUG)
    elif args.verbose==1:
        logging.basicConfig(filename="log.txt", filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(filename="log.txt", filemode="w", level=logging.ERROR)  
    logger= logging.getLogger(__name__)

    if args.offline and args.comp_unaware:
        logger.error("No or one option from --offline and --comp_unaware are available, not both. Exiting.")
        sys.exit(1)

    size = args.model
    language = args.lan

    t = time.time()
    logger.info(f"Loading Whisper {size} model for {language}...")

    if args.backend == "faster-whisper":
        asr_cls = FasterWhisperASR
    else:
        asr_cls = WhisperTimestampedASR

    asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir, device=args.device, compute_type=args.compute_type)

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
        logger.info("setting VAD filter")
        asr.use_vad()

    
    min_chunk = args.min_chunk_size
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None
    online = OnlineASRProcessor(asr,tokenizer,logfile=logger,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    if os.path.isdir(args.audio_path): 
        audios_path = os.listdir(args.audio_path)
        audios_path.sort()
        audios_path = [os.path.join(args.audio_path, f) for f in audios_path]
        logger.info(f"Processing files in {args.audio_path} ({len(audios_path)} files)")
    else:
        audios_path = [args.audio_path]
    processing_times = {}
    latencies = []
    # tqdm loop

    for audio_path in tqdm(audios_path, total=len(audios_path)):
        if not audio_path.endswith(".wav") and not audio_path.endswith(".mp3"):
            continue
        SAMPLING_RATE = 16000
        duration = len(load_audio(audio_path))/SAMPLING_RATE

        logger.info(f"Processing {audio_path} (duration is {duration:.2f}s)")


        # load the audio into the LRU cache before we start the timer
        a = load_audio_chunk(audio_path,0,1)

        # warm up the ASR, because the very first transcribe takes much more time than the other
        asr.transcribe(a)

        beg = args.start_at
        start = time.time()-beg

        processing_times[audio_path] = {'segments_duration' : [], 'segments_timestamps': [], 'segments_processing_times': []}
        if args.offline: ## offline mode processing (for testing/debugging)
            start_time = time.time()
            a = load_audio(audio_path)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
                end_time = time.time()
            except AssertionError:
                logger.info("assertion error")
                pass
            else:
                output_transcript(o, start)
            processing_times[audio_path]['segments_duration'].append(duration)
            processing_times[audio_path]['segments_timestamps'].append((0,duration))
            processing_times[audio_path]['segments_processing_times'].append(end_time-start_time)
            logger.info(f"Finished processing {audio_path} in {end_time-start_time:.2f}s")
            now = None
        elif args.comp_unaware:  # computational unaware mode 
            end = beg + min_chunk
            with tqdm(total=duration) as pbar:
                while True:
                    start_time = time.time()
                    a = load_audio_chunk(audio_path,beg,end)
                    online.insert_audio_chunk(a)
                    try:
                        o = online.process_iter()
                        end_time = time.time()
                    except AssertionError:
                        logger.info("assertion error")
                        pass
                    else:
                        output_transcript(o, start, now=end)
                    logger.debug(f"## last processed {end:.2f}s")
                    processing_times[audio_path]['segments_duration'].append(end-beg)
                    processing_times[audio_path]['segments_timestamps'].append((beg,end))
                    processing_times[audio_path]['segments_processing_times'].append(end_time-start_time)
                    if end >= duration:
                        pbar.n = round(duration,3)
                        pbar.refresh()
                        break
                    pbar.n = round(end,3)
                    pbar.refresh()
                    beg = end
                    if end + min_chunk > duration:
                        end = duration
                    else:
                        end += min_chunk
                now = duration
        
        else: # online = simultaneous mode
            processing_times[audio_path]['latencies'] = []
            end = 0
            with tqdm(total=duration) as pbar:
                while True:
                    now = time.time() - start
                    if now < end+min_chunk:
                        time.sleep(min_chunk+end-now)
                    end = time.time() - start

                    start_time = time.time()
                    a = load_audio_chunk(audio_path,beg,end)
                    
                    online.insert_audio_chunk(a)
                    try:
                        o = online.process_iter()
                        end_time = time.time()

                    except AssertionError:
                        logger.info("assertion error")
                        pass
                    else:
                        output_transcript(o,start)
                    
                    now = time.time() - start
                    processing_times[audio_path]['segments_duration'].append(end-beg)
                    processing_times[audio_path]['segments_timestamps'].append((beg,end))
                    processing_times[audio_path]['segments_processing_times'].append(end_time-start_time)
                    processing_times[audio_path]['latencies'].append(now-end)
                    logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")
                    pbar.n = round(end,3)
                    pbar.refresh()
                    beg = end
                    if end >= duration:
                        break
                    
                now = None

        o = online.finish()
        output_transcript(o, start, now=now)

    export_processing_times(args, processing_times)
    export_params(args)




