from whisper_online import * 
import argparse
import os
import csv


from tqdm import tqdm
from pathlib import Path
from ast import literal_eval

def export_latencies(args, latencies):

    os.makedirs(args.latency_path,exist_ok=True)

    with open(os.path.join(args.latency_path,"latencies.csv"),"w") as f:
        wr = csv.writer(f)
        wr.writerow(["audio_path","latencies"])
        for i in latencies:
            row = [i[0]]
            row.extend(i[1])
            wr.writerow()
    
    with open(os.path.join(args.latency_path,"latencies.txt"),"w") as f:
        f.write(f"Latency statistics\n")
        f.write(f"Number of files: {len(latencies)}\n")
        for i in latencies:
            f.write(f"{i[0]}: {len(i[1])} latencies values\n")
            f.write(f"\tTotal time: {np.sum(i[1]):.2f}\n")
            f.write(f"\tMean latency: {np.mean(i[1]):.2f}\n")
            f.write(f"\tMax latency: {np.max(i[1]):.2f}\n")
            f.write(f"\tMin latency: {np.min(i[1]):.2f}\n")
            f.write(f"\tStd latency: {np.std(i[1]):.2f}\n")
            f.write(f"\tMedian latency: {np.median(i[1]):.2f}\n")
        

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
    parser.add_argument('--latency_path', type=str, default="latency", help='Where to store the latencies.')
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

    asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir, device=args.device)

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
    latencies = []
    for audio_path in audios_path:
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

        latencies.append([audio_path, []])

        if args.offline: ## offline mode processing (for testing/debugging)
            a = load_audio(audio_path)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError:
                logger.info("assertion error")
                pass
            else:
                output_transcript(o)
            now = None
        elif args.comp_unaware:  # computational unaware mode 
            end = beg + min_chunk
            latencies.clear()
            with tqdm(total=duration) as pbar:
                while True:
                    a = load_audio_chunk(audio_path,beg,end)
                    online.insert_audio_chunk(a)
                    try:
                        o = online.process_iter()
                    except AssertionError:
                        logger.info("assertion error")
                        pass
                    else:
                        output_transcript(o, now=end)

                    logger.debug(f"## last processed {end:.2f}s")

                    if end >= duration:
                        break
                    
                    beg = end
                    
                    if end + min_chunk > duration:
                        end = duration
                    else:
                        end += min_chunk
                now = duration
        
        else: # online = simultaneous mode
            end = 0
            with tqdm(total=duration) as pbar:
                while True:
                    now = time.time() - start
                    if now < end+min_chunk:
                        time.sleep(min_chunk+end-now)
                    end = time.time() - start
                    a = load_audio_chunk(audio_path,beg,end)
                    beg = end
                    online.insert_audio_chunk(a)

                    try:
                        o = online.process_iter()
                    except AssertionError:
                        logger.info("assertion error")
                        pass
                    else:
                        output_transcript(o,start)
                    now = time.time() - start
                    latencies[-1][-1].append(now-end)
                    logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")
                    pbar.n = round(end,3)
                    pbar.refresh()
                    if end >= duration:
                        break
                    
                now = None

        o = online.finish()
        output_transcript(o, start, now=now)

    export_latencies(args, latencies)
    export_params(args)




