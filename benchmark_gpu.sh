export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH="${PYTHONPATH}:/home/abert/abert/speech-army-knife"
min_chunk_size=2
data=${2:-../data-fr/normal/smartphone.mp3}
size=${3:-small}
language=${4:-fr}
FILE=$data
f="$(basename -- $FILE)"
f=${f%.mp3}
path="${f}_${1}"
path_cpu="${path}/koios/cpu"
path_gpu="${path}/koios/gpu"

path_cpu_fw="${path_cpu}/faster"
path_cpu_tw="${path_cpu}/timestamped"
path_gpu_fw="${path_gpu}/faster"
path_gpu_tw="${path_gpu}/timestamped"
mkdir -p $path_gpu_fw
mkdir -p $path_gpu_tw


# Faster whisper - test precision - no vad, greedy
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8_greedy 
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/float16_greedy --compute_type float16 
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/float32_greedy --compute_type float32 
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8float16_greedy --compute_type int8_float16 

# # Whisper Timestamped - test precision - no vad, greedy
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/float32_greedy 
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/float16_greedy --compute_type float16 

# Test beam search/greedy

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --latency_path $path_gpu_fw/int8_beam
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --latency_path $path_gpu_tw/float32_beam

# Test condition previous text
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --previous_text --latency_path $path_gpu_fw/int8_greedy_previous-text
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --previous_text --latency_path $path_gpu_tw/float32_greedy_previous-text

# Test VAD - both backends, best precision (fp32 for timestamped, int8 for faster), beam search and greedy

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_fw/int8_greedy_vad 
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_fw/float32_greedy_vad --compute_type float32 
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_tw/float32_greedy_vad 
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad auditok --latency_path $path_gpu_tw/float32_greedy_vad-auditok

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_gpu_fw/int8_beam_vad
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_gpu_tw/float32_beam_vad

    # data with long silence
if [[ $data =~ "data-fr" ]]
then
    data="../data-fr/silence"
    time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_fw/int8_greedy_vad_silence
    time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_fw/float32_greedy_vad_silence --compute_type float32 
    time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_tw/float32_greedy_vad_silence 
    time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad auditok --latency_path $path_gpu_tw/float32_greedy_vad-auditok_silence

    time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_gpu_fw/int8_beam_vad_silence
    time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_gpu_tw/float32_beam_vad_silence

    time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8_greedy_silence
    time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/float32_greedy_silence 
fi
