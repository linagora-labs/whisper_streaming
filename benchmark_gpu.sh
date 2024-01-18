export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="${PYTHONPATH}:/home/abert/abert/speech-army-knife"
min_chunk_size=2
data=${2:-../data/smartphone.mp3}
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
mkdir -p $path_cpu_fw
mkdir -p $path_cpu_tw
mkdir -p $path_gpu_fw
mkdir -p $path_gpu_tw


# Faster whisper - test precision - no vad, greedy
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8_greedy > $path_gpu_fw/int8_greedy.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/float16_greedy --compute_type float16 > $path_gpu_fw/float16_greedy.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/float32_greedy --compute_type float32 > $path_gpu_fw/float32_greedy.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8float16_greedy --compute_type int8_float16 > $path_gpu_fw/int8float16_greedy.txt

# Whisper Timestamped - test precision - no vad, greedy
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/float32_greedy > $path_gpu_tw/float32_greedy.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/float16_greedy --compute_type float16 > $path_gpu_tw/float16_greedy.txt

# Test VAD - both backends, best precision (fp32 for timestamped, int8 for faster), beam search and greedy








# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --vad --latency_path $path_gpu_fw/int8_vad > $path_gpu_fw/int8_vad.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8_greedy > $path_gpu_fw/int8_greedy.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_fw/int8_vad_greedy > $path_gpu_fw/int8_vad_greedy.txt

# # Whisper Timestamped

# time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --latency_path $path_gpu_tw/fp32 > $path_gpu_tw/fp32.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --vad --latency_path $path_gpu_tw/fp32_vad > $path_gpu_tw/fp32_vad.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/fp32_greedy > $path_gpu_tw/fp32_greedy.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_tw/fp32_vad_greedy > $path_gpu_tw/fp32_vad_greedy.txt



