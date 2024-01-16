export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
min_chunk_size=2
data=${2:-../data}
size=${3:-small}
language=${4:-en}
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

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --latency_path $path_gpu_fw/int8 > $path_gpu_fw/int8.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --latency_path $path_gpu_tw/int8 > $path_gpu_tw/int8.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --vad --latency_path $path_gpu_fw/int8_vad > $path_gpu_fw/int8_vad.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --vad --latency_path$path_gpu_tw/int8_vad > $path_gpu_tw/int8_vad.txt

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_fw/int8_greedy > $path_gpu_fw/int8_greedy.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_gpu_tw/int8_greedy > $path_gpu_tw/int8_greedy.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_fw/int8_vad_greedy > $path_gpu_fw/int8_vad_greedy.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_gpu_tw/int8_vad_greedy > $path_gpu_tw/int8_vad_greedy.txt

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --latency_path $path_gpu_tw/fp16 --compute_type fp16 > $path_gpu_tw/fp16.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --latency_path $path_gpu_tw/fp16 --compute_type fp16 > $path_gpu_tw/fp16.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --vad --latency_path $path_gpu_tw/fp16_vad --compute_type fp16 > $path_gpu_tw/fp16_vad.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --vad --latency_path $path_gpu_tw/fp16_vad --compute_type fp16 > $path_gpu_tw/fp16_vad.txt

# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --device cpu --min-chunk-size $min_chunk_size --latency_path $path/faster/cpu > $path/faster/cpu.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --device cpu --min-chunk-size $min_chunk_size --vad --latency_path $path/faster/cpu_vad > $path/faster/cpu_vad.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --device cpu --min-chunk-size $min_chunk_size --latency_path $path/timestamped/cpu > $path/timestamped/cpu.txt
# time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --device cpu --min-chunk-size $min_chunk_size --vad --latency_path $path/timestamped/cpu_vad > $path/timestamped/cpu_vad.txt

