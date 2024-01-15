export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
min_chunk_size=${2:-2}
data=${3:-../data/apollo11.mp3}
size=${4:-small}
language=${5:-en}
FILE=$data
f="$(basename -- $FILE)"
f=${f%.mp3}
path="${f}_${1}"
mkdir -p $path
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --latency_path $path/out_fw_gpu > $path/out_fw_gpu.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --latency_path $path/out_wt_gpu > $path/out_wt_gpu.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --vad --latency_path $path/out_fw_gpu_vad > $path/out_fw_gpu_vad.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --vad --latency_path $path/out_wt_gpu_vad > $path/out_wt_gpu_vad.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --device cpu --min-chunk-size $min_chunk_size --latency_path $path/out_fw_cpu > $path/out_fw_cpu.txt
time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --device cpu --min-chunk-size $min_chunk_size --vad --latency_path $path/out_fw_cpu_vad > $path/out_fw_cpu_vad.txt

