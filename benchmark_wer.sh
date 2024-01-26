export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH="${PYTHONPATH}:/home/abert/abert/speech-army-knife"
min_chunk_size=2
data=${2:-../data-fr/normal/}
size=${3:-large-v3}
language=${4:-fr}
FILE=$data
f="$(basename -- $FILE)"
f=${f%.mp3}
path="${f}_wer_${1}"
path_gpu="${path}/koios/gpu"

path_gpu_fw="${path_gpu}/faster"
path_gpu_tw="${path_gpu}/timestamped"
mkdir -p $path_gpu_fw
mkdir -p $path_gpu_tw



# Test offline VS streaming
time python benchmark.py $data --model $size --language $language --task transcribe --offline --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --sub_folders --latency_path $path_gpu_fw/int8_beam_vad_offline
time python benchmark.py $data --model $size --language $language --task transcribe --offline --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --sub_folders --latency_path $path_gpu_tw/float32_beam_vad_offline

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --sub_folders --latency_path $path_gpu_fw/int8_beam_vad
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --sub_folders --latency_path $path_gpu_tw/float32_beam_vad

# Test method

time python benchmark.py $data --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --sub_folders --latency_path $path_gpu_fw/int8_greedy_vad
time python benchmark.py $data --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --sub_folders --latency_path $path_gpu_tw/float32_greedy_vad

# Test model size

time python benchmark.py $data --model medium --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --sub_folders --latency_path $path_gpu_fw/int8_beam_vad
time python benchmark.py $data --model medium --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --sub_folders --latency_path $path_gpu_tw/float32_beam_vad

# Test vad
time python benchmark.py $data --model medium --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --sub_folders --latency_path $path_gpu_fw/int8_beam
time python benchmark.py $data --model medium --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --sub_folders --latency_path $path_gpu_tw/float32_beam
