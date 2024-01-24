export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH="${PYTHONPATH}:/home/abert/abert/speech-army-knife"
min_chunk_size=2
data=${2:-../data-fr/test/smartphone.mp3}
size=${3:-large-v3}
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

mkdir -p $path_cpu_fw/int8_greedy
mkdir -p $path_cpu_fw/float16_greedy
mkdir -p $path_cpu_fw/float32_greedy
mkdir -p $path_cpu_fw/int8float16_greedy

mkdir -p $path_cpu_tw/float32_greedy
mkdir -p $path_cpu_tw/float16_greedy

mkdir -p $path_cpu_fw/int8_beam
mkdir -p $path_cpu_tw/float32_beam

mkdir -p $path_cpu_fw/int8_greedy_vad
mkdir -p $path_cpu_fw/float32_greedy_vad
mkdir -p $path_cpu_tw/float32_greedy_vad
mkdir -p $path_cpu_tw/float32_greedy_vad-auditok

mkdir -p $path_cpu_fw/int8_beam_vad
mkdir -p $path_cpu_tw/float32_beam_vad


# TEST THREADS
/usr/bin/time -o $path_cpu_fw/int8_greedy_16t/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --cpu_threads 16 --latency_path $path_cpu_fw/int8_greedy_16t
/usr/bin/time -o $path_cpu_fw/int8_greedy_8t/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --cpu_threads 8 --latency_path $path_cpu_fw/int8_greedy_8t
/usr/bin/time -o $path_cpu_fw/int8_greedy_2t/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --cpu_threads 2 --latency_path $path_cpu_fw/int8_greedy_2t

/usr/bin/time -o $path_cpu_tw/float32_greedy_16t/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --cpu_threads 16 --latency_path $path_cpu_tw/float32_greedy_16t
/usr/bin/time -o $path_cpu_tw/float32_greedy_8t/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --cpu_threads 8 --latency_path $path_cpu_tw/float32_greedy_8t
/usr/bin/time -o $path_cpu_tw/float32_greedy_2t/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --cpu_threads 2 --latency_path $path_cpu_tw/float32_greedy_2t


# Faster whisper - test precision - no vad, greedy
/usr/bin/time -o $path_cpu_fw/int8_greedy/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_fw/int8_greedy
/usr/bin/time -o $path_cpu_fw/float16_greedy/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_fw/float16_greedy --compute_type float16 
/usr/bin/time -o $path_cpu_fw/float32_greedy/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_fw/float32_greedy --compute_type float32 
# /usr/bin/time -o $path_cpu_fw/int8float16_greedy/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_fw/int8float16_greedy --compute_type int8_float16 

# Whisper Timestamped - test precision - no vad, greedy
/usr/bin/time -o $path_cpu_tw/float32_greedy/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_tw/float32_greedy 
# /usr/bin/time -o $path_cpu_tw/float16_greedy/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_tw/float16_greedy --compute_type float16 

# Test beam search/greedy

/usr/bin/time -o $path_cpu_fw/int8_beam/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --latency_path $path_cpu_fw/int8_beam
/usr/bin/time -o $path_cpu_tw/float32_beam/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --latency_path $path_cpu_tw/float32_beam

# Test VAD - both backends, best precision (fp32 for timestamped, int8 for faster), beam search and greedy

/usr/bin/time -o $path_cpu_fw/int8_greedy_vad/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_cpu_fw/int8_greedy_vad 
/usr/bin/time -o $path_cpu_fw/float32_greedy_vad/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_cpu_fw/float32_greedy_vad --compute_type float32 
/usr/bin/time -o $path_cpu_tw/float32_greedy_vad/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_cpu_tw/float32_greedy_vad 
/usr/bin/time -o $path_cpu_tw/float32_greedy_vad-auditok/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py python benchmark.py $data --device cpu --model $size --language $language --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad auditok --latency_path $path_cpu_tw/float32_greedy_vad-auditok

/usr/bin/time -o $path_cpu_fw/int8_beam_vad/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_cpu_fw/int8_beam_vad
/usr/bin/time -o $path_cpu_tw/float32_beam_vad/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_cpu_tw/float32_beam_vad

if [[ $data =~ "data-fr" ]]
then
    data="../data-fr/silence"
    mdir -p $path_cpu_fw/int8_greedy_vad_silence
    mkdir -p $path_cpu_fw/float32_greedy_vad_silence
    mkdir -p $path_cpu_tw/float32_greedy_vad_silence
    mkdir -p $path_cpu_tw/float32_greedy_vad-auditok_silence

    mkdir -p $path_cpu_fw/int8_beam_vad_silence
    mkdir -p $path_cpu_tw/float32_beam_vad_silence

    mkdir -p $path_cpu_fw/int8_greedy_silence
    mkdir -p $path_cpu_tw/float32_greedy_silence
    
    /usr/bin/time -o $path_cpu_fw/int8_greedy_vad_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_cpu_fw/int8_greedy_vad_silence
    /usr/bin/time -o $path_cpu_fw/float32_greedy_vad_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_cpu_fw/float32_greedy_vad_silence --compute_type float32 
    /usr/bin/time -o $path_cpu_tw/float32_greedy_vad_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad --latency_path $path_cpu_tw/float32_greedy_vad_silence 
    /usr/bin/time -o $path_cpu_tw/float32_greedy_vad-auditok_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --vad auditok --latency_path $path_cpu_tw/float32_greedy_vad-auditok_silence

    # /usr/bin/time -o $path_cpu_fw/int8_beam_vad_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_cpu_fw/int8_beam_vad_silence
    # /usr/bin/time -o $path_cpu_tw/float32_beam_vad_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method beam-search --vad --latency_path $path_cpu_tw/float32_beam_vad_silence

    /usr/bin/time -o $path_cpu_fw/int8_greedy_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend faster-whisper --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_fw/int8_greedy_silence
    /usr/bin/time -o $path_cpu_tw/float32_greedy_silence/ram.txt -f "Maximum RSS size: %M KB\nCPU percentage used: %P" python benchmark.py $data --model $size --language $language --device cpu --task transcribe --backend whisper_timestamped --min-chunk-size $min_chunk_size --method greedy --latency_path $path_cpu_tw/float32_greedy_silence 
fi


