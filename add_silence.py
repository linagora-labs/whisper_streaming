
import os
from pydub import AudioSegment
import random
import argparse
# input_dir = "/home/abert/abert/data-fr/normal"
# output_dir = "/home/abert/abert/data-fr/silence"
input_dir = "/home/abert/Linagora/data-fr/test/smartphone_35.mp3"

output_dir = None



def add_silence(file_path, output_dir=None, number_of_silence=3, silence_duration=20000):
    if file_path.endswith(".wav") or file_path.endswith(".mp3") or file_path.endswith(".flac"):
        if file_path.endswith(".wav"):
            sound = AudioSegment.from_wav(file_path)
        elif file_path.endswith(".mp3"):
            sound = AudioSegment.from_mp3(file_path)
        elif file_path.endswith(".flac"):
            sound = AudioSegment.from_file(file_path)
        for i in range(number_of_silence):
            silence = AudioSegment.silent(duration=silence_duration)  #duration in milliseconds
            random_position = random.randint(0, len(sound))
            sound = sound[:random_position] + silence + sound[random_position:]
        file = file_path.split("/")[-1]
        path = os.path.join(*file_path.split("/")[:-1])
        extension = "."+file.split(".")[-1]
        basename = file.split(".")[0]
        if output_dir:
            print(os.path.join(output_dir, file))
            sound.export(os.path.join(output_dir, file), format=extension[1:])
        else:
            print(os.path.join(path, basename+"_silenced.wav"+extension))
            sound.export(os.path.join(path, basename+"_silenced"+extension), format=extension[1:])

if __name__=="__main__":
    # copy all files from input_dir to output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(input_dir):
        add_silence(input_dir, output_dir)
    else: 
        for file in os.listdir(input_dir):
            add_silence(os.path.join(input_dir, file), output_dir)