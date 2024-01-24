
import os
from pydub import AudioSegment
import random

input_dir = "/home/abert/abert/data-fr/normal"
output_dir = "/home/abert/abert/data-fr/silence"

if __name__=="__main__":
    # copy all files from input_dir to output_dir
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            number_of_silences = 3
            sound = AudioSegment.from_wav(os.path.join(input_dir, file))
            for i in range(number_of_silences):
                silence = AudioSegment.silent(duration=20000)  #duration in milliseconds
                random_position = random.randint(0, len(sound))
                sound = sound[:random_position] + silence + sound[random_position:]
            print(output_dir, file)
            sound.export(os.path.join(output_dir, file), format="wav")
