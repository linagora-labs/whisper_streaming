# script that show duration for each audio in a folder

if __name__=="__main__":
    import argparse
    import os
    import subprocess
    import sys
    import time
    import librosa
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Show duration for each audio in a folder')
    parser.add_argument('folder', help='folder to process')
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print("Error: {} is not a folder".format(folder))
        sys.exit(1)
    total = []
    for file in tqdm(os.listdir(folder), total=len(os.listdir(folder))):
        if file.endswith(".wav") or file.endswith(".mp3"):
            duration = librosa.get_duration(path=os.path.join(folder, file))
            total.append(duration)
            print(f"{file}: {duration:.0f}s")
    print()
    print(f"Total: {sum(total):.0f}s ({sum(total)/60:.1f}min) ({len(total)} files)")