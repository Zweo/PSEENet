import os
from sleepedf_process import transEDF2MAT

def download_SleepEDF():
    print('Start downloading...')
    os.system('wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/')
    print('The data is saved into pysionet.org/files/sleep-edfx/1.0.0/sleep-cassette')

def preprocess(inPath, outPath):
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    transEDF2MAT(inPath, outPath)

def run():
    os.system('python train.py')

download_SleepEDF()
path = 'pysionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
preprocess(path, 'data')
run()
