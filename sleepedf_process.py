# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:15:33 2021

@author: User01
"""
import pyedflib
import numpy as np
from scipy import io as sio
from scipy import signal
import os
import glob
import ntpath
import librosa
from sklearn import preprocessing


def transEDF2MAT(path, output_dir):
    psg_name = glob.glob(f"{path}*PSG.edf")
    label_name = glob.glob(path + "*Hypnogram.edf")
    epoch = 30
    fs = 100

    for l in range(len(label_name)):

        raw = pyedflib.EdfReader(label_name[l])
        n = raw.signals_in_file
        labels = raw.getSignalLabels()
        signal_headers = raw.getSignalHeaders()
        annotations = raw.readAnnotations()
        stages_lookup1 = {
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage ?': -1,
            'Sleep stage R': 4,
            'Sleep stage W': 0
        }
        stage_idx1 = []
        stages1 = []
        for i, annot in enumerate(annotations[2]):
            for stage in stages_lookup1:
                if stage in annot:
                    stage_idx1 += [i]
                    stages1 += [stages_lookup1[stage]]
        time_begin = annotations[0][stage_idx1[0]]
        time_end = annotations[0][stage_idx1[-1]] + annotations[1][
            stage_idx1[-1]]
        num1 = []
        k = 0
        for j in range(len(stages1)):
            if stages1[j] != -1:
                num1 += [j]
                k += 1
        raw.close()

        print(psg_name[l])
        raw1 = pyedflib.EdfReader(psg_name[l])
        labels1 = raw1.getSignalLabels()
        signal_headers1 = raw1.getSignalHeaders()

        print('\n')
        print('Filename: {}'.format(psg_name[l]))
        print('loading the raw_data:\n')

        #EEG
        eeg_channel1 = 'EEG Fpz-Cz'
        eeg_channel2 = 'EEG Pz-Oz'
        eeg0 = labels1.index(eeg_channel1)
        eeg1 = labels1.index(eeg_channel2)
        eeg_header = signal_headers1[eeg1]
        fs_eeg = int(eeg_header['sample_rate'])
        print('EEG sample rate: {}'.format(fs_eeg))
        eeg_signal1 = raw1.readSignal(int(eeg0),
                                      int(time_begin) * fs_eeg,
                                      len(stages1) * epoch * fs_eeg)
        eeg_signal2 = raw1.readSignal(int(eeg1),
                                      int(time_begin) * fs_eeg,
                                      len(stages1) * epoch * fs_eeg)
        print('-------------------------------\n')

        #EOG
        eog_channel1 = 'EOG horizontal'
        eog = labels1.index(eog_channel1)
        print('EOG channel1: {}'.format(eog_channel1))
        eog_header = signal_headers1[int(eog)]
        fs_eog = int(eog_header['sample_rate'])
        print('EOG sample rate: {}'.format(fs_eog))
        eog_signal1 = raw1.readSignal(int(eog),
                                      int(time_begin) * fs_eog,
                                      len(stages1) * epoch * fs_eog)
        print('-------------------------------\n')

        #EMG
        emg_channel1 = 'EMG submental'
        if emg_channel1 in labels1:
            emg = labels1.index(emg_channel1)
            print('EMG channel1: {}'.format(emg_channel1))
            #emg_header = signal_headers1[int(emg[1])]
            emg_header = signal_headers1[emg]
            fs_emg = int(emg_header['sample_rate'])
            print('EMG sample rate: {}'.format(fs_emg))
            emg_signal = raw1.readSignal(emg,
                                         int(time_begin) * fs_emg,
                                         len(stages1) * epoch * fs_emg)
        else:
            print('No EMG ')
        emg_signal = emg_signal * 10**6
        print('-------------------------------\n')

        #滤波
        b1, a1 = signal.butter(2, [0.3 * 2 / fs_eeg, 35 * 2 / fs_eeg],
                               'bandpass')
        eeg_signal1 = signal.filtfilt(b1, a1, eeg_signal1)
        eeg_signal2 = signal.filtfilt(b1, a1, eeg_signal2)

        b2, a2 = signal.butter(2, [0.3 * 2 / fs_eog, 35 * 2 / fs_eog],
                               'bandpass')
        eog_signal1 = signal.filtfilt(b2, a2, eog_signal1)
        '''
        #去均值
        eeg_signal = eeg_signal - np.mean(eeg_signal)
        eog_signal = eog_signal - np.mean(eog_signal)
        emg_signal = emg_signal - np.mean(emg_signal)
        '''

        #降采样
        eeg_signal1 = librosa.resample(y=np.squeeze(eeg_signal1),
                                       orig_sr=fs_eeg,
                                       target_sr=fs)
        eeg_signal2 = librosa.resample(y=np.squeeze(eeg_signal2),
                                       orig_sr=fs_eeg,
                                       target_sr=fs)
        eog_signal1 = librosa.resample(y=np.squeeze(eog_signal1),
                                       orig_sr=fs_eog,
                                       target_sr=fs)

        #改变形状
        eeg_signal1 = np.reshape(eeg_signal1, [-1, epoch * fs])
        eeg_signal2 = np.reshape(eeg_signal2, [-1, epoch * fs])
        eog_signal1 = np.reshape(eog_signal1, [-1, epoch * fs])

        eeg_signal1 = preprocessing.scale(eeg_signal1)
        eeg_signal2 = preprocessing.scale(eeg_signal2)
        eog_signal1 = preprocessing.scale(eog_signal1)

        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        y = []
        for i in range(len(num1)):
            x1 += [eeg_signal1[num1[i], :]]
            x2 += [eeg_signal2[num1[i], :]]
            x3 += [eog_signal1[num1[i], :]]
            # x5 += [emg_signal1[num1[i],:]]
            y += [stages1[num1[i]]]

        x1 = np.array(x1)
        x1 = np.reshape(x1, [-1, epoch * fs])
        x2 = np.array(x2)
        x2 = np.reshape(x2, [-1, epoch * fs])
        x3 = np.array(x3)
        x3 = np.reshape(x3, [-1, epoch * fs])
        x5 = np.array(x5)
        x5 = np.reshape(x5, [-1, epoch * fs])
        y = np.array(y)

        y = y.astype(np.int32)
        y = np.reshape(y, [-1, 1])

        raw1.close()

        filename = os.path.split(psg_name[l])[-1]
        '''
        #STFT
        win_size  = 2
        overlap = 1
        nfft = 256
        #EEG STFT
        n_eeg = len(x1[:,1])
        x_eeg = np.zeros(((int(n_eeg), 29, 129)))
        eps=1e-6
        for p in range(n_eeg):
            f_eeg, t_eeg, zxx_eeg = signal.stft(x1[p,:], nperseg = win_size*fs, noverlap = overlap*fs, nfft = nfft, boundary = None)
            zxx_eeg = 20*(np.log10(abs(zxx_eeg)+eps))
            zxx_eeg = np.transpose(zxx_eeg)
            x_eeg[p,:,:] = zxx_eeg       
        
        #EOG STFT
        n_eog = len(x2[:,1])
        x_eog = np.zeros(((int(n_eog), 29, 129)))
        for p in range(n_eog):
            f_eog, t_eog, zxx_eog = signal.stft(x2[p,:], nperseg = win_size*fs, noverlap = overlap*fs, nfft = nfft, boundary = None)
            zxx_eog = 20*(np.log10(abs(zxx_eog)+eps))
            zxx_eog = np.transpose(zxx_eog)
            x_eog[p,:,:] = zxx_eeg       
        
        #EMG STFT
        n_emg = len(x3[:,1])
        x_emg = np.zeros(((int(n_emg), 29, 129)))
        for p in range(n_emg):
            f_emg, t_emg, zxx_emg = signal.stft(x3[p,:], nperseg = win_size*fs, noverlap = overlap*fs, nfft = nfft, boundary = None)
            zxx_emg = 20*(np.log10(abs(zxx_emg)+eps))
            zxx_emg = np.transpose(zxx_emg)
            x_emg[p,:,:] = zxx_emg       
        '''
        filename_seq = ntpath.basename(filename).replace(".edf", ".mat")
        sio.savemat(
            os.path.join(output_dir, filename_seq),
            {
                "eeg1fpz": x1,
                "eeg2pz": x2,
                "eog": x3,
                # "emg": x5,
                "y": y,
            })
        print('\n Finish')
        print("\n=======================================\n")
