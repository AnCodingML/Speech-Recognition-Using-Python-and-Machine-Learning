#coding: utf-8
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms
from sklearn import svm
import pandas as pd
from scipy.io import wavfile
from pylab import*
import matplotlib
import pyaudio
import pygame


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "assalamualaikum1.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

file = 'assalamualaikum1.wav'
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(file)
pygame.mixer.music.play()


#Database": Gerbang Logika AND
#Membaca dara fari file
FileDB = 'database.txt'

#selain dalam format csv , file dapat berbentuk text, misal : logikaand.txt
Database = pd.read_csv(FileDB, sep= " ", header=0)
print(Database)


#x = Data, y = Target
x = Database[[u'Feature1', u'Feature2', u'Feature3', u'Feature4', u'Feature5', u'Feature6', u'Feature7', u'Feature8', u'Feature9', u'Feature10', u'Feature11', u'Feature12']] #ciri1, ciri2, dst
y = Database.Target


#training and classify
clf = svm.SVC()
clf.fit(x,y)


FILEWAV = "assalamualaikum1.wav"

def wavread(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16") / 32768.0
    wf.close()
    return x, float(fs)

def hz2mel(f):
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels):
    fmax = fs / 2
    melmax = hz2mel(fmax)
    nmax = nfft // 2
    df = fs / nfft
    dmel = melmax // (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            i = int (i)
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            i = int (i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters

def preEmphasis(signal, p):
    return scipy.signal.lfilter([1.0, -p], 1, signal)

def mfcc(signal, nfft, fs, nceps):
    p = 0.97
    signal = preEmphasis(signal, p)

    hammingWindow = np.hamming(len(signal))
    signal = signal * hammingWindow

    spec = np.abs(np.fft.fft(signal, nfft))[:nfft//2]
    fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:nfft//2]
    plot(fscale, spec)
    xlabel("frequency [Hz]")
    ylabel("amplitude spectrum")
    savefig("Grafik\spectrum2.png")
    #show()

    numChannels = 20
    df = fs / nfft
    filterbank, fcenters = melFilterBank(fs, nfft, numChannels)

    for c in np.arange(0, numChannels):
        plot(np.arange(0,nfft / 2) * df, filterbank[c])
    savefig("Grafik\melfilterbank8.png")
    #show()

    mspec = []
    for c in np.arange(0, numChannels):
        mspec.append(np.log10(sum(spec * filterbank[c])))
    mspec = np.array(mspec)

    mspec = np.log10(np.dot(spec, filterbank.T))

    subplot(211)
    plot(fscale, np.log10(spec))
    xlabel("frequency")
    xlim(0, 25000)

    subplot(212)
    plot(fcenters, mspec, "o-")
    xlabel("frequency")
    xlim(0,25000)
    savefig("Grafik/result_melfilters8.png")
    #show()

    ceps = scipy.fftpack.realtransforms.dct(mspec, type=2, norm="ortho", axis=-1)

    return ceps[:nceps]

if __name__ == "__main__":
    wav, fs = wavread(FILEWAV)
    t = np.arange(0.0, len(wav) / fs, 1/fs)

    center = len(wav) / 2
    cuttime = 0.04
    z = int(center - cuttime / 2 * fs)
    q = int(center + cuttime / 2 * fs)
    wavdata = wav[z : q]
    nfft = 2048
    nceps = 12
    ceps = mfcc(wavdata, nfft, fs, nceps)
    print ("mfcc:", ceps)

    if clf.predict([ceps]) == 0:
        print("Sama-sama")
        file = 'sama-sama.wav'
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

    elif clf.predict([ceps]) == 1:
        print("waalaikumsalam")
        file = 'waalaikumsalam.wav'
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()


fs = 44100.0;
rate,data = wavfile.read(FILEWAV)
title('Visualusasi Data Audio')
Time = np.linspace(0, len(data)/fs, num=len(data))
plot(Time, data)
xlabel("Time")
ylabel("Amplitude")
savefig("Grafik\soundamplitude8.png")
#show()
