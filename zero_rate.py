import math
import wave

import numpy as np
import pylab as pl


def zeroRate(waveData, frameSize, overlap):
    '''

    :param waveData:    音频中读取的数据
    :param frameSize:   帧长度
    :param overlap:     帧移
    :return:
    '''
    wlen = len(waveData)
    step = frameSize - overlap
    frameNum = math.ceil(wlen / step)
    zrc = np.zeros((frameNum, 1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
        curFrame = curFrame - np.mean(curFrame)  # zero justified
        zrc[i] = sum(curFrame[0:-1] * curFrame[1::] <= 0)

    return zrc


def shortTimeEngerny(frame):
    print("shape is\t", frame.shape)
    print(frame)
    return sum([abs(x) ** 2 for x in frame]) / len(frame)


fw = wave.open(r"I:\data\voice\data_thchs30\data\A2_0.wav", 'rb')
params = fw.getparams()
print(params)
nchannels, sampleWidth, frameRate, nFrames = params[:4]
str_data = fw.readframes(nFrames)
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 1
fw.close()
# calcuate zero cross rate
frameSize = 256
overlap = 0
zrc = zeroRate(wave_data, frameSize, overlap)
# plot the wav
time = np.arange(0, len(wave_data)) * (1.0 / frameRate)
time2 = np.arange(0, len(zrc)) * (len(wave_data) / len(zrc) / frameRate)
pl.subplot(311)
pl.plot(time, wave_data)
pl.ylabel("Amplitude")
pl.subplot(312)
pl.plot(time2, zrc)
pl.ylabel("ZCR")
pl.xlabel("time (seconds)")
pl.subplot(313)
print(wave_data)
frame = wave_data
print(frame)
pl.plot(time, shortTimeEngerny(wave_data[1]))
pl.ylabel("short-time engerny")
pl.xlabel("time (seconds)")
pl.show()
