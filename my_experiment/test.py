# from python_speech_features  import mfcc
# from python_speech_features  import logfbank
# import scipy.io.wavfile as wav
# from matplotlib import pyplot as plt
# (rate,sig) = wav.read(r"I:\newwork\tacotron-tts\test.wav")
# mfcc_feat = mfcc(sig,rate)
# fbank_feat = logfbank(sig,rate)
# plt.plot(mfcc_feat)
# plt.show()

import os


def write_txt_from_wav_file(wav_path, save_path):
    '''
        wav file name is the content of text
    :param wav_path:
    :param save_path:
    :return:
    '''
    wavs = os.listdir(wav_path)
    texts = []
    for wav in wavs:
        if wav.endswith(".wav"):
            name = os.path.basename(wav).split(".")[0]
            texts.append(name)
            with open(os.path.join(save_path, name + ".txt"), "w") as wt:
                wt.write(name)
    print("write wavfile name to text done..")


wav_path = r"I:\data\voice\CarNumLabeled\wavs"
save_path = r"I:\data\voice\CarNumLabeled\text"
write_txt_from_wav_file(wav_path, save_path)
