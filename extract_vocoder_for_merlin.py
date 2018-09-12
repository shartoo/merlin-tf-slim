import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

from tool_packages.magphase import libutils as lu
from tool_packages.magphase import magphase as mp
from util import file_util, log_util

log = log_util.get_logger("extract vocoder features")

fs_nFFT_dict = {16000: 1024,
                22050: 1024,
                44100: 2048,
                48000: 2048}
fs_alpha_dict = {16000: 0.58,
                 22050: 0.65,
                 44100: 0.76,
                 48000: 0.77}

raw_dir = "raw"
sp_dir = "sp"
mgc_dir = "mgc"
bap_dir = "bap"
ap_dir = "ap"
f0_dir = "f0"
lf0_dir = "lf0"
# out_feat_dir must contain all of above feature name
feat_dir = ["raw", "sp", "mgc", "bap", "ap", "f0", "lf0"]

merlin_dir = ""
straight = os.path.join(merlin_dir, "tools/bin/straight")
world = os.path.join(merlin_dir, "tools/bin/WORLD")
worldv2 = os.path.join(merlin_dir, "tools/bin/WORLD")
sptk = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")
reaper = os.path.join(merlin_dir, "tools/bin/REAPER")
magphase = os.path.join(merlin_dir, 'tools', 'magphase', 'src')


def extract_vocoder_feats_for_merlin(merlin_path, vocoder_type, wav_dir, out_dir, sample_rate):
    '''
        extract vocoder feature for merlin with different vocoder type
    :param merlin_path:         root dir of merlin
    :param vocoder_type:        type of vocoder,possible value are maghase,straight,world,world2
    :param wav_dir:             wav file path to extract
    :param out_dir:             root dir to save extracted features
    :param sample_rate:         sample rate of radio, possible value are 16000,44100,48000
    :return:
    '''
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    path_list = [os.path.join(out_dir, feat_dir) for feat_dir in os.listdir(feat_dir)]
    file_util.create_path_list(path_list)
    print("--- Feature extraction started ---")
    start_time = time.time()

    # get wav files list
    wav_files = file_util.read_file_list_from_path(wav_dir, ".wav", True)
    process = None
    params = None
    if vocoder_type == "magphase":
        sys.path.append(os.path.realpath(magphase))
    elif vocoder_type == "straight":
        process = extract_feats_by_straight
        params = [straight, sptk, wav_files, sample_rate]
    elif vocoder_type == "world":

    # do multi-processing
    pool = mp.Pool(mp.cpu_count())
    pool.map(process, params)

    # clean temporal files
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(sp_dir, ignore_errors=True)
    shutil.rmtree(f0_dir, ignore_errors=True)
    shutil.rmtree(ap_dir, ignore_errors=True)

    print("You should have your features ready in: " + out_dir)
    (m, s) = divmod(int(time.time() - start_time), 60)
    print(("--- Feature extraction completion time: %d min. %d sec ---" % (m, s)))


def extract_feats_by_magphase(magphase, wav_dir, out_dir):
    '''
        extract vocoder features by magphase
    :param merlin_dir:
    :param wav_dir:
    :param out_dir:
    :return:
    '''
    sys.path.append(os.path.realpath(magphase))
    lu.mkdir(out_dir)
    l_wavfiles = file_util.read_file_list_from_path(wav_dir, file_type=".wav", if_recursive=True)

    def feat_extraction(wav_file, out_feats_dir):
        # Parsing path:
        file_name_token = os.path.basename(os.path.splitext(wav_file)[0])
        log.debug("Analysing file: " + file_name_token + '.wav' + "................................")
        mp.analysis_for_acoustic_modelling(wav_file, out_feats_dir)
        return

    # MULTIPROCESSING EXTRACTION
    lu.run_multithreaded(feat_extraction, l_wavfiles, out_dir)


def extract_feats_by_straight(straight, sptk, wav_file, sample_rate):
    '''
        extract vocoder feature by straight
    :param merlin_dir:
    :param wav_dir:
    :param out_dir:
    :param sample_rate:
    :return:
    '''
    file_id = os.path.basename(wav_file).split(".")[0]
    print(file_id)
    nFFT = fs_nFFT_dict[sample_rate]
    alpha = fs_alpha_dict[sample_rate]
    mcsize = 59
    order = 24
    nFFTHalf = 1 + nFFT / 2
    fshift = 5
    sox_wav_2_raw_cmd = 'sox %s -b 16 -c 1 -r %s -t raw %s' % (wav_file, \
                                                               sample_rate, \
                                                               os.path.join(raw_dir, file_id + '.raw'))
    os.system(sox_wav_2_raw_cmd)
    ### STRAIGHT ANALYSIS -- extract vocoder parameters ###
    ### extract f0, sp, ap ###
    straight_f0_analysis_cmd = "%s -nmsg -maxf0 400 -uf0 400 -minf0 50 -lf0 50 -f0shift %s -f %s -raw %s %s" % (
        os.path.join(straight, 'tempo'), \
        fshift, sample_rate, \
        os.path.join(raw_dir, file_id + '.raw'), \
        os.path.join(f0_dir, file_id + '.f0'))
    os.system(straight_f0_analysis_cmd)

    straight_ap_analysis_cmd = "%s -nmsg -f %s -fftl %s -apord %s -shift %s -f0shift %s -float -f0file %s -raw %s %s" % (
        os.path.join(straight, 'straight_bndap'), \
        sample_rate, nFFT, nFFTHalf, fshift, fshift, \
        os.path.join(f0_dir, file_id + '.f0'), \
        os.path.join(raw_dir, file_id + '.raw'), \
        os.path.join(ap_dir, file_id + '.ap'))
    os.system(straight_ap_analysis_cmd)

    straight_sp_analysis_cmd = "%s -nmsg -f %s -fftl %s -apord %s -shift %s -f0shift %s -order %s -f0file %s -pow -float -raw %s %s" % (
        os.path.join(straight, 'straight_mcep'), \
        sample_rate, nFFT, nFFTHalf, fshift, fshift, mcsize, \
        os.path.join(f0_dir, file_id + '.f0'), \
        os.path.join(raw_dir, file_id + '.raw'), \
        os.path.join(sp_dir, file_id + '.sp'))

    os.system(straight_sp_analysis_cmd)

    ### convert f0 to lf0 ###
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    lf0_file = os.path.join(lf0_dir, file_id + '.lf0')
    f0_to_lf0(f0_file, lf0_file)

    ### convert sp to mgc ###
    file_in = os.path.join(sp_dir, file_id + '.sp')
    file_out = os.path.join(mgc_dir, file_id + '.mgc')
    sptk_mcep_cmd(3, alpha, mcsize, nFFT, file_in, file_out)
    ### convert ap to bap ###
    file_in = os.path.join(ap_dir, file_id + '.ap')
    file_out = os.path.join(bap_dir, file_id + '.bap')
    sptk_mcep_cmd(1, alpha, mcsize, nFFT, file_in, file_out)

def extract_feat_by_world(wav_file, sample_rate, b_use_reaper=True):
    ''''''

    nFFTHalf = fs_nFFT_dict[sample_rate]
    alpha = fs_alpha_dict[sample_rate]
    mcsize = 59
    file_id = os.path.basename(wav_file).split(".")[0]
    print('\n' + file_id)

    ### WORLD ANALYSIS -- extract vocoder parameters ###
    ### extract sp, ap ###
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    f0_world_file = f0_file
    if b_use_reaper:
        f0_world_file = f0_file + "_world"
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    sp_file = os.path.join(sp_dir, file_id + '.sp')
    bapd_file = os.path.join(bap_dir, file_id + '.bapd')
    world_analysis(wav_file, f0_file, sp_file, bapd_file)
    ### Extract f0 using reaper ###
    if b_use_reaper:
        reaper_f0_extract(wav_file, f0_world_file, f0_file)
    ### convert f0 to lf0 ###
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    lf0_file = os.path.join(lf0_dir, file_id + '.lf0')
    f0_to_lf0(f0_file, lf0_file)
    ### convert sp to mgc ###
    sp_file = os.path.join(sp_dir, file_id + '.sp')
    mgc_file = os.path.join(mgc_dir, file_id + '.mgc')
    sp_to_mgc(sp_file, mgc_file, alpha, mcsize, nFFTHalf)

    ### convert bapd to bap ###
    sptk_x2x_df_cmd2 = "%s +df %s > %s " % (os.path.join(sptk, "x2x"), \
                                            os.path.join(bap_dir, file_id + ".bapd"), \
                                            os.path.join(bap_dir, file_id + '.bap'))
    os.system(sptk_x2x_df_cmd2)


def extract_feat_by_worldv2(wav_file, sample_rate):
    '''

    :param wav_file:
    :param sample_rate:
    :return:
    '''
    nFFTHalf = fs_nFFT_dict[sample_rate]
    alpha = fs_alpha_dict[sample_rate]
    mcsize = 59
    order = 4
    file_id = os.path.basename(wav_file).split(".")[0]
    print('\n' + file_id)
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    sp_file = os.path.join(sp_dir, file_id + '.sp')
    ap_file = os.path.join(bap_dir, file_id + '.ap')
    world_analysis(wav_file, f0_file, sp_file, ap_file)
    ### convert f0 to lf0 ###
    f0_file = os.path.join(f0_dir, file_id + '.f0')
    lf0_file = os.path.join(lf0_dir, file_id + '.lf0')
    f0_to_lf0(f0_file, lf0_file)
    ### convert sp to mgc ###
    sp_file = os.path.join(sp_dir, file_id + '.sp')
    mgc_file = os.path.join(mgc_dir, file_id + '.mgc')
    sp_to_mgc(sp_file, mgc_file, alpha, mcsize, nFFTHalf)
    ### convert ap to bap ###
    sptk_x2x_df_cmd2 = "%s +df %s | %s | %s >%s" % (os.path.join(sptk, 'x2x'), \
                                                    os.path.join(sp_dir, file_id + '.ap'), \
                                                    os.path.join(sptk, 'sopr') + ' -R -m 32768.0', \
                                                    os.path.join(sptk, 'mcep') + ' -a ' + str(alpha) + ' -m ' + str(
                                                        order) + ' -l ' + str(
                                                        nFFTHalf) + ' -e 1.0E-8 -j 0 -f 0.0 -q 3 ', \
                                                    os.path.join(mgc_dir, file_id + '.bap'))
    os.system(sptk_x2x_df_cmd2)


def sptk_mcep_cmd(cmd_order, alpha, mcsize, nFFT, file_in, file_out):
    '''

    :param cmd_order:  3 or 1
    :param alpha:
    :param mcsize:
    :param nFFT:
    :param file_in:
    :param file_out:
    :return:
    '''
    sptk_mcep = "%s -a %s -m %s -l %s -e 1.0E-8 -j 0 -f 0.0 -q %d %s > %s" \
                % (os.path.join(sptk, 'mcep'), alpha, mcsize, nFFT, int(cmd_order), file_in, file_out)
    os.system(sptk_mcep)


def world_analysis(wav_file, f0_file, out1, out2):
    '''
        world analysis process,both for world and worldv2
    :param wav_file:       original wav file
    :param f0_file:
    :param out1:        ".sp" file
    :param out2:        for worldv2 is ".ap", for world is ".bapd"
    :return:
    '''
    file_id = os.path.basename(wav_file).split(".")[0]
    world_analysis_cmd = "%s %s %s %s %s" % (os.path.join(world, 'analysis'), wav_file, f0_file, out1, out2)
    os.system(world_analysis_cmd)

def f0_to_lf0(f0_file, lf0_file):
    '''
        convert f0 file to lf0
    :param :f0_file
    :param lf0_file:
    :return:
    '''
    sptk_x2x_af_cmd = "%s +af %s | %s > %s " % (os.path.join(sptk, 'x2x'), \
                                                f0_file, \
                                                os.path.join(sptk, 'sopr') + ' -magic 0.0 -LN -MAGIC -1.0E+10', \
                                                lf0_file)
    os.system(sptk_x2x_af_cmd)


def sp_to_mgc(sp_file, mgc_file, alpha, mcsize, nFFTHalf):
    '''

    :param sp_file:
    :param mgc_file:
    :return:
    '''

    sptk_x2x_df_cmd1 = "%s +df %s | %s | %s >%s" % (os.path.join(sptk, 'x2x'), \
                                                    sp_file, \
                                                    os.path.join(sptk, 'sopr') + ' -R -m 32768.0', \
                                                    os.path.join(sptk, 'mcep') + ' -a ' + str(alpha) + ' -m ' + str(
                                                        mcsize) + ' -l ' + str(
                                                        nFFTHalf) + ' -e 1.0E-8 -j 0 -f 0.0 -q 3 ', \
                                                    mgc_file)
    os.system(sptk_x2x_df_cmd1)


#########used for world vocoder #######

def read_reaper_f0_file(est_file, skiprows=7):
    '''
    Reads f0 track into numpy array from EST file generated by REAPER.
    '''
    v_f0 = np.loadtxt(est_file, skiprows=skiprows, usecols=[2])
    v_f0[v_f0 < 0] = 0
    return v_f0


def reaper_f0_extract(in_wavfile, f0_file_ref, f0_file_out, frame_shift_ms=5.0):
    '''
    Extracts f0 track using REAPER.
    To keep consistency with the vocoder, it also fixes for the difference in number
    of frames between the REAPER f0 track and the acoustic parameters extracted by the vocoder.
    f0_file_ref: f0 extracted by the vocoder. It is used as a reference to fix the number of frames, as explained.
    '''

    # Run REAPER:
    log.debug("Running REAPER f0 extraction...")
    cmd = "%s -a -s -x 400 -m 50 -u %1.4f -i %s -f %s" % (
        os.path.join(reaper, 'reaper'), frame_shift_ms / 1000.0, in_wavfile, f0_file_out + "_reaper")
    os.system(cmd)

    # Protection - number of frames:
    v_f0_ref = file_util.read_binfile(f0_file_ref, dim=1)
    v_f0 = read_reaper_f0_file(f0_file_out + "_reaper")
    frm_diff = v_f0.size - v_f0_ref.size
    if frm_diff < 0:
        v_f0 = np.r_[v_f0, np.zeros(-frm_diff) + v_f0[-1]]
    if frm_diff > 0:
        v_f0 = v_f0[:-frm_diff]

    # Save f0 file:
    file_util.write_binfile(v_f0, f0_file_out)
    return
