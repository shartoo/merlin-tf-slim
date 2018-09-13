import multiprocessing as mp
import os
import shutil
import time

import fastdtw
import numpy as np

from util import file_util, system_cmd_util, log_util

log = log_util.get_logger("dtw align")
# path to tools
tools_dir = ""
sptk = os.path.join(tools_dir, "bin/SPTK-3.9")
speech_tools = os.path.join(tools_dir, "speech_tools/bin")
festvox = os.path.join(tools_dir, "festvox")

tools_dir = ""
# Source features directory
src_feat_dir = ""
# Target features directory
tgt_feat_dir = ""
# Source-aligned features directory
src_aligned_feat_dir = ""
# bap dimension
bap_dim = ""
temp_dir = os.path.join(src_aligned_feat_dir, "../temp")
if not os.path.exists(src_aligned_feat_dir):
    os.makedirs(src_aligned_feat_dir)
alignments_dir = os.path.join(src_aligned_feat_dir, "../dtw_alignments")

src_mgc_dir = os.path.join(src_feat_dir, "mgc")
tgt_mgc_dir = os.path.join(tgt_feat_dir, "mgc")

src_bap_dir = os.path.join(src_feat_dir, "bap")
tgt_bap_dir = os.path.join(tgt_feat_dir, "bap")

src_lf0_dir = os.path.join(src_feat_dir, "lf0")
tgt_lf0_dir = os.path.join(tgt_feat_dir, "lf0")

src_aligned_mgc_dir = os.path.join(src_aligned_feat_dir, "mgc")
src_aligned_bap_dir = os.path.join(src_aligned_feat_dir, "bap")
src_aligned_lf0_dir = os.path.join(src_aligned_feat_dir, "lf0")

feat_list = [src_mgc_dir, tgt_mgc_dir, src_bap_dir, tgt_bap_dir, src_lf0_dir, tgt_lf0_dir, src_aligned_mgc_dir,
             src_aligned_bap_dir, src_aligned_lf0_dir]
file_util.create_path_list(feat_list)
file_util.create_blank_file("touch %s/i.lab" % (temp_dir))
file_util.create_blank_file("touch %s/o.lab" % (temp_dir))


def align(mgc_dir, dtw_type="default"):
    '''
        entrance of doing align
    :param mgc_dir:
    :return:
    '''
    # get mag files list
    mgc_files = file_util.read_file_list_from_path(mgc_dir, file_type=".mgc", if_recursive=True)

    # do multi-processing
    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    if dtw_type == "default":
        pool.map(align_default, mgc_files)

    (m, s) = divmod(int(time.time() - start_time), 60)
    log.info(("--- DTW alignment completion time: %d min. %d sec ---" % (m, s)))
    # clean temporal files
    shutil.rmtree(alignments_dir, ignore_errors=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.exists(src_aligned_feat_dir):
        log.error("DTW alignment unsucessful!!")
    else:
        log.info("You should have your src feats(aligned with target) ready in: %s" % (src_aligned_feat_dir))


def align_src_feats(src_feat_file, src_aligned_feat_file, feat_dim, dtw_path_dict):
    '''
    align source feats as per the dtw path (matching target length)
    '''
    src_features, frame_number = file_util.load_binary_file_frame(src_feat_file, feat_dim)
    tgt_length = len(dtw_path_dict)
    src_aligned_features = np.zeros((tgt_length, feat_dim))
    for i in range(tgt_length):
        src_aligned_features[i,] = src_features[dtw_path_dict[i]]
    file_util.array_to_binary_file(src_aligned_features, src_aligned_feat_file)


def load_dtw_path(dtw_path):
    dtw_path_dict = {}
    nframes = len(dtw_path)
    for item, i in zip(dtw_path, range(nframes)):
        if item[1] not in dtw_path_dict:
            dtw_path_dict[item[1]] = item[0]

    return dtw_path_dict


def align_default(src_mgc_file, tgt_mgc_file):
    '''

    :param mgc_file:
    :return:
    '''
    mgc_dim = 60
    lf0_dim = 1
    file_id = os.path.basename(src_mgc_file).split(".")[0]
    print(file_id)
    ### DTW alignment -- align source with target parameters ###
    src_features, src_frame_number = file_util.load_binary_file_frame(src_mgc_file, mgc_dim)
    tgt_features, tgt_frame_number = file_util.load_binary_file_frame(tgt_mgc_file, mgc_dim)
    ### dtw align src with tgt ###
    distance, dtw_path = fastdtw.fastdtw(src_features, tgt_features)
    ### load dtw path
    dtw_path_dict = load_dtw_path(dtw_path)
    assert len(dtw_path_dict) == tgt_frame_number  # dtw length not matched
    ### align features
    align_src_feats(os.path.join(src_mgc_dir, file_id + ".mgc"),
                    os.path.join(src_aligned_mgc_dir, file_id + ".mgc"), mgc_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_bap_dir, file_id + ".bap"),
                    os.path.join(src_aligned_bap_dir, file_id + ".bap"), bap_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_lf0_dir, file_id + ".lf0"),
                    os.path.join(src_aligned_lf0_dir, file_id + ".lf0"), lf0_dim, dtw_path_dict)


def align_by_festvox(src_mgc_file, tgt_mgc_file):
    '''

    :param src_mgc_file:
    :param tgt_mgc_file:
    :return:
    '''
    mgc_dim = 60
    lf0_dim = 1
    file_id = os.path.basename(src_mgc_file).split(".")[0]
    src_features, src_frame_number = file_util.load_binary_file_frame(src_mgc_file, mgc_dim)
    tgt_features, tgt_frame_number = file_util.load_binary_file_frame(tgt_mgc_file, mgc_dim)
    ### dtw align src with tgt ###
    dtw_alignment_file = os.path.join(alignments_dir, file_id + ".dtw")
    src_ascii_mgc = os.path.join(temp_dir, file_id + "_src_ascii.mgc")
    src_binary_mgc = os.path.join(temp_dir, file_id + "_src_binary.mgc")
    system_cmd_util.sptk_x2x_xargs(sptk, src_mgc_file, mgc_dim, src_ascii_mgc)
    tgt_ascii_mgc = os.path.join(temp_dir, file_id + "_tgt_ascii.mgc")
    tgt_binary_mgc = os.path.join(temp_dir, file_id + "_tgt_binary.mgc")
    system_cmd_util.sptk_x2x_xargs(sptk, tgt_mgc_file, mgc_dim, tgt_ascii_mgc)
    system_cmd_util.speech_tools_chtrack(speech_tools, src_ascii_mgc, src_binary_mgc)
    system_cmd_util.speech_tools_chtrack(speech_tools, tgt_ascii_mgc, tgt_binary_mgc)
    in_lab = os.path.join(temp_dir, "i.lab")
    out_lab = os.path.join(temp_dir, "o.lab")
    system_cmd_util.festvox_phone_align_cmd(festvox, tgt_binary_mgc, src_binary_mgc, in_lab, out_lab,
                                            dtw_alignment_file)
    ### load dtw path
    dtw_path_dict = file_util.load_ascii_dtw_file(dtw_alignment_file)
    assert len(dtw_path_dict) == tgt_frame_number  # dtw length not matched
    ### align features
    align_src_feats(os.path.join(src_mgc_dir, file_id + ".mgc"),
                    os.path.join(src_aligned_mgc_dir, file_id + ".mgc"), mgc_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_bap_dir, file_id + ".bap"),
                    os.path.join(src_aligned_bap_dir, file_id + ".bap"), bap_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_lf0_dir, file_id + ".lf0"),
                    os.path.join(src_aligned_lf0_dir, file_id + ".lf0"), lf0_dim, dtw_path_dict)


def align_by_magphase_festvox(src_mag_file, tgt_mag_file):
    mag_dim = 60
    real_dim = 45
    imag_dim = 45
    lf0_dim = 1
    file_id = os.path.basename(src_mag_file).split(".")[0]
    print(file_id)

    ### DTW alignment -- align source with target parameters ###
    src_features, src_frame_number = file_util.load_binary_file_frame(src_mag_file, mag_dim)
    tgt_features, tgt_frame_number = file_util.load_binary_file_frame(tgt_mag_file, mag_dim)
    ### dtw align src with tgt ###
    dtw_alignment_file = os.path.join(alignments_dir, file_id + ".dtw")

    src_ascii_mag = os.path.join(temp_dir, file_id + "_src_ascii.mag")
    src_binary_mag = os.path.join(temp_dir, file_id + "_src_binary.mag")

    tgt_ascii_mag = os.path.join(temp_dir, file_id + "_tgt_ascii.mag")
    tgt_binary_mag = os.path.join(temp_dir, file_id + "_tgt_binary.mag")

    system_cmd_util.sptk_x2x_xargs(sptk, src_mag_file, mag_dim, src_ascii_mag)
    system_cmd_util.sptk_x2x_xargs(sptk, tgt_mag_file, mag_dim, tgt_ascii_mag)
    system_cmd_util.speech_tools_chtrack(speech_tools, src_ascii_mag, src_binary_mag)
    system_cmd_util.speech_tools_chtrack(speech_tools, tgt_ascii_mag, tgt_binary_mag)
    in_lab = os.path.join(temp_dir, "i.lab")
    out_lab = os.path.join(temp_dir, "o.lab")
    system_cmd_util.festvox_phone_align_cmd(festvox, tgt_binary_mag, src_binary_mag, in_lab, out_lab,
                                            dtw_alignment_file)
    ### load dtw path
    dtw_path_dict = file_util.load_ascii_dtw_file(dtw_alignment_file)
    assert len(dtw_path_dict) == tgt_frame_number  # dtw length not matched

    ### align features
    align_src_feats(os.path.join(src_feat_dir, file_id + ".mag"),
                    os.path.join(src_aligned_feat_dir, file_id + ".mag"), mag_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_feat_dir, file_id + ".real"),
                    os.path.join(src_aligned_feat_dir, file_id + ".real"), real_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_feat_dir, file_id + ".imag"),
                    os.path.join(src_aligned_feat_dir, file_id + ".imag"), imag_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_feat_dir, file_id + ".lf0"),
                    os.path.join(src_aligned_feat_dir, file_id + ".lf0"), lf0_dim, dtw_path_dict)


def align_by_magphase(src_mag_file, tgt_mag_file):
    '''

    :param src_mgc:
    :param tgt_mgc:
    :return:
    '''
    mag_dim = 60  # TODO: Change this (avoid hardcoded)
    real_dim = 10
    imag_dim = 10
    lf0_dim = 1

    file_id = os.path.basename(src_mag_file).split(".")[0]
    print(file_id)

    ### DTW alignment -- align source with target parameters ###
    src_features, src_frame_number = file_util.load_binary_file_frame(src_mag_file, mag_dim)
    tgt_features, tgt_frame_number = file_util.load_binary_file_frame(tgt_mag_file, mag_dim)

    ### dtw align src with tgt ###
    distance, dtw_path = fastdtw.fastdtw(src_features, tgt_features)

    ### load dtw path
    dtw_path_dict = load_dtw_path(dtw_path)
    assert len(dtw_path_dict) == tgt_frame_number  # dtw length not matched

    ### align features
    align_src_feats(os.path.join(src_feat_dir, file_id + ".mag"),
                    os.path.join(src_aligned_feat_dir, file_id + ".mag"), mag_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_feat_dir, file_id + ".real"),
                    os.path.join(src_aligned_feat_dir, file_id + ".real"), real_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_feat_dir, file_id + ".imag"),
                    os.path.join(src_aligned_feat_dir, file_id + ".imag"), imag_dim, dtw_path_dict)
    align_src_feats(os.path.join(src_feat_dir, file_id + ".lf0"),
                    os.path.join(src_aligned_feat_dir, file_id + ".lf0"), lf0_dim, dtw_path_dict)
