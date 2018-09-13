import os

from util import log_util

log = log_util.get_logger("system command")


def sptk_x2x_xargs(sptk_path, src_file, file_dim, target):
    x2x_cmd1 = "%s +fa %s | xargs -n%d > %s" % (sptk_path, src_file, file_dim, target)
    log.debug("execuate system command %s" + x2x_cmd1)
    os.system(x2x_cmd1)


def speech_tools_chtrack(speech_tools, src, target):
    '''

    :param speech_tools:
    :param src:
    :param target:
    :return:
    '''
    chtrack_cmd1 = "%s -s 0.005 -otype est_binary %s -o %s" % (os.path.join(speech_tools, "ch_track"), src, target)
    log.debug("execuate system command %s" + chtrack_cmd1)
    os.system(chtrack_cmd1)


def festvox_phone_align_cmd(festvox_path, src_mgc, taret_mgc, in_lab, out_lab, dtw_alignment_file):
    '''

    :param festvox_path: root path of festivox
    :param src_mgc:
    :param taret_mgc:
    :param in_lab:
    :param out_lab:
    :param dtw_alignment_file
    :return:
    '''
    phone_align_cmd = "%s -itrack %s -otrack %s -ilabel %s -olabel %s -verbose -withcosts > %s" % (
        os.path.join(festvox_path, "src/general/phonealign"), \
        src_mgc, taret_mgc, \
        in_lab, out_lab, dtw_alignment_file)
    log.debug("execuate system command %s" + phone_align_cmd)
    os.system(phone_align_cmd)


def sptk_mcep_cmd(sptk, cmd_order, alpha, mcsize, nFFT, file_in, file_out):
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


def world_analysis(world, wav_file, f0_file, out1, out2):
    '''
        world analysis process,both for world and worldv2
    :param wav_file:       original wav file
    :param f0_file:
    :param out1:        ".sp" file
    :param out2:        for worldv2 is ".ap", for world is ".bapd"
    :return:
    '''
    world_analysis_cmd = "%s %s %s %s %s" % (os.path.join(world, 'analysis'), wav_file, f0_file, out1, out2)
    os.system(world_analysis_cmd)


def sptk_f0_to_lf0(sptk, f0_file, lf0_file):
    '''
        convert f0 file to lf0
    :param :f0_file
    :param lf0_file:
    :return:
    '''
    f0a = f0_file.replace(".f0", ".f0a")
    sptk_f0_to_f0a = "%s +da %s>%s" % (os.path.join(sptk, 'x2x'), f0_file, f0a)
    os.system(sptk_f0_to_f0a)
    sptk_x2x_af_cmd = "%s +af %s | %s > %s " % (os.path.join(sptk, 'x2x'), \
                                                f0a, \
                                                os.path.join(sptk, 'sopr') + ' -magic 0.0 -LN -MAGIC -1.0E+10', \
                                                lf0_file)
    os.system(sptk_x2x_af_cmd)


def sptk_lf0_to_f0(sptk, lf0_file, f0_file):
    '''
            convert lf0 file to f0
    :param lf0_file:
    :param f0_file:
    :return:
    '''
    f0a = lf0_file.replace(".lf0", ".f0a")
    lf0_f0_cmd1 = "%s -magic -1.0E+10 -EXP -MAGIC 0.0 %s | %s +fa > %s" % \
                  (os.path.join(sptk, "sopr"), lf0_file, os.path.join(sptk, "x2x"), f0a)
    lf0_f0_cmd2 = "%s +ad %s >%s" % \
                  (os.path.join(sptk, "x2x"), f0a, f0_file)
    os.system(lf0_f0_cmd1)
    os.system(lf0_f0_cmd2)


def sptk_mgc_to_apsp(sptk, alpha, mcsize, nFFTHalf, mgc, apsp):
    '''
      convert mgc to sp: $sptk/mgc2sp -a $alpha -g 0 -m $mcsize -l $nFFTHalf -o 2 ${mgc_dir}/$file_id.mgc |
                             $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > ${resyn_dir}/$file_id.resyn.sp
      convert bap to ap: $sptk/mgc2sp -a $alpha -g 0 -m $order -l $nFFTHalf -o 2 ${bap_dir}/$file_id.bap |
                            $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > ${resyn_dir}/$file_id.resyn.ap
    :param alpha:
    :param mcsize:         mcsize for mgc to sp / order for  bap to ap
    :param nFFTHalf:
    :param mgc:             mgc file or bap file
    :param apsp:            ap for mgc to ap / sp for mgc to sp
    :return:
    '''
    mgc2sp_cmd = "%s -a %f -g 0 -m  %d -l %d -o 2 %s | %s -d 32768.0 -P | %s +fd > %s" % \
                 (os.path.join(sptk, "mgc2sp"), alpha, mcsize, nFFTHalf, mgc, os.path.join(sptk, "sopr"),
                  os.path.join(sptk, "x2x"), apsp)
    os.system(mgc2sp_cmd)


def sptk_sp_to_mgc(sptk, sp_file, mgc_file, alpha, mcsize, nFFTHalf):
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
