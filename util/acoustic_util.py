import os

import numpy as np

from src.base_class.vocoder.htk_io import HTK_Parm_IO
from util import file_util, log_util

log = log_util.get_logger("acoustic tool")


def interpolate_f0(data):
    '''
        interpolate F0, if F0 has already been interpolated, nothing will be changed after passing this function
    :param data:
    :return:
    '''
    data = np.reshape(data, (data.size, 1))
    vuv_vector = np.zeros((data.size, 1))
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0
    ip_data = data
    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]
    return ip_data, vuv_vector


def compute_dynamic_vector(vector, dynamic_win, frame_number=None):
    '''
            compute dynamic vector
            delta_win = [-0.5, 0.0, 0.5]
            acc_win   = [1.0, -2.0, 1.0]
    :param vector:
    :param dynamic_win:
    :param frame_number:
    :return:
    '''
    flag = True
    if frame_number is None:
        frame_number = vector.size
        flag = False
    vector = np.reshape(vector, (frame_number, 1))
    win_length = len(dynamic_win)
    win_width = int(win_length / 2)
    temp_vector = np.zeros((frame_number + 2 * win_width, 1))
    delta_vector = np.zeros((frame_number, 1))
    if flag:
        temp_vector[win_width:frame_number + win_width] = vector
    else:
        temp_vector[win_width:frame_number + win_width, ] = vector
    for w in range(win_width):
        temp_vector[w, 0] = vector[0, 0]
        temp_vector[frame_number + win_width + w, 0] = vector[frame_number - 1, 0]
    for i in range(frame_number):
        for w in range(win_length):
            delta_vector[i] += temp_vector[i + w, 0] * dynamic_win[w]
    return delta_vector


def compute_delta(vector, delta_win):
    #        delta_win = [-0.5, 0.0, 0.5]
    #        acc_win   = [1.0, -2.0, 1.0]
    frame_number = vector.size
    win_length = len(delta_win)
    win_width = int(win_length / 2)
    temp_vector = np.zeros((frame_number + 2 * win_width, 1))
    delta_vector = np.zeros((frame_number, 1))
    temp_vector[win_width:frame_number + win_width, ] = vector
    for w in range(win_width):
        temp_vector[w, 0] = vector[0, 0]
        temp_vector[frame_number + win_width + w, 0] = vector[frame_number - 1, 0]
    for i in range(frame_number):
        for w in range(win_length):
            delta_vector[i] += temp_vector[i + w, 0] * delta_win[w]
    return delta_vector


def load_cmp_file(file_name, mgc_dim, bap_dim, lf0_dim):
    '''
        cmp_norm = CMPNormalisation(mgc_dim=50, bap_dim=25, lf0_dim=1)
        self.mgc_dim = mgc_dim * 3
        self.bap_dim = bap_dim * 3
        self.lf0_dim = lf0_dim * 3

    :param file_name:
    :param mgc_dim:
    :param bap_dim:
    :param lf0_dim:
    :return:
    '''
    mgc_dim = mgc_dim * 3
    bap_dim = bap_dim * 3
    lf0_dim = lf0_dim * 3

    htk_reader = HTK_Parm_IO()
    htk_reader.read_htk(file_name)
    cmp_data = htk_reader.data
    mgc_data = cmp_data[:, 0:mgc_dim]
    # this only extracts the static lf0 because we need to interpolate it, then add deltas ourselves later
    lf0_data = cmp_data[:, mgc_dim]
    bap_data = cmp_data[:, mgc_dim + lf0_dim:mgc_dim + lf0_dim + bap_dim]
    log.debug('loaded %s of shape %s' % (file_name, cmp_data.shape))
    log.debug('  with: %d mgc + %d lf0 + %d bap = %d' % (
        mgc_dim, lf0_dim, bap_dim, mgc_dim + lf0_dim + bap_dim))
    assert ((mgc_dim + lf0_dim + bap_dim) == cmp_data.shape[1])
    return mgc_data, bap_data, lf0_data


def produce_nn_cmp(in_file_list, out_file_list):
    delta_win = [-0.5, 0.0, 0.5]
    acc_win = [1.0, -2.0, 1.0]
    file_number = len(in_file_list)
    for i in range(file_number):
        mgc_data, bap_data, lf0_data = load_cmp_file(in_file_list[i])
        ip_lf0, vuv_vector = interpolate_f0(lf0_data)
        delta_lf0 = compute_delta(ip_lf0, delta_win)
        acc_lf0 = compute_delta(ip_lf0, acc_win)
        frame_number = ip_lf0.size
        cmp_data = np.concatenate((mgc_data, ip_lf0, delta_lf0, acc_lf0, vuv_vector, bap_data), axis=1)
        file_util.array_to_binary_file(cmp_data, out_file_list[i])
    log.info('finished creation of %d binary files' % file_number)


def acoustic_decomposition(in_file_list, out_dimension_dict, file_extension_dict):
    stream_start_index = {}
    dimension_index = 0
    recorded_vuv = False
    vuv_dimension = None
    for feature_name in list(out_dimension_dict.keys()):
        if feature_name != 'vuv':
            stream_start_index[feature_name] = dimension_index
        else:
            vuv_dimension = dimension_index
            recorded_vuv = True
        dimension_index += out_dimension_dict[feature_name]
    for file_name in in_file_list:
        dir_name = os.path.dirname(file_name)
        file_id = os.path.splitext(os.path.basename(file_name))[0]
