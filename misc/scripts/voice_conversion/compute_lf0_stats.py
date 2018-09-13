import sys

import numpy

from util import file_util


def compute_mean_and_std(lf0_dir):
    '''
    :param lf0_dir:
    :return:
    '''
    lf0_file_list = file_util.read_file_list_from_path(lf0_dir, file_type=".lf0", if_recursive=True)
    all_files_lf0_arr = numpy.zeros(200000)
    current_index = 0
    for lf0_file in lf0_file_list:
        lf0_arr, frame_number = file_util.load_binary_file_frame(lf0_file, 1)
        for lf0_value in lf0_arr:
            all_files_lf0_arr[current_index] = numpy.exp(lf0_value)
            current_index += 1

    all_files_lf0_arr = all_files_lf0_arr[all_files_lf0_arr > 0]
    all_files_lf0_arr = numpy.log(all_files_lf0_arr)

    mean_f0 = numpy.mean(all_files_lf0_arr)
    std_f0 = numpy.std(all_files_lf0_arr)
    return mean_f0, std_f0


if __name__ == "__main__":
    # parse the arguments
    lf0_dir = sys.argv[1]
    lf0_stats_file = sys.argv[2]

    lf0_file_list = get_lf0_filelist(lf0_dir)
    mean_f0, std_f0 = compute_mean_and_std(lf0_file_list)

    out_f = open(lf0_stats_file, 'w')
    out_f.write('%f %f\n' % (mean_f0, std_f0))
    out_f.close()
