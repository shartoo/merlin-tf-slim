import argparse
import os
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

def transform_f0(src_lf0_arr, stats_dict):
    mu_src = stats_dict['mu_src']
    mu_tgt = stats_dict['mu_tgt']

    std_src = stats_dict['std_src']
    std_tgt = stats_dict['std_tgt']

    tgt_lf0_arr = numpy.zeros(len(src_lf0_arr))
    for i in range(len(src_lf0_arr)):
        lf0_src = src_lf0_arr[i]
        f0_src = numpy.exp(lf0_src)
        if f0_src <= 0:
            tgt_lf0_arr[i] = lf0_src
        else:
            tgt_lf0_arr[i] = (mu_tgt + (std_tgt / std_src) * (lf0_src - mu_src))
    return tgt_lf0_arr


def transform_lf0_file(src_lf0_file, tgt_lf0_file, stats_dict):
    src_lf0_arr, frame_number = file_util.load_binary_file_frame(src_lf0_file, 1)
    tgt_lf0_arr = transform_f0(src_lf0_arr, stats_dict)
    file_util.array_to_binary_file(tgt_lf0_arr, tgt_lf0_file)

def transform_lf0_dir(src_lf0_file_list, tgt_lf0_file_list, stats_dict):
    for i in range(len(src_lf0_file_list)):
        src_lf0_file = src_lf0_file_list[i]
        tgt_lf0_file = tgt_lf0_file_list[i]
        transform_lf0_file(src_lf0_file, tgt_lf0_file, stats_dict)

if __name__ == "__main__":
    # parse the arguments
    lf0_dir = sys.argv[1]
    lf0_stats_file = sys.argv[2]
    lf0_file_list = file_util.read_file_by_line(lf0_dir)
    mean_f0, std_f0 = compute_mean_and_std(lf0_file_list)
    out_f = open(lf0_stats_file, 'w')
    out_f.write('%f %f\n' % (mean_f0, std_f0))
    out_f.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('--srcstatsfile', required=True, help='path to source lf0 stats file')
    parser.add_argument('--tgtstatsfile', required=True, help='path to target lf0 stats file')
    parser.add_argument('--srcdir', type=str, help='path to source lf0 data directory')
    parser.add_argument('--tgtdir', type=str, help='path to target lf0 data directory')
    parser.add_argument('--filelist', type=str, help='path to file ID list')
    parser.add_argument('--srcfile', type=str, help='path to source lf0 data file')
    parser.add_argument('--tgtfile', type=str, help='path to target lf0 data file')
    opt = parser.parse_args()

    if opt.srcdir is None and opt.srcfile is None:
        print("at least one of --srcdir and --srcfile is required")
        sys.exit(1)

    if opt.tgtdir is None and opt.tgtfile is None:
        print("at least one of --tgtdir and --tgtfile is required")
        sys.exit(1)

    if opt.srcdir is not None and opt.filelist is None:
        print("file ID list is required")
        sys.exit(1)

    src_lf0_stats_file = opt.srcstatsfile
    tgt_lf0_stats_file = opt.tgtstatsfile

    if os.path.isfile(src_lf0_stats_file):
        in_f = open(src_lf0_stats_file, 'r')
        data = in_f.readlines()
        in_f.close()

        [src_mean_f0, src_std_f0] = map(float, data[0].strip().split())
    else:
        print("File doesn't exist!! Please check path: %s" % (src_lf0_stats_file))

    if os.path.isfile(tgt_lf0_stats_file):
        in_f = open(tgt_lf0_stats_file, 'r')
        data = in_f.readlines()
        in_f.close()

        [tgt_mean_f0, tgt_std_f0] = map(float, data[0].strip().split())
    else:
        print("File doesn't exist!! Please check path: %s" % (tgt_lf0_stats_file))

    # print(src_mean_f0, src_std_f0)
    # print(tgt_mean_f0, tgt_std_f0)

    stats_dict = {}

    stats_dict['mu_src'] = src_mean_f0
    stats_dict['mu_tgt'] = tgt_mean_f0

    stats_dict['std_src'] = src_std_f0
    stats_dict['std_tgt'] = tgt_std_f0

    if opt.srcdir is not None and opt.tgtdir is not None:
        file_id_list = file_util.read_file_by_line(opt.filelist)
        src_lf0_file_list = file_util.prepare_file_path_list(file_id_list, opt.srcdir, '.lf0')
        tgt_lf0_file_list = file_util.prepare_file_path_list(file_id_list, opt.tgtdir, '.lf0')

        transform_lf0_dir(src_lf0_file_list, tgt_lf0_file_list, stats_dict)

    elif opt.srcfile is not None and opt.tgtfile is not None:

        transform_lf0_file(opt.srcfile, opt.tgtfile, stats_dict)

