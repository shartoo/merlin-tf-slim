################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################


import re
import sys
from multiprocessing.dummy import Pool as ThreadPool

from src.frontend import silence_util
from util import file_util


class SilenceRemover(object):
    def __init__(self, n_cmp, silence_pattern=['*-#+*'], label_type="state_align", remove_frame_features=True,
                 subphone_feats="none"):
        self.silence_pattern = silence_pattern
        self.silence_pattern_size = len(silence_pattern)
        self.label_type = label_type
        self.remove_frame_features = remove_frame_features
        self.subphone_feats = subphone_feats
        self.n_cmp = n_cmp

    def remove_silence(self, in_data_list, in_align_list, out_data_list, dur_file_list=None):
        file_number = len(in_data_list)
        align_file_number = len(in_align_list)

        if file_number != align_file_number:
            print("The number of input and output files does not equal!\n")
            sys.exit(1)
        if file_number != len(out_data_list):
            print("The number of input and output files does not equal!\n")
            sys.exit(1)
        def _remove_silence(i):
            if self.label_type == "phone_align":
                if dur_file_list:
                    dur_file_name = dur_file_list[i]
                else:
                    dur_file_name = None
                nonsilence_indices = self.load_phone_alignment(in_align_list[i], dur_file_name)
            else:
                nonsilence_indices = self.load_alignment(in_align_list[i])
            ori_cmp_data, _ = file_util.load_binary_file(in_data_list[i], self.n_cmp)

            frame_number = ori_cmp_data.size / self.n_cmp

            if len(nonsilence_indices) == frame_number:
                print('WARNING: no silence found!')
                # previsouly: continue -- in fact we should keep non-silent data!

            ## if labels have a few extra frames than audio, this can break the indexing, remove them:
            nonsilence_indices = [ix for ix in nonsilence_indices if ix < frame_number]

            new_cmp_data = ori_cmp_data[nonsilence_indices,]

            file_util.array_to_binary_file(new_cmp_data, out_data_list[i])

        pool = ThreadPool()
        pool.map(_remove_silence, range(file_number))
        pool.close()
        pool.join()

    def load_phone_alignment(self, alignment_file_name, dur_file_name=None):

        if dur_file_name:
            dur_dim = 1  ## hard coded for now
            manual_dur_data, _ = file_util.load_binary_file(dur_file_name, dur_dim)

        ph_count = 0
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                # to do - support different frame shift - currently hardwired to 5msec
                # currently under beta testing: supports different frame shift
                if dur_file_name:
                    frame_number = manual_dur_data[ph_count]
                    ph_count = ph_count + 1
                else:
                    frame_number = int((end_time - start_time) / 50000)

            label_binary_flag = silence_util.check_silence_pattern(self.silence_pattern, full_label)

            if self.remove_frame_features:
                if label_binary_flag == 0:
                    for frame_index in range(frame_number):
                        nonsilence_frame_index_list.append(base_frame_index + frame_index)
                base_frame_index = base_frame_index + frame_number
            elif self.subphone_feats == 'none':
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1

        fid.close()

        return nonsilence_frame_index_list

    def load_alignment(self, alignment_file_name, dur_file_name=None):

        state_number = 5
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            if len(temp_list) == 1:
                state_index = state_number
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]
                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]
                state_index = int(state_index) - 1
                frame_number = int((end_time - start_time) / 50000)

            label_binary_flag = silence_util.check_silence_pattern(self.silence_pattern, full_label)

            if self.remove_frame_features:
                if label_binary_flag == 0:
                    for frame_index in range(frame_number):
                        nonsilence_frame_index_list.append(base_frame_index + frame_index)
                base_frame_index = base_frame_index + frame_number
            elif self.subphone_feats == 'state_only':
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1
            elif self.subphone_feats == 'none' and state_index == state_number:
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1

        fid.close()

        return nonsilence_frame_index_list


if __name__ == '__main__':
    cmp_file_list_name = ''
    lab_file_list_name = ''
    align_file_list_name = ''

    n_cmp = 229
    n_lab = 898

    in_cmp_list = ['/group/project/dnn_tts/data/nick/nn_cmp/nick/herald_001.cmp']
    in_lab_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.lab']
    in_align_list = ['/group/project/dnn_tts/data/cassia/nick_lab/herald_001.lab']

    out_cmp_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.tmp.cmp']
    out_lab_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.tmp.no.lab']

    remover = SilenceRemover(in_cmp_list, in_align_list, n_cmp, out_cmp_list)
    remover.remove_silence()
