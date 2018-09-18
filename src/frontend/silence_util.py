import math

import numpy as np

from util import file_util


## OSW: rewrote above more succintly
def check_silence_pattern(silence_pattern, label):
    for current_pattern in silence_pattern:
        current_pattern = current_pattern.strip('*')
        if current_pattern in label:
            return 1
    return 0


def trim_silence(in_list, out_list, in_dimension, label_list, label_dimension, silence_feature_index,
                 percent_to_keep=0):
    '''
    Function to trim silence from binary label/speech files based on binary labels.
        in_list: list of binary label/speech files to trim
        out_list: trimmed files
        in_dimension: dimension of data to trim
        label_list: list of binary labels which contain trimming criterion
        label_dimesion:
        silence_feature_index: index of feature in labels which is silence: 1 means silence (trim), 0 means leave.
    '''
    assert len(in_list) == len(out_list) == len(label_list)
    for (infile, outfile, label_file) in zip(in_list, out_list, label_list):

        data, _ = file_util.load_binary_file(infile, in_dimension)
        label, _ = file_util.load_binary_file(label_file, label_dimension)

        audio_label_difference = data.shape[0] - label.shape[0]
        assert math.fabs(audio_label_difference) < 3, '%s and %s contain different numbers of frames: %s %s' % (
            infile, label_file, data.shape[0], label.shape[0])

        ## In case they are different, resize -- keep label fixed as we assume this has
        ## already been processed. (This problem only arose with STRAIGHT features.)
        if audio_label_difference < 0:  ## label is longer -- pad audio to match by repeating last frame:
            print('audio too short -- pad')
            padding = np.vstack([data[-1, :]] * int(math.fabs(audio_label_difference)))
            data = np.vstack([data, padding])
        elif audio_label_difference > 0:  ## audio is longer -- cut it
            print('audio too long -- trim')
            new_length = label.shape[0]
            data = data[:new_length, :]
        # else: -- expected case -- lengths match, so do nothing
        silence_flag = label[:, silence_feature_index]
        #         print silence_flag
        if not (np.unique(silence_flag) == np.array([0, 1])).all():
            ## if it's all 0s or 1s, that's ok:
            assert (np.unique(silence_flag) == np.array([0]).all()) or \
                   (np.unique(silence_flag) == np.array([1]).all()), \
                'dimension %s of %s contains values other than 0 and 1' % (silence_feature_index, infile)
        print('Remove %d%% of frames (%s frames) as silence... ' % (
            100 * np.sum(silence_flag / float(len(silence_flag))), int(np.sum(silence_flag))))
        non_silence_indices = np.nonzero(
            silence_flag == 0)  ## get the indices where silence_flag == 0 is True (i.e. != 0)
        if percent_to_keep != 0:
            assert type(percent_to_keep) == int and percent_to_keep > 0
            # print silence_flag
            silence_indices = np.nonzero(silence_flag == 1)
            ## nonzero returns a tuple of arrays, one for each dimension of input array
            silence_indices = silence_indices[0]
            every_nth = 100 / percent_to_keep
            silence_indices_to_keep = silence_indices[::every_nth]  ## every_nth used +as step value in slice
            ## -1 due to weird error with STRAIGHT features at line 144:
            ## IndexError: index 445 is out of bounds for axis 0 with size 445
            if len(silence_indices_to_keep) == 0:
                silence_indices_to_keep = np.array([1])  ## avoid errors in case there is no silence
            print('   Restore %s%% (every %sth frame: %s frames) of silent frames' % (
                percent_to_keep, every_nth, len(silence_indices_to_keep)))

            ## Append to end of utt -- same function used for labels and audio
            ## means that violation of temporal order doesn't matter -- will be consistent.
            ## Later, frame shuffling will disperse silent frames evenly across minibatches:
            non_silence_indices = (np.hstack([non_silence_indices[0], silence_indices_to_keep]))
            ##  ^---- from tuple and back (see nonzero note above)

        trimmed_data = data[non_silence_indices, :]  ## advanced integer indexing
        file_util.array_to_binary_file(trimmed_data, outfile)
