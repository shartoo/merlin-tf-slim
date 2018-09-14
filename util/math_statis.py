import numpy as np

from util import log_util, file_util

log = log_util.get_logger("statistic function")

'''
### for htk
io_funcs = HTKFeat_read()
io_funcs.getall(in_file_list[i])



htk_writer = HTKFeat_write(veclen=io_funcs.veclen, sampPeriod=io_funcs.sampPeriod, paramKind=9)
htk_writer.writeall(norm_features, out_file_list[i])

## for normal 
io_funcs = BinaryIOCollection()
io_funcs.load_binary_file_frame(in_file_list[i], self.feature_dimension)

io_funcs.array_to_binary_file(norm_features, out_file_list[i])


'''


class Statis(object):
    def __init__(self, feature_dimension, read_func=None, writer_func=None, min_value=0.01, max_value=0.99,
                 min_vector=0.0, max_vector=0.0,
                 exclude_columns=[]):
        self.target_min_value = min_value
        self.target_max_value = max_value
        self.feature_dimension = feature_dimension
        self.min_vector = min_vector
        self.max_vector = max_vector
        self.exclude_columns = exclude_columns
        self.read_func = read_func
        self.writer_func = writer_func

        if type(min_vector) != float:
            try:
                assert (len(self.min_vector) == self.feature_dimension)
            except AssertionError:
                log.critical('inconsistent feature_dimension (%d) and length of min_vector (%d)' % (
                    self.feature_dimension, len(self.min_vector)))
                raise
        if type(max_vector) != float:
            try:
                assert (len(self.max_vector) == self.feature_dimension)
            except AssertionError:
                log.critical('inconsistent feature_dimension (%d) and length of max_vector (%d)' % (
                    self.feature_dimension, len(self.max_vector)))
                raise

        log.debug('MinMaxNormalisation created for feature dimension of %d' % self.feature_dimension)

    def feature_normalisation(self, in_file_list, out_file_list):
        logger = log.getLogger('feature_normalisation')
        try:
            assert len(in_file_list) == len(out_file_list)
        except  AssertionError:
            logger.critical('The input and output file numbers are not the same! %d vs %d' % (
                len(in_file_list), len(out_file_list)))
            raise
        if self.mean_vector == None:
            self.mean_vector = self.compute_mean(in_file_list, 0, self.feature_dimension)
        if self.std_vector == None:
            self.std_vector = self.compute_std(in_file_list, self.mean_vector, 0, self.feature_dimension)
        file_number = len(in_file_list)
        for i in range(file_number):
            features, current_frame_number = self.read_func(in_file_list[i], self.feature_dimension)
            mean_matrix = np.tile(self.mean_vector, (current_frame_number, 1))
            std_matrix = np.tile(self.std_vector, (current_frame_number, 1))
            norm_features = (features - mean_matrix) / std_matrix
            if self.writer_func is not None:
                self.writer_func((norm_features, out_file_list[i]))
        return self.mean_vector, self.std_vector

    def feature_denormalisation(self, in_file_list, out_file_list, mean_vector, std_vector):

        file_number = len(in_file_list)
        try:
            assert len(in_file_list) == len(out_file_list)
        except  AssertionError:
            log.critical('The input and output file numbers are not the same! %d vs %d' % (
                len(in_file_list), len(out_file_list)))
            raise
        try:
            assert mean_vector.size == self.feature_dimension and std_vector.size == self.feature_dimension
        except AssertionError:
            log.critical(
                'the dimensionalities of the mean and standard derivation vectors are not the same as the dimensionality of the feature')
            raise
        for i in range(file_number):
            features, current_frame_number = self.read_func(in_file_list[i], self.feature_dimension)
            mean_matrix = np.tile(mean_vector, (current_frame_number, 1))
            std_matrix = np.tile(std_vector, (current_frame_number, 1))
            norm_features = features * std_matrix + mean_matrix
            if self.writer_func is not None:
                self.writer_func((norm_features, out_file_list[i]))

    def compute_mean(self, file_list, start_index, end_index):
        local_feature_dimension = end_index - start_index
        mean_vector = np.zeros((1, local_feature_dimension))
        all_frame_number = 0
        for file_name in file_list:
            features, current_frame_number = self.read_func(file_name, self.feature_dimension)
            mean_vector += np.reshape(np.sum(features[:, start_index:end_index], axis=0),
                                      (1, local_feature_dimension))
            # current_frame_number = features.size // self.feature_dimension
            all_frame_number += current_frame_number
        mean_vector /= float(all_frame_number)
        # setting the print options in this way seems to break subsequent printing of numpy float32 types
        # no idea what is going on - removed until this can be solved
        log.info('computed mean vector of length %d :' % mean_vector.shape[1])
        log.info(' mean: %s' % mean_vector)
        self.mean_vector = mean_vector
        return mean_vector

    def compute_std(self, file_list, mean_vector, start_index=None, end_index=None):
        logger = log.getLogger('feature_normalisation')
        if start_index == None and end_index == None:
            local_feature_dimension = self.feature_dimension
        else:
            local_feature_dimension = end_index - start_index
        std_vector = np.zeros((1, self.feature_dimension))
        all_frame_number = 0
        for file_name in file_list:
            features, current_frame_number = self.read_func(file_name, self.feature_dimension)
            mean_matrix = np.tile(mean_vector, (current_frame_number, 1))
            if start_index == None and end_index == None:
                std_vector += np.reshape(np.sum((features - mean_matrix) ** 2, axis=0),
                                         (1, self.feature_dimension))
            else:
                std_vector += np.reshape(np.sum((features[:, start_index:end_index] - mean_matrix) ** 2, axis=0),
                                         (1, local_feature_dimension))
            all_frame_number += current_frame_number
        std_vector /= float(all_frame_number)
        std_vector = std_vector ** 0.5
        # setting the print options in this way seems to break subsequent printing of numpy float32 types
        # no idea what is going on - removed until this can be solved
        logger.info('computed  std vector of length %d' % std_vector.shape[1])
        logger.info('  std: %s' % std_vector)
        self.std_vector = std_vector
        return std_vector

    def normal_standardization(self, in_file_list, out_file_list, feature_dimension):

        #        self.dimension_dict = dimension_dict
        self.feature_dimension = feature_dimension
        mean_vector = self.compute_mean(in_file_list, 0, feature_dimension)
        std_vector = self.compute_std(in_file_list, mean_vector, 0, feature_dimension)
        file_number = len(in_file_list)

        for i in range(file_number):
            features, current_frame_number = self.read_func(in_file_list[i], self.feature_dimension)
            mean_matrix = np.tile(mean_vector, (current_frame_number, 1))
            std_matrix = np.tile(std_vector, (current_frame_number, 1))
            norm_features = (features - mean_matrix) / std_matrix
            if self.writer_func is not None:
                self.writer_func((norm_features, out_file_list[i]))
        return mean_vector, std_vector

    def find_min_max_values(self, in_file_list, start_index, end_index):
        local_feature_dimension = end_index - start_index
        file_number = len(in_file_list)
        min_value_matrix = np.zeros((file_number, local_feature_dimension))
        max_value_matrix = np.zeros((file_number, local_feature_dimension))
        for i in range(file_number):
            features, _ = file_util.load_binary_file(in_file_list[i], self.feature_dimension)
            temp_min = np.amin(features[:, start_index:end_index], axis=0)
            temp_max = np.amax(features[:, start_index:end_index], axis=0)
            min_value_matrix[i,] = temp_min;
            max_value_matrix[i,] = temp_max;
        self.min_vector = np.amin(min_value_matrix, axis=0)
        self.max_vector = np.amax(max_value_matrix, axis=0)
        self.min_vector = np.reshape(self.min_vector, (1, local_feature_dimension))
        self.max_vector = np.reshape(self.max_vector, (1, local_feature_dimension))
        self.logger.info('found min/max values of length %d:' % local_feature_dimension)
        self.logger.info('  min: %s' % self.min_vector)
        self.logger.info('  max: %s' % self.max_vector)

    def normalise_data(self, in_file_list, out_file_list):
        file_number = len(in_file_list)

        fea_max_min_diff = self.max_vector - self.min_vector
        diff_value = self.target_max_value - self.target_min_value
        fea_max_min_diff = np.reshape(fea_max_min_diff, (1, self.feature_dimension))

        target_max_min_diff = np.zeros((1, self.feature_dimension))
        target_max_min_diff.fill(diff_value)

        target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        for i in range(file_number):
            features, _ = self.read_func(in_file_list[i], self.feature_dimension)
            frame_number = features.size // self.feature_dimension
            fea_min_matrix = np.tile(self.min_vector, (frame_number, 1))
            target_min_matrix = np.tile(self.target_min_value, (frame_number, self.feature_dimension))
            fea_diff_matrix = np.tile(fea_max_min_diff, (frame_number, 1))
            diff_norm_matrix = np.tile(target_max_min_diff, (frame_number, 1)) / fea_diff_matrix
            norm_features = diff_norm_matrix * (features - fea_min_matrix) + target_min_matrix
            ## If we are to keep some columns unnormalised, use advanced indexing to
            ## reinstate original values:
            m, n = np.shape(features)
            for col in self.exclude_columns:
                norm_features[list(range(m)), [col] * m] = features[list(range(m)), [col] * m]
            if self.writer_func is not None:
                self.writer_func(norm_features, out_file_list[i])
