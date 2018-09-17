import os
import shutil

import numpy as np

from util import log_util

log = log_util.get_logger("file process")


def create_blank_file(file_name):
    '''
        create a blank file
    :param file_name:
    :return:
    '''
    with open(file_name, 'w') as wt:
        wt.write("")
    log.debug("blank file %s created.." % file_name)

def read_file_list_from_path(path, file_type=None, if_recursive=False):
    '''
        get all file list from path
    :param path:
    :param file_type:
    :return:
    '''
    file_list = []
    for file in os.listdir(path):
        tmp_file = os.path.join(path, file)
        if if_recursive:
            if os.path.isfile(tmp_file):
                if file_type:
                    if str(tmp_file).endswith(file_type):
                        file_list.append(tmp_file)
                else:
                    file_list.append(tmp_file)
            elif os.path.isdir(tmp_file):
                file_list += read_file_list_from_path(tmp_file, file_type, if_recursive)
        else:
            if file_type is not None:
                if file.endswith(file_type):
                    file_list.append(tmp_file)
            else:
                file_list.append(tmp_file)
    file_list.sort()
    return file_list


def read_file_by_line(filename):
    '''
        read every line of a file
    :param filename: file to read
    :return:        list of line content in file
    '''
    line_list = []
    with open(filename, 'r') as rd:
        lines = rd.readlines()
        for line in lines:
            line = line.strip()
            if len(line) < 1:
                continue
            line_list.append(line)
    return line_list

def write2file(content, save_file):
    '''
            write content to file
    :param content:     content to write to file
    :param save_file:   where should content be written to
    :return:
    '''
    with open(save_file, "w") as wt:
        if isinstance(content, list):
            for con in content:
                wt.write(con + "\r")
        else:
            wt.write(content)
    log.debug(" write content to %s " % save_file)


def copy_filepath(src_path, target_path):
    '''
        copy file from source to target
    :param src_path:
    :param target_path:
    :return:
    '''

    shutil.copytree(src_path, target_path)
    log.debug("copy directory from %s  to %s finished.." % (src_path, target_path))


def del_path_list(path_list):
    '''
    delete file paths or file in list
    :param path_list:   file path (or file)list to be deleted
    :return:
    '''
    for path in path_list:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                os.remove(path)
            log.debug(" file path %s was deleted" % path)


def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)
    return file_name_list

def create_path_list(path_list):
    '''
    create file path in list
    :param path_list:   file path to be created in list
    :return:
    '''
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)
            log.debug(" file path %s created " % path)


def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64')  # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return m_data


def write_binfile(m_data, filename, dtype=np.float64):
    '''
    Writes numpy array into binary file.
    '''
    m_data = np.array(m_data, dtype)
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return

def array_to_binary_file(self, data, output_file_name):
    data = np.array(data, 'float32')
    fid = open(output_file_name, 'wb')
    data.tofile(fid)
    fid.close()

def load_binary_file_frame(self, file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0, 'specified dimension not compatible with data'
    frame_number = features.size / dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return features, frame_number

if __name__ == "__main__":
    list = read_file_list_from_path("D:/test", if_recursive=True)
    print(list)
