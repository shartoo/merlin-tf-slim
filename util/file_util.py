import os
import shutil

from util import log_util

log = log_util.get_logger("file process")


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


if __name__ == "__main__":
    list = read_file_list_from_path("D:/公司资料", if_recursive=True)
    print(list)
