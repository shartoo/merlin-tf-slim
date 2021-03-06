# -*- coding:utf-8 -*-

import logging
import os


def get_logger(log_name, log_save_file=".system.log"):
    if not (os.path.exists(log_save_file)):
        fo = open(log_save_file, "w")
        fo.close()
    fsize = os.path.getsize(log_save_file)
    fsize = fsize / float(1024 * 1024)
    if fsize > 40:
        os.remove(log_save_file)
        print("log file is remove log file..", log_save_file)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    # file log handler
    fh = logging.FileHandler(log_save_file)
    fh.setLevel(logging.DEBUG)
    # console hander
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # define handler output format
    formatter = logging.Formatter('%(asctime)s %(filename)s\t[line:%(lineno)d] %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
