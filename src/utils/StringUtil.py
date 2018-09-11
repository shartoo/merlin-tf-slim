import os

'''
some operation for file
'''


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return file_id_list


def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return out_file_lists
