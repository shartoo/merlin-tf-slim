import re

from util import log_util

logger = log_util.get_logger("HTK Question")


def load_question_set_continous(qs_file_name):
    fid = open(qs_file_name)
    binary_qs_index = 0
    continuous_qs_index = 0
    binary_dict = {}
    continuous_dict = {}
    LL = re.compile(re.escape('LL-'))

    for line in fid.readlines():
        line = line.replace('\n', '').replace('\t', ' ')
        if len(line) > 5:
            temp_list = line.split('{')
            temp_line = temp_list[1]
            temp_list = temp_line.split('}')
            temp_line = temp_list[0]
            temp_line = temp_line.strip()
            question_list = temp_line.split(',')

            temp_list = line.split(' ')
            question_key = temp_list[1]
            #                print   line
            if temp_list[0] == 'CQS':
                assert len(question_list) == 1
                processed_question = wildcards2regex(question_list[0], convert_number_pattern=True)
                continuous_dict[str(continuous_qs_index)] = re.compile(
                    processed_question)  # save pre-compiled regular expression
                continuous_qs_index = continuous_qs_index + 1
            elif temp_list[0] == 'QS':
                re_list = []
                for temp_question in question_list:
                    processed_question = wildcards2regex(temp_question)
                    if LL.search(question_key):
                        processed_question = '^' + processed_question
                    re_list.append(re.compile(processed_question))

                binary_dict[str(binary_qs_index)] = re_list
                binary_qs_index = binary_qs_index + 1
            else:
                logger.critical('The question set is not defined correctly: %s' % (line))
                raise Exception
                #                question_index = question_index + 1
    return binary_dict, continuous_dict


def load_question_set(qs_file_name):
    fid = open(qs_file_name)
    question_index = 0
    question_dict = {}
    ori_question_dict = {}
    for line in fid.readlines():
        line = line.replace('\n', '')
        if len(line) > 5:
            temp_list = line.split('{')
            temp_line = temp_list[1]
            temp_list = temp_line.split('}')
            temp_line = temp_list[0]
            question_list = temp_line.split(',')
            question_dict[str(question_index)] = question_list
            ori_question_dict[str(question_index)] = line
            question_index += 1
    fid.close()

    logger.debug('loaded question set with %d questions' % len(question_dict))

    return question_dict, ori_question_dict


def wildcards2regex(question, convert_number_pattern=False):
    """
    Convert HTK-style question into regular expression for searching labels.
    If convert_number_pattern, keep the following sequences unescaped for
    extracting continuous values):
        (\d+)       -- handles digit without decimal point
        ([\d\.]+)   -- handles digits with and without decimal point
    """
    ## handle HTK wildcards (and lack of them) at ends of label:
    prefix = ""
    postfix = ""
    if '*' in question:
        if not question.startswith('*'):
            prefix = "\A"
        if not question.endswith('*'):
            postfix = "\Z"
    question = question.strip('*')
    question = re.escape(question)
    ## convert remaining HTK wildcards * and ? to equivalent regex:
    question = question.replace('\\*', '.*')
    question = question.replace('\\?', '.')
    question = prefix + question + postfix

    if convert_number_pattern:
        question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
        question = question.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
    return question


qs = "I:/newwork/merlin-tf-slim/data/questions/questions-mandarin.hed"
question_dict, ori_question_dict = load_question_set(qs)

# for (k,v) in ori_question_dict.items():
#     print(k,v)

for (k, v) in question_dict.items():
    print(k, v)
