import configparser

'''
a configuration file reader
'''
cf = configparser.ConfigParser()


def get_section_string(section, name):
    '''
        read variable value from cfg file whose type is string
    :param section:     section in configuration file
    :param name:        the key of value you want to read
    :return:            variable value
    '''
    return cf.get(section, name)


def get_section_int(section, name):
    '''
         read variable value from cfg file whose type is int
     :param section:     section in configuration file
     :param name:        the key of value you want to read
     :return:            variable value
     '''
    return cf.getint(section, name)


def get_section_bool(section, name):
    '''
         read variable value from cfg file whose type is boolean
     :param section:     section in configuration file
     :param name:        the key of value you want to read
     :return:            variable value
     '''
    return cf.getboolean(section, name)
