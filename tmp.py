import os


def create_io_file_for_htk(source, target, io_file="iofile.txt"):
    '''

    :param source:
    :param target:
    :return:
    '''
    lines = []
    sigs = [os.path.join(source, x) for x in os.listdir(source)]
    for sig in sigs:
        lines.append(sig + " " + target + "/" + os.path.basename(sig) + ".mfcc")
    with open(io_file, "w") as io:
        for line in lines:
            io.write(line + "\r")


source = r"I:\soft\HTK\samples\htk\samples\HTKDemo\data\yesno\sig"
target = r"I:\soft\HTK\samples\htk\samples\HTKDemo\data\yesno\mfcc"
create_io_file_for_htk(source, target)
