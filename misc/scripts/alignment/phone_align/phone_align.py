import os
import shutil

from util import file_util

MerlinDir = "merlin"
frontend = os.path.join(MerlinDir, "misc", "scripts", "frontend")
ESTDIR = os.path.join(MerlinDir, "tools", "speech_tools")
FESTDIR = os.path.join(MerlinDir, "tools", "festival")
FESTVOXDIR = os.path.join(MerlinDir, "tools", "festvox")
engdataset = "slt_arctic"
clustergen = "%s setup_cg cmu us %s" % (os.path.join(FESTVOXDIR, "src", "clustergen"), engdataset)
file_util.copy_filepath("../cmuarctic.data", " etc/txt.done.data")
file_util.copy_filepath("../cmuarctic.data", " etc/txt.done.data")
slt_wavs = file_util.read_file_list_from_path("../slt_wav/", file_type=".wav")
for wav in slt_wavs:
    shutil.copy(wav, os.path.join("wav", os.path.basename(wav)))
