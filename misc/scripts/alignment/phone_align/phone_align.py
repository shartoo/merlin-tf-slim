import os
import shutil

from util import file_util
from util.frontend.normalize_lab_for_merlin import normalize_label_files

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

os.system("./bin/do_build build_prompts")
os.system("./bin/do_build label")
os.system("./bin/do_build build_utts")

'''
cd ../
cat cmuarctic.data | cut -d " " -f 2 > file_id_list.scp
'''
scp_file_id_list = ""
make_label_cmd = "%s full-context-labels %s %s %s" % \
                 (os.path.join(frontend, "festival_utt_to_lab", "make_labels"), "cmu_us_slt_arctic/festival/utts",
                  os.path.join(FESTDIR, "examples", "dumpfeats"), os.path.join(frontend, "festival_utt_to_lab"))
os.system(make_label_cmd)

in_lab_dir = "full-context-labels/full"
out_lab_dir = "label_phone_align"
label_style = "phone_align "
file_id_list = file_util.read_file_by_line(scp_file_id_list)
write_time_stamps = True

for id in file_id_list:
    filename = id.strip() + '.lab'
    print(filename)
    in_lab_file = os.path.join(in_lab_dir, filename)
    out_lab_file = os.path.join(out_lab_dir, filename)
    normalize_label_files(in_lab_file, out_lab_file, label_style, write_time_stamps)
