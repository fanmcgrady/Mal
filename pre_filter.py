import os
from config import *

from gym_malware.envs.utils.reward import ClamAV

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]
sample_dir_path = os.path.join(module_path, "gym_malware", "envs", "utils", "samples")
ca = ClamAV()
benign = 0
for sample in os.listdir(sample_dir_path):
    sample_path = os.path.join(sample_dir_path, sample)
    if ca.scan(sample_path) == 0:
        benign += 1
        os.remove(sample_path)
        with open(os.path.join(LOG_PATH, "preprocess_log.txt"), "a+") as f:
            f.write("{}: delete {}\n".format(benign, sample_path))
print(benign)