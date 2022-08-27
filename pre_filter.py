import os
from config import *
from gym_malware.envs.utils import interface

from gym_malware.envs.utils.reward import ClamAV

ca = ClamAV()
benign = 0
samples = interface.get_samples()
for sample in samples:
    sample_path = os.path.join(interface.SAMPLE_PATH, sample)
    if ca.scan(sample_path) == 0:
        benign += 1
        os.remove(sample_path)
        with open(os.path.join(LOG_PATH, "preprocess_log.txt"), "a+") as f:
            f.write("{}: delete {}\n".format(benign, sample_path))
print(benign)