import os
from config import *
from gym_malware.envs.utils import interface

from gym_malware.envs.utils.reward import ClamAV

ca = ClamAV()
benign = 0
samples = interface.get_samples()
for sample in samples:
    if ca.scan(sample) == 0:
        benign += 1
        os.remove(sample)
        with open(os.path.join(LOG_PATH, "preprocess_log.txt"), "a+") as f:
            f.write("{}: delete {}\n".format(benign, sample))
print(benign)