import os
import sys

ROOT_PATH = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]
TEST_SAMPLE_PATH = r"C:/Windows/System32/notepad.exe"
MAL_SAMPLE_PATH = r"D:/code/Mal/gym_malware/envs/utils/samples/DoS.Win32.ARPKiller.12"
LOG_PATH = os.path.join(ROOT_PATH, "logs")
SAMPLE_PATH = os.path.join(ROOT_PATH, "gym_malware", "envs", "utils", "samples")
MODEL_PATH = os.path.join(ROOT_PATH, "models")
TEST_SIZE = 1
