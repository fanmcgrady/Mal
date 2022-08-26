import os
import sys

ROOT_PATH = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]
TEST_SAMPLE_PATH = r"C:/Windows/System32/notepad.exe"
MAL_SAMPLE_PATH = r"D:/code/Mal/gym_malware/envs/utils/samples/DoS.Win32.ARPKiller.13"
LOG_PATH = ROOT_PATH + r"/logs"
SAMPLE_PATH = ROOT_PATH + r"/gym_malware/envs/utils/samples"
TEST_SIZE = 1
