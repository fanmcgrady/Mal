import subprocess

import pyclamd
from config import *
import gym_malware.envs.utils.interface as interface
import gym_malware.envs.utils.pefeatures as pefeatures

class ClamAV():
    def __init__(self):
        self.cd = pyclamd.ClamdAgnostic()
        print(self.cd.version())
    def scan(self, input):
        if type(input) == bytes:
            result = self.cd.scan_stream(input)
        else:
            result = self.cd.scan_file(input)
        print(result)
        return 0 if result == None else 1

def label_function(bytez):
    # TODO: 处理内存和硬盘样本的转换，可以是写，也可以看下文档
    # https://docs.clamav.net/
    sample = None

    reward = ClamAV()
    reward.scan(sample)

if __name__ == '__main__':
    sample_path = MAL_SAMPLE_PATH
    with open(sample_path, 'rb') as sp:
        sample = sp.read()
    # print('sample:',sample)
    # st = pefeatures.SectionInfo()
    # sample_parse = lief.parse(sample)
    # print(st(sample_parse))
    reward = ClamAV()
    # print(reward.scan(sample))
