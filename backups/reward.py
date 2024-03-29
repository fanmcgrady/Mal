import glob
import os
import random
import subprocess
import sys

# from gym_malware.envs.utils.engine import Engine
# class ClamAV(Engine):

class ClamAV():
    def __init__(self):
        # 对抗检测引擎的路径
        self.clamav_root = "E:\\AppData\\ClamAV"

    # 启动检测守护进程
    def run_detect_daemon(self):
        # The necessary work before detection is to open a daemon, Clamd, for detection
        p = subprocess.Popen(self.clamav_root + '\\clamd', shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             encoding='gb2312')
        # 有输出才启动成功
        for i in iter(p.stdout.readline, 'b'):
            if i is not None:
                break

    def scan(self, path):
        result = subprocess.getoutput(self.clamav_root + '\\clamdscan ' + path)     # --no-summary
        print(result)
        return 0 if result.split()[-1] == "OK" else 1

# class KaperSky(Engine):
#     pass

def init_engines():
    ClamAV().run_detect_daemon()

def label_function(bytez):
    # TODO: 处理内存和硬盘样本的转换，可以是写，也可以看下文档
    # https://docs.clamav.net/
    fname = '{}.exe'.format(random.random())
    print(fname)
    with open(fname, "wb") as f:  # 写文件时指定模式为"b"
        f.write(bytez)
        reward = ClamAV()
    reward.scan(fname)

if __name__ == '__main__':
    sample = r"E:\AppData\Endermanch@Ana.exe"
    with open(sample, 'rb') as rb:
        bytez = rb.read()
    print(label_function(bytez))