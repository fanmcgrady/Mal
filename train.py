# coding=UTF-8
# ! /usr/bin/python
import argparse
import linecache
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from chainer import optimizers
from chainerrl import experiments, explorers, replay_buffer, misc

from gym_malware import sha256_holdout
from gym_malware.envs.controls import action as manipulate
from gym_malware.envs.utils import pefeatures

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}

net_layers = [256, 64]

# 用于快速调用chainerrl的训练方法，参数如下：
# 1、命令行启动visdom
# ➜  ~ source activate new
# (new) ➜  ~ python -m visdom.server -p 8888
# 2、运行train
# python train.py
def main():
    pass
    # test
    # if not args.test:
    #     print("training...")
    #
    #     # 反复多次重新训练模型，避免手工操作
    #     for _ in range(args.rounds):
    #         args.outdir = experiments.prepare_output_dir(
    #             args, args.outdir, argv=sys.argv)
    #         print('Output files are saved in {}'.format(args.outdir))
    #
    #         env, agent = train_agent(args)
    #         # env, agent = train_keras_dqn_model(args)
    #
    #         with open(os.path.join(args.outdir, 'scores.txt'), 'a') as f:
    #             f.write(
    #                 "total_turn/episode->{}({}/{})\n".format(env.total_turn / env.episode, env.total_turn, env.episode))
    #             f.write("history:\n")
    #
    #             count = 0
    #             success_count = 0
    #             for k, v in env.history.items():
    #                 count += 1
    #                 if v['evaded']:
    #                     success_count += 1
    #                     f.write("{}:{}->{}\n".format(count, k, v['evaded_sha256']))
    #                 else:
    #                     f.write("{}:{}->\n".format(count, k))
    #
    #             f.write("success count:{}".format(success_count))
    #             f.write("{}".format(env.history))
    #
    #         # 标识成功失败
    #         dirs = os.listdir(args.outdir)
    #         second_line = linecache.getline(os.path.join(args.outdir, 'scores.txt'), 2)
    #         success_score = second_line.strip('\n').split('\t')[3]
    #
    #         # 训练提前结束，标识成功
    #         success_flag = False
    #         for file in dirs:
    #             if file.endswith('_finish') and not file.startswith(str(args.steps)):
    #                 success_flag = True
    #                 break
    #
    #         os.rename(args.outdir, '{}-{}{}'.format(args.outdir, success_score, '-success' if success_flag else ''))
    #
    #         # 重置outdir到models
    #         args.outdir = 'models'
    # else:
    #     print("testing...")
    #     model_fold = os.path.join(args.outdir, args.load)
    #     scores_file = os.path.join(model_fold, 'scores.txt')
    #
    #     # baseline: choose actions at random
    #     if args.test_random:
    #         random_action = lambda bytez: np.random.choice(list(manipulate.ACTION_TABLE.keys()))
    #         random_success, misclassified = evaluate(random_action)
    #         total = len(sha256_holdout) - len(misclassified)  # don't count misclassified towards success
    #
    #         with open(scores_file, 'a') as f:
    #             random_result = "random: {}({}/{})\n".format(len(random_success) / total, len(random_success), total)
    #             f.write(random_result)
    #             f.write("==========================\n")
    #
    #     total = len(sha256_holdout)
    #     fe = pefeatures.PEFeatureExtractor()
    #
    #     def agent_policy(agent):
    #         def f(bytez):
    #             # first, get features from bytez
    #             feats = fe.extract2(bytez)
    #             action_index = agent.act(feats)
    #             return ACTION_LOOKUP[action_index]
    #
    #         return f
    #
    #     # ddqn
    #     env = gym.make('malware-test-v0')
    #     agent = create_ddqn_agent(env, args)
    #     mm = get_latest_model_dir_from(model_fold)
    #     agent.load(mm)
    #     success, _ = evaluate(agent_policy(agent))
    #     blackbox_result = "black: {}({}/{})".format(len(success) / total, len(success), total)
    #     with open(scores_file, 'a') as f:
    #         f.write("{}->{}\n".format(mm, blackbox_result))

if __name__ == '__main__':
    main()
