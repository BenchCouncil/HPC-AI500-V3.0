#!/usr/bin/python

import os
import json
import pandas as pd
from functools import reduce


def get_throughput(output_dir):
    obj = json.load(open(os.path.join(output_dir, "models/training.json")))
    # print(obj.keys())
    # dict_keys(['run', 'epoch', 'iter', 'event'])
    # print(obj["iter"])
    # print(obj["iter"].keys())
    for k in obj["iter"].keys():
        obj["iter"][k] = reduce(lambda x, y: x+y, obj["iter"][k])
    df_iter = pd.DataFrame(obj["iter"])
    print("%50s\t%9.2f\t%7.2f" % (output_dir, df_iter.imgs_per_sec.mean(), df_iter.imgs_per_sec.std()))


all_output_dirs = [
        "fp16-90-SEED10-2-8", "fp16-90-SEED10-4-8", "fp16-90-SEED10-6-8", "fp16-90-SEED10-8-8",
        "fp32-90-SEED10-2-8", "fp32-90-SEED10-4-8", "fp32-90-SEED10-6-8", "fp32-90-SEED10-8-8",
        "fp32-90-SEED10-1-8-nowarmup", "fp32-90-SEED10-8-8-nowarmup",
        "fp32-90-SEED10-1-8-xla", "fp32-90-SEED10-2-8-xla", "fp32-90-SEED10-4-8-xla", "fp32-90-SEED10-8-8-xla",
        "fp32-90-SEED10-8-8-dali",
        "fp32-90-1"
    ]

print("%50s\t%13s\t%13s" % ("output_dir", "throuput mean", "through std"))
for output_dir in all_output_dirs:
    output_dir_path = os.path.join("/data/imagenet-output", output_dir)
    get_throughput(output_dir_path)

