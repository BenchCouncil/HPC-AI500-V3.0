#!/usr/bin/python
import os
import json
import pandas as pd


def get_all_acc(output_dir_path, csv_name):
    model_dirs = filter(lambda x: "eval-model.ckpt-" in x, os.listdir(output_dir_path))
    model_steps = sorted(map(lambda x: int(x[len("eval-model.ckpt-"):]), model_dirs))
    model_dirs = list(map(lambda x: os.path.join(output_dir_path, "eval-model.ckpt-%d" % x), model_steps))
    model_acc_list = []
    for model_step, model_dir in zip(model_steps, model_dirs):
        obj = json.load(open(os.path.join(model_dir, "acc.json")))
        obj["step"] = model_step
        model_acc_list.append(obj)
    df = pd.DataFrame(model_acc_list)
    csv_path = os.path.join(output_dir_path, csv_name)
    df.to_csv(csv_path, index=False)
    # print("model acc was already saved in %s" % csv_path)
    print(csv_path, end=" ")


all_output_dirs = [
        "fp16-90-SEED10-2-8", "fp16-90-SEED10-4-8", "fp16-90-SEED10-6-8", "fp16-90-SEED10-8-8",
        "fp32-90-SEED10-2-8", "fp32-90-SEED10-4-8", "fp32-90-SEED10-6-8", "fp32-90-SEED10-8-8",
       "fp32-90-SEED10-1-8-nowarmup", "fp32-90-SEED10-8-8-nowarmup",
       "fp32-90-SEED10-1-8-xla", "fp32-90-SEED10-2-8-xla", "fp32-90-SEED10-4-8-xla", "fp32-90-SEED10-8-8-xla",
       "fp32-90-SEED10-8-8-dali", "fp32-90-SEED10-8-8-xla-dali"
   ]
for output_dir in all_output_dirs:
    output_dir_path = os.path.join("/data/imagenet-output", output_dir)
    get_all_acc(output_dir_path, "acc.csv")
print("")
for output_dir in all_output_dirs:
    output_dir_path = os.path.join("/data/imagenet-output", output_dir)
    df = pd.read_csv(os.path.join(output_dir_path, "acc.csv"))
    print("%40s: %10.2f, %10.2f" % (output_dir, df.iloc[-1].top1_accuracy, df.iloc[-1].top5_accuracy))


