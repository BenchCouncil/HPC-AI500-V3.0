#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
A python script for data resampling under the same folder.
usage: python data_sampling.py [-input input data dir] [-nums number of data copies] [-output the dir of output data]
output: generate an equal number of folders as 'nums' that contain images after sampled.
'''
import argparse
from os import listdir
from os.path import isfile, join 
from pathlib import Path
import shutil
import numpy as np
import random
from scipy.stats import poisson

def data_sampling(data_dir):

    image_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    nums_images = len(image_files)
    random.shuffle(image_files)

    image_dict = {}
    keys = []
    for index, file_name in enumerate(image_files):
        image_dict[index] = file_name
        keys.append(index)

    #possion distribution is default
    keys_sampling = np.random.choice(keys, size = nums_images, replace=True, p = possion_distribution(keys, 1, nums_images))
    #uniform distribution or other distribution can also be used
    #keys_sampling = np.random.choice(keys, size = nums_images, replace=True)
    
    keys_sampling_dict = {}
    for index, counter in enumerate(keys_sampling):
        file_name = image_dict[index]
        keys_sampling_dict[file_name] = counter
    

    return keys_sampling_dict 
    

def possion_distribution(X, lmbda, size):
    
    poisson_pd = poisson.pmf(X, lmbda)
    return poisson_pd 

def generate_data(file_name_counter, input_dir, output_dir, index):
    output_path = output_dir + 'sampled_data'+ '_' + str(index) 
    #create output directory
    Path(output_path).mkdir(exist_ok=True)
    for key in file_name_counter:
        file_name = key
        counter = file_name_counter[key]
        input_path = input_dir + "/" + key
        while counter > 0:
            prefix = output_path + "/"
            suffix = str(counter) + "_" + key
            output_path_sampled = prefix + suffix
            shutil.copy2(input_path, output_path_sampled)
            counter = counter - 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True, help = 'define input data dir.') 
    parser.add_argument('-nums', type=int, required=True, help = 'define number of data copies after sampling.')
    parser.add_argument('-output', required=True, help = 'define output dir.')
    
    args = parser.parse_args()
    
    input_dir = args.input
    nums = args.nums
    output_dir = args.output


    for i in range(nums):
        file_name_counter = data_sampling(input_dir)
        generate_data(file_name_counter, input_dir, output_dir, i)










