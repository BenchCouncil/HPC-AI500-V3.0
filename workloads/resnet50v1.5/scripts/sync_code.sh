#!/bin/bash

filename=$1
path="/workspace/nvidia-examples/resnet50v1.5/model"
port=22
echo $filename 

declare -a hosts=("172.168.0.1" "172.168.0.3" "172.168.0.13" "172.168.0.15" "172.168.0.7" "172.168.0.9" "172.168.0.11" "172.168.0.27")


for host in ${hosts[*]}
do
	echo $host;
	scp -P "${port}" "$path$filename" "$host:$path"
done


