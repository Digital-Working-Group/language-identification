#!/bin/bash

if [ $# -eq 1 ];
then
    container_name=$1
else
    container_name="lang-id-ctr"
fi

if [ $# -eq 2 ];
then
    docker_name=$1
else
    docker_name="lang-id"
fi
dir_up=$(realpath "../../")
docker run -v $dir_up:/scripts -it --rm --gpus all --name $container_name $docker_name bash