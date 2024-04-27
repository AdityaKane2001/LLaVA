#! /usr/bin/bash

for i in $(seq 1 301)
do 
	echo "Downloading $i"
	wget -O ./MMVP\ Images/$i.jpg "https://huggingface.co/datasets/MMVP/MMVP/resolve/main/MMVP%20Images/$i.jpg?download=true"
done
