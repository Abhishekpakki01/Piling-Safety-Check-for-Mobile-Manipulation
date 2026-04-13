#!/bin/bash
xhost +local:docker
gnome-terminal -- sudo docker run -it \
    --network host \
    --gpus device=0 \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/$(whoami)/Abhishek:/noodle_project \
    noodle_sim_ready:latest

