#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

echo "Running on ${orange}nvidia${reset_color} hardware"
ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"

xhost +
docker run -it --rm \
    $ARGS \
    --ipc host \
    --privileged \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/${USER}:/home/${USER} \
    -p ${UID}0:22 \
    --name quant_dis \
    quant_dis:latest
xhost -

docker exec --user root \
    quant_dis bash -c "/etc/init.d/ssh start"