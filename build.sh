#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

echo "Building for ${orange}nvidia${reset_color} hardware"
DOCKERFILE=Dockerfile

docker build . \
    -f $DOCKERFILE \
    --build-arg UID=${UID} \
    --build-arg GID=${UID} \
    -t vsa_encoder:latest