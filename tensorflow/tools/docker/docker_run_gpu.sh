#!/bin/bash

set -e

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ ! -d ${CUDA_HOME}/lib64 ]; then
  echo "Failed to locate CUDA libs at ${CUDA_HOME}/lib64."
  exit 1
fi

export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda* | \
                    xargs -I{} echo '-v {}:{}')
export DEVICES=$(\ls /dev/nvidia* | \
                    xargs -I{} echo '--device {}:{}')
export CUDA_SRCS="-v ${CUDA_HOME}:${CUDA_HOME} -v /usr/share/nvidia:/usr/share/nvidia"

if [[ "${DEVICES}" = "" ]]; then
  echo "Failed to locate NVidia device(s). Did you want the non-GPU container?"
  exit 1
fi

docker run -it $CUDA_SO $CUDA_SRCS $DEVICES b.gcr.io/tensorflow/tensorflow-full-gpu "$@"
