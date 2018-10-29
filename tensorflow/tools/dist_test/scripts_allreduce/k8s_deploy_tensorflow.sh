#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

function usage {
  script_name=$0
  echo "Usage:"
  echo "  $script_name [--image docker_image] [--num_containers num_of_containers]"
  echo "               [--deployment deployment_name] [--config_map config_map]"
  echo "               [--cp] [--src local_src_dir] [--dest container_dest_dir]"
  echo "               [--port container_ssh_port] [--hostnet] [--shared_volume]"
  echo "               [--delete] [--help]"
  echo ""
  echo "  Parameters:"
  echo "    image:          docker image used to create container."
  echo "    num_containers: number of containers that will be launched."
  echo "    deployment:     deployment name. (default: k8s-ml-deployment)"
  echo "    config_map:     config map name. (default: k8s-config-map)"
  echo "    cp:             upload file to all containers. (src and dest must"
  echo "                    be provided along with cp option)"
  echo "    src:            path to local source file. (used for cp option)"
  echo "    dest:           path to destination in container. (used for cp option)"
  echo "    port:           ssh port in container. Set ssh port (other than 22)"
  echo "                    when host network mode is enabled"
  echo "    hostnet:        enable host network mode. (default: disable)"
  echo "    shared_volume:  mount shared volume. (default: disable)"
  echo "    delete:         delete deployment and configmap."
  echo "                    (default: k8s-ml-deployment and k8s-config-map)"
  echo "    help:           print usage."
}

# Create temporary directory
TMP_DIR=$(mktemp -d)

# Temporary k8s yaml file
YAML_TMP_FILE="${TMP_DIR}/k8s_ml.yaml"

# Temporary hostfile
HOST_FILE="${TMP_DIR}/hostfile"

# Docker image and number of containers
DOCKER_IMAGE=""
NUM_CONTAINERS=0

# Default ssh port
SSH_PORT=22

# Default config map
CONFIG_MAP="k8s-config-map"

# Default Deployment
DEPLOYMENT="k8s-ml-deployment"

# Used for uploading file to all docker containers
CP=0
SRC=""
DEST=""

# Python script to generate yaml file for k8s TensorFlow cluster
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_GEN_ALLREDUCE_TF_YAML="${CUR_SCRIPT_DIR}/k8s_generate_yaml.py"

# Create or delete tensorflow cluster
# DELETE=0: Create cluster
# DELETE=1: Delete cluster
DELETE=0

# Used to enable host network mode to achieve best performance
# USE_HOSTNET=0: Flannel network mode
# USE_HOSTNET=1: Host network mode
USE_HOSTNET=0

# Used to mount shared volume
USE_SHARED_VOLUME=0

if [[ $# -lt 1 ]]; then
  echo "Error: illegal number of parameters"
  usage
  exit 1
fi

while [[ $# -ge 1 ]]; do
  key="$1"
  case $key in
    --image)
      DOCKER_IMAGE="$2"
      shift
      ;;
    --num_containers)
      NUM_CONTAINERS="$2"
      shift
      ;;
    --config_map)
      CONFIG_MAP="$2"
      shift
      ;;
    --deployment)
      DEPLOYMENT="$2"
      shift
      ;;
    --cp)
      CP=1
      ;;
    --src)
      SRC="$2"
      shift
      ;;
    --dest)
      DEST="$2"
      shift
      ;;
    --port)
      SSH_PORT="$2"
      shift
      ;;
    --hostnet)
      USE_HOSTNET=1
      ;;
    --shared_volume)
      USE_SHARED_VOLUME=1
      ;;
    --delete)
      DELETE=1
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      usage
      exit 1
      ;;
  esac
  shift
done

function generate_yaml_file {
  if [[ ! -f ${K8S_GEN_ALLREDUCE_TF_YAML} ]]; then
    echo "Error: can not find yaml-generating script ${K8S_GEN_ALLREDUCE_TF_YAML}"
    exit 1
  fi

  echo ""
  echo "Generating k8s cluster yaml config file with the following settings"
  echo "  Docker image: ${DOCKER_IMAGE}"
  echo "  Number of containers: ${NUM_CONTAINERS}"
  echo "  Config map: ${CONFIG_MAP}"
  echo "  Deployment: ${DEPLOYMENT}"

  if [[ $USE_HOSTNET -eq 1 ]]; then
    echo "  Host network mode: True"
    echo "  Container ssh port: ${SSH_PORT}"
  fi

  python ${K8S_GEN_ALLREDUCE_TF_YAML} \
    --docker_image ${DOCKER_IMAGE} \
    --num_containers ${NUM_CONTAINERS} \
    --config_map ${CONFIG_MAP} \
    --deployment ${DEPLOYMENT} \
    --ssh_port ${SSH_PORT} \
    --use_hostnet ${USE_HOSTNET} \
    --use_shared_volume ${USE_SHARED_VOLUME} \
    > ${YAML_TMP_FILE}
}

# Note: this function remove the yaml file to make sure that the key automatically
# generated inside the container is not reused in other deployment
function remove_yaml_file {
  rm -rf ${YAML_TMP_FILE}
}

function upload_file_to_all_containers {
  ${KUBECTL_BIN} get pods | grep ${DEPLOYMENT} \
    | awk '{print $1}' | \
    while read line;
    do
      echo "Uploading $1 to $line:$2"
      ${KUBECTL_BIN} cp $1 $line:$2
    done
}

function generate_container_hostfile {
  # This line assumes that --output=wide prints the IP addresses
  # in the 6th column
  ${KUBECTL_BIN} get pods --output=wide | grep ${DEPLOYMENT} \
      | awk '{print $6}' > ${HOST_FILE}

  echo ""
  echo "Containers hostfile locates at ${HOST_FILE}"
}

function launch_container {
  generate_yaml_file
  echo ""
  echo "Launching k8s cluster..."
  ${KUBECTL_BIN} create -f ${YAML_TMP_FILE}
  generate_container_hostfile
  remove_yaml_file
}

function delete_deployment_configmap {
  ${KUBECTL_BIN} delete deployment ${DEPLOYMENT}
  ${KUBECTL_BIN} delete configmap ${CONFIG_MAP}
}

# Check kubectl binary
KUBECTL_BIN=kubectl
if [[ ! -x "$(command -v ${KUBECTL_BIN})" ]]; then
  echo 'Error: cannot find kubectl binary'
  exit 1
fi

if [[ $DELETE -eq 1 ]]; then
  echo "Deleting deployment ${DEPLOYMENT} and config map ${CONFIG_MAP}..."
  delete_deployment_configmap
elif [[ $CP -eq 1 || -n "$SRC" || -n "$DEST" ]] ; then
  if [[ "$CP" -eq 1 && -n "$SRC" && -n "$DEST" ]]; then
    upload_file_to_all_containers $SRC $DEST
  else
    echo "Error: all cp, src and dest are required to upload file to container"
    exit 1
  fi
else
  if [[ -z "$DOCKER_IMAGE" ]]; then
    echo "Error: docker image is missing"
    exit 1
  fi

  if [[ "$NUM_CONTAINERS" -le 0 ]]; then
    echo "Error: illegal number of containers"
    exit 1
  fi

  if [[ $USE_HOSTNET -eq 1 && $SSH_PORT -eq 22 ]]; then
    echo "Error: please set container ssh port with --port (other than 22)" \
        "when host network mode is enabled"
    exit 1
  fi

  launch_container
fi
