#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Build TensorFlow Docker images for remote build
#
# Usage:
#   ci_rbe_docker_build.sh -c # docker image for cpu build
#   ci_rbe_docker_build.sh -g # docker image for gpu build

function main {
  cpu_build=false
  gpu_build=false
  publish=false

  script_dir=$(dirname "$(readlink -f "$0")")
  cd $script_dir

  set_script_flags $@

  build_tf_image

  if [ "$publish" = true ] ; then
    publish_tf_image
  fi
}


function set_script_flags {
  OPTIND=1 # Reset for getopts, just in case.
  while getopts "cf:ghn" opt; do
    case "$opt" in
      c)
        cpu_build=true
        ;;
      g)
        gpu_build=true
        ;;
      h)
        print_usage
        ;;
      p)
        publish=true
        ;;
      *)
        print_usage "ERROR: unknown option"
        ;;
    esac
  done
  [[ "$cpu_build" = true ]] || [[ "$gpu_build" = true ]] || print_usage "ERROR: must specify build at least for one build type: cpu or gpu"

}


function print_usage {
  echo "Usage: $(basename $0) -c | -g [options]"
  echo "  -c build image for CPU build (base image debian8-clang)"
  echo "  -g build image for GPU build (base image nvidia-clang)"
  echo "[option] is one of"
  echo "  -n not publish the locally-built image to GCR;"
  echo "     the build process will publish image to GCR by default"
  echo "  -h display help messages"
  if [[ -n $1 ]]; then
    echo $1
  fi
  exit 1
}

function build_tf_image {
  if [ "$cpu_build" = true ] ; then
    dockerfile="Dockerfile.rbe.cpu"
    tf_image="tensorflow-rbe-cpu"
  else
    dockerfile="Dockerfile.rbe.gpu"
    tf_image="tensorflow-rbe-gpu"
  fi

  docker build -f $dockerfile -t $tf_image .
}

function publish_tf_image {
  gcr_tf_image="gcr.io/tensorflow/${tf_image}"
  docker tag $tf_image $gcr_tf_image
  gcloud docker -- push $gcr_tf_image
}

main $@
