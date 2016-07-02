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
#
# Start a Kubernetes (k8s) cluster on the local machine.
#
# This script assumes that git, docker, and golang are installed and on
# the path. It will attempt to install the version of etcd recommended by the
# kubernetes source.
#
# Usage: start_local_k8s_service.sh
#
# This script obeys the following environment variables:
# TF_DIST_K8S_SRC_DIR:     Overrides the default directory for k8s source code.
# TF_DIST_K8S_SRC_BRANCH:  Overrides the default branch to run the local k8s
#                          cluster with.


# Configurations
K8S_SRC_REPO=https://github.com/kubernetes/kubernetes.git
K8S_SRC_DIR=${TF_DIST_K8S_SRC_DIR:-/local/kubernetes}
K8S_SRC_BRANCH=${TF_DIST_K8S_SRC_BRANCH:-release-1.2}

# Helper functions
die() {
    echo $@
    exit 1
}

# Start docker service. Try multiple times if necessary.
COUNTER=0
while true; do
  ((COUNTER++))
  service docker start
  sleep 1

  service docker status
  if [[ $? == "0" ]]; then
    echo "Docker service started successfully."
    break;
  else
    echo "Docker service failed to start"

    # 23 is the exit code to signal failure to start docker service in the dind
    # container.
    exit 23

  fi
done

# Wait for docker0 net interface to appear
echo "Waiting for docker0 network interface to appear..."
while true; do
  if [[ -z $(netstat -i | grep "^docker0") ]]; then
    sleep 1
  else
    break
  fi
done
echo "docker0 interface has appeared."

# Set docker0 to promiscuous mode
ip link set docker0 promisc on || \
    die "FAILED to set docker0 to promiscuous"
echo "Turned promisc on for docker0"

# Check promiscuous mode of docker0
netstat -i

umask 000
if [[ ! -d "${K8S_SRC_DIR}/.git" ]]; then
  mkdir -p ${K8S_SRC_DIR}
  git clone ${K8S_SRC_REPO} ${K8S_SRC_DIR} || \
      die "FAILED to clone k8s source from GitHub from: ${K8S_SRC_REPO}"
fi

pushd ${K8S_SRC_DIR}
git checkout ${K8S_SRC_BRANCH} || \
    die "FAILED to checkout k8s source branch: ${K8S_SRC_BRANCH}"
git pull origin ${K8S_SRC_BRANCH} || \
    die "FAILED to pull from k8s source branch: ${K8S_SRC_BRANCH}"

# Create kubectl binary

# Install etcd
hack/install-etcd.sh

export PATH=$(pwd)/third_party/etcd:${PATH}

# Setup golang
export PATH=/usr/local/go/bin:${PATH}

echo "etcd path: $(which etcd)"
echo "go path: $(which go)"

# Create shortcut to kubectl
echo '#!/bin/bash' > /usr/local/bin/kubectl
echo "$(pwd)/cluster/kubectl.sh \\" >> /usr/local/bin/kubectl
echo '    $@' >> /usr/local/bin/kubectl
chmod +x /usr/local/bin/kubectl

# Bring up local cluster
export KUBE_ENABLE_CLUSTER_DNS=true
hack/local-up-cluster.sh

popd
