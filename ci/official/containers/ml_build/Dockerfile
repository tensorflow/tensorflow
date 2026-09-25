# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

################################################################################
ARG BASE_IMAGE=ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986
FROM $BASE_IMAGE AS devel
# See https://docs.docker.com/reference/dockerfile/#understand-how-arg-and-from-interact
# on why we cannot reference BASE_IMAGE again unless we declare it again.
################################################################################

# Install devtoolset build dependencies
COPY setup.sources.sh /setup.sources.sh
COPY setup.packages.sh /setup.packages.sh
COPY builder.packages.txt /builder.packages.txt

RUN /setup.sources.sh && /setup.packages.sh /builder.packages.txt

# Setup Python
COPY setup.python.sh /setup.python.sh
COPY builder.requirements.txt /builder.requirements.txt
RUN /setup.python.sh python3.10 /builder.requirements.txt
RUN /setup.python.sh python3.11 /builder.requirements.txt
RUN /setup.python.sh python3.13 /builder.requirements.txt
RUN /setup.python.sh python3.13-nogil /builder.requirements.txt
RUN /setup.python.sh python3.14 /builder.requirements.txt
RUN /setup.python.sh python3.14-nogil /builder.requirements.txt
RUN /setup.python.sh python3.15 /builder.requirements.txt
RUN /setup.python.sh python3.15-nogil /builder.requirements.txt

# Since we are using python3.12 as the default python version, we need to
# install python3.12 last for now.
# TODO(b/376338367): switch to pyenv.
RUN /setup.python.sh python3.12 /builder.requirements.txt

ARG INSTALL_NVIDIA_PACKAGES=false
ARG NVIDIA_PACKAGES_FILE=cuda12.1_cudnn9.8.packages.txt
COPY ${NVIDIA_PACKAGES_FILE} /nvidia.packages.txt
COPY setup.sources.cudnn.sh /setup.sources.cudnn.sh

ARG INSTALL_CMAKE_TOOLCHAIN=false
ARG CMAKE_TOOLCHAIN_PACKAGES_FILE=cmake_toolchain.packages.txt
COPY ${CMAKE_TOOLCHAIN_PACKAGES_FILE} /cmake_toolchain.packages.txt
COPY setup.toolchain.cmake.sh /setup.toolchain.cmake.sh

RUN if [ "${INSTALL_NVIDIA_PACKAGES}" = "true" ]; then \
      echo "Installing Nvidia packages"; \
      /setup.sources.cudnn.sh && /setup.packages.sh /nvidia.packages.txt; \
    else \
      echo "Nvidia packages will not be installed"; \
    fi

RUN if [ "${INSTALL_CMAKE_TOOLCHAIN}" = "true" ]; then \
      echo "Installing CMake toolchain"; \
      /setup.toolchain.cmake.sh && /setup.packages.sh /cmake_toolchain.packages.txt; \
    else \
      echo "CMake toolchain will not be installed"; \
    fi

# Setup links for TensorFlow to compile.
# Referenced in devel.usertools/*.bazelrc.
# Set python3.12 as the default python version.
# TF does not support python3.13.
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3
RUN ln -sf /usr/bin/python3.12 /usr/bin/python
RUN ln -sf /usr/lib/python3.12 /usr/lib/tf_python

# Link the compat driver to the location if available.
RUN if [ -e "/usr/local/cuda/compat/libcuda.so.1" ]; then ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; fi

# Install various tools.
# - bats: bash unit testing framework
# - bazelisk: always use the correct bazel version
# - buildifier: clean bazel build deps
# - buildozer: clean bazel build deps
# - gcloud SDK: communicate with Google Cloud Platform (GCP) for RBE, CI
# - patchelf: Utility tool to modify existing ELF executables and libraries
RUN git clone --branch v1.14.0 https://github.com/bats-core/bats-core.git && bats-core/install.sh /usr/local && rm -rf bats-core
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    echo "5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992  /usr/local/bin/bazel" | sha256sum --check && \
    chmod +x /usr/local/bin/bazel
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v10.1.0/buildifier-linux-amd64 -O /usr/local/bin/buildifier && \
    echo "31b6a8aa1e5c746696788f428729701770ad91925873d8256cb885c60e12c77e  /usr/local/bin/buildifier" | sha256sum --check && \
    chmod +x /usr/local/bin/buildifier
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v10.1.0/buildozer-linux-amd64 -O /usr/local/bin/buildozer && \
    echo "3513b8b23619f5fb7ccad546e955af6e318475582ce7b6f2b47b634bd3b8dcbd  /usr/local/bin/buildozer" | sha256sum --check && \
    chmod +x /usr/local/bin/buildozer

RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-586.0.0-linux-x86_64.tar.gz && \
    echo "c1cd1823624a33f2341d0d384aafe2d7b24c1b77c9b131087d772ee0791ff2be  google-cloud-cli-586.0.0-linux-x86_64.tar.gz" | sha256sum --check && \
    tar zxf google-cloud-cli-586.0.0-linux-x86_64.tar.gz && \
    rm google-cloud-cli-586.0.0-linux-x86_64.tar.gz && \
    google-cloud-sdk/install.sh --quiet && \
    ln -s /google-cloud-sdk/bin/gcloud /usr/bin/gcloud
ENV PATH="$PATH:/google-cloud-sdk/bin/:/usr/local/cuda/bin/"

# Download and install gh CLI v2.101.0 from GitHub.
RUN wget https://github.com/cli/cli/releases/download/v2.101.0/gh_2.101.0_linux_amd64.tar.gz && \
    echo "9bca2d1c16825f109907a23307628a2f0698fbf99662b73a5cf0b020293072b8  gh_2.101.0_linux_amd64.tar.gz" | sha256sum --check && \
    tar zxf gh_2.101.0_linux_amd64.tar.gz && \
    mv gh_2.101.0_linux_amd64/bin/gh /usr/local/bin/gh && \
    rm -rf gh_2.101.0_linux_amd64*

# Download and install patchelf v0.18.0 from GitHub. The default Ubuntu focal
# packages only provide the "0.10-2build1" version. We use patchelf to manipulate
# certain shared libraries during the wheel building process (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/build_pip_package.sh#L255-L262).
# When we use Patchelf versions <0.12, those shared libraries end up with a
# corrupted PT_NOTE program header. This was fixed in v0.12, see https://github.com/NixOS/patchelf/commit/43a33482b501b0f5ee9da312aabfca3806570cc9.
RUN wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz && \
    echo "ce84f2447fb7a8679e58bc54a20dc2b01b37b5802e12c57eece772a6f14bf3f0  patchelf-0.18.0-x86_64.tar.gz" | sha256sum --check && \
    tar -zxvf patchelf-0.18.0-x86_64.tar.gz -C /usr && \
    rm -rf patchelf-0.18.0-x86_64.tar.gz

# Don't use the bazel cache when a new docker image is created.
RUN echo build --action_env=DOCKER_CACHEBUSTER=$(date +%s%N)$RANDOM >> /etc/bazel.bazelrc
RUN echo build --host_action_env=DOCKER_HOST_CACHEBUSTER=$(date +%s%N)$RANDOM >> /etc/bazel.bazelrc
