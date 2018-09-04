FROM ubuntu:16.04

LABEL maintainer="Cong Xu <cong.xu@intel.com>"

# These parameters can be overridden by parameterized_docker_build.sh
ARG TF_BUILD_VERSION=r1.9
ARG PYTHON="python"
ARG PYTHON3_DEV=""
ARG WHL_DIR="/tmp/pip"
ARG PIP="pip"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        ${PYTHON3_DEV} \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        wget \
        numactl \
        openssh-client \
        openssh-server \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    ${PYTHON} get-pip.py && \
    rm get-pip.py

RUN ${PIP} --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        keras_applications==1.0.5 \
        keras_preprocessing==1.0.3 \
        matplotlib \
        mock \
        numpy \
        scipy \
        sklearn \
        pandas \
        && \
    ${PYTHON} -m ipykernel.kernelspec

RUN if [ "${PYTHON}" = "python3" ]; then \
  ln -s -f /usr/bin/python3 /usr/bin/python; \
  fi

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.15.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
WORKDIR /tensorflow

# Download and build TensorFlow.
# Enable checking out both tags and branches
RUN export TAG_PREFIX="v" && \
    echo ${TF_BUILD_VERSION} | grep -q ^${TAG_PREFIX}; \
    if [ $? -eq 0 ]; then \
        git clone --depth=1 https://github.com/tensorflow/tensorflow.git . && \
        git fetch --tags && \
        git checkout ${TF_BUILD_VERSION}; \
   else \
        git clone --depth=1 --branch=${TF_BUILD_VERSION} https://github.com/tensorflow/tensorflow.git . ; \
    fi

RUN yes "" | ${PYTHON} configure.py

ENV CI_BUILD_PYTHON ${PYTHON}

# Set bazel build parameters in .bazelrc in parameterized_docker_build.sh
# Use --copt=-march values to get optimized builds appropriate for the hardware
#   platform of your choice.
# For ivy-bridge or sandy-bridge
# --copt=-march="avx" \
# For haswell, broadwell, or skylake
# --copt=-march="avx2" \
COPY .bazelrc /root/.bazelrc

RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel --bazelrc=/root/.bazelrc build -c opt \
    tensorflow/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package "${WHL_DIR}" && \
    ${PIP} --no-cache-dir install --upgrade "${WHL_DIR}"/tensorflow-*.whl && \
    rm -rf /root/.cache
# Clean up Bazel cache when done.

WORKDIR /root

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar zxf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
RUN echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Install Horovod
RUN ${PIP} install --no-cache-dir horovod

# Install OpenSSH for MPI to communicate between containers
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR /root
