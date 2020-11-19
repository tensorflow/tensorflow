FROM centos:${CENTOS_VERSION} AS base

ARG CENTOS_VERSION=8

# Enable both PowerTools and EPEL otherwise some packages like hdf5-devel fail to install
RUN dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --set-enabled PowerTools && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-"${CENTOS_VERSION}".noarch.rpm && \
    dnf clean all

RUN yum update -y && \
    yum install -y \
        curl \
        freetype-devel \
        gcc \
        gcc-c++ \
        git \
        hdf5-devel \
        java-1.8.0-openjdk \
        java-1.8.0-openjdk-headless \
        libcurl-devel \
        make \
        pkg-config \
        rsync \
        sudo \
        unzip \
        zeromq-devel \
        zip \
        zlib-devel && \
        yum clean all

ENV CI_BUILD_PYTHON python

# CACHE_STOP is used to rerun future commands, otherwise cloning tensorflow will be cached and will not pull the most recent version
ARG CACHE_STOP=1
# Check out TensorFlow source code if --build-arg CHECKOUT_TF_SRC=1
ARG CHECKOUT_TF_SRC=0
ARG TF_BRANCH=master
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git --branch "${TF_BRANCH}" --single-branch /tensorflow_src || true
