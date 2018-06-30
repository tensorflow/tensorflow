FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer="Jan Prach <jendap@google.com>"

# In the Ubuntu 16.04 images, cudnn is placed in system paths. Move them to
# /usr/local/cuda
RUN cp -P /usr/include/cudnn.h /usr/local/cuda/include
RUN cp -P /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64

# Copy and run the install scripts.
COPY install/*.sh /install/
ARG DEBIAN_FRONTEND=noninteractive
RUN /install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN /install/install_deb_packages.sh
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh
RUN /install/install_golang.sh

# Set up the master bazelrc configuration file.
COPY install/.bazelrc /etc/bazel.bazelrc
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Configure the build for our CUDA configuration.
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES 3.0
