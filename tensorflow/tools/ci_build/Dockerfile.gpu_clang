FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04

LABEL maintainer="Ilya Biryukov <ibiryukov@google.com>"

# In the Ubuntu 14.04 images, cudnn is placed in system paths. Move them to
# /usr/local/cuda
RUN cp /usr/include/cudnn.h /usr/local/cuda/include
RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa

# LLVM requires cmake version 3.4.3, but ppa:george-edison55/cmake-3.x only
# provides version 3.2.2.
# So we skip it in `install_deb_packages.sh`, and later install it from
# https://cmake.org in `install_cmake_for_clang.sh`.
RUN /install/install_deb_packages.sh --without_cmake
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh
RUN /install/install_golang.sh

# Install cmake and build clang
RUN /install/install_cmake_for_clang.sh
RUN /install/build_and_install_clang.sh

# Set up the master bazelrc configuration file.
COPY install/.bazelrc /etc/bazel.bazelrc
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Configure the build for our CUDA configuration.
ENV TF_NEED_CUDA 1
ENV TF_CUDA_CLANG 1
ENV CLANG_CUDA_COMPILER_PATH /usr/local/bin/clang
ENV TF_CUDA_COMPUTE_CAPABILITIES 3.0
