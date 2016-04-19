FROM nvidia/cuda:7.5-cudnn4-devel

MAINTAINER Jan Prach <jendap@google.com>

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    add-apt-repository -y ppa:mc3man/trusty-media && \
    add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN /install/install_deb_packages.sh
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh

# Set up bazelrc.
COPY install/.bazelrc /root/.bazelrc
ENV BAZELRC /root/.bazelrc

# Set up CUDA variables
ENV CUDA_PATH /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64

# Configure the build for our CUDA configuration.
ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV CUDNN_INSTALL_PATH /usr/local/cuda
ENV TF_NEED_CUDA 1
