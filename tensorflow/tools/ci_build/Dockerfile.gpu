FROM nvidia/cuda:8.0-cudnn5-devel

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
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Configure the build for our CUDA configuration.
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES 3.0
