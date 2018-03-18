FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL maintainer="Nick Lopez <ngiraldo@google.com>"

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
RUN /install/install_golang.sh

# Install clang from pre-built package
RUN cd /tmp && \
    wget https://storage.googleapis.com/clang-builds-stable/clang-ubuntu16_04/clang_r323528.tar.gz && \
    echo "26752d9f5785df07193fac8316ba5d5ba3bec36d970c29a1577360848818ac74  clang_r323528.tar.gz" | sha256sum -c && \
    tar -C /usr/local -xf clang_r323528.tar.gz && \
    rm clang_r323528.tar.gz

