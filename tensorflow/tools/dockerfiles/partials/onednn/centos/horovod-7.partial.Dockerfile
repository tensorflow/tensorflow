# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=v0.21.1

ENV LC_ALL=en_US.UTF-8
ENV LC_CTYPE=en_US.UTF-8

RUN yum update -y && \
    yum install -y centos-release-scl && \
    yum install -y \
        devtoolset-8 \
        devtoolset-8-make \
        llvm-toolset-7-cmake \
        ${PYTHON}-devel \
        sclo-git25 && \
    yum clean all

ENV PATH=/opt/rh/devtoolset-8/root/usr/bin:/opt/rh/sclo-git25/root/usr/bin:/opt/rh/llvm-toolset-7/root/usr/bin:${PATH}

RUN ${PYTHON} -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
