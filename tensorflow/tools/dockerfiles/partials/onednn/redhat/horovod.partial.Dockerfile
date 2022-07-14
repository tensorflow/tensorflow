# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=v0.21.1

RUN yum update -y && yum install -y \
    cmake \
    gcc \
    gcc-c++ \
    git \
    make \
    ${PYTHON}-devel && \
    yum clean all

RUN ${PYTHON} -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
