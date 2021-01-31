# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=0.21.0

RUN yum update -y && yum install -y \
    cmake \
    gcc \
    gcc-c++ \
    git \
    make \
    python36-devel && \
    yum clean all

RUN ${PYTHON} -m pip install git+https://github.com/horovod/horovod.git@v${HOROVOD_VERSION}
