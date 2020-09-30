# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=

RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    python36-devel && \
    yum clean all

RUN ${PYTHON} -m pip install --no-cache-dir horovod${HOROVOD_VERSION:+==${HOROVOD_VERSION}}
