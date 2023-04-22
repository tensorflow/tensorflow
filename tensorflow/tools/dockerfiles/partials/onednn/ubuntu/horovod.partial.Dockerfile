# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=v0.21.1

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    cmake \
    g++-8 \
    gcc-8 \
    git \
    ${PYTHON}-dev

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 700 --slave /usr/bin/g++ g++ /usr/bin/g++-7 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

RUN ${PYTHON} -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
