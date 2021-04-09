# Install Horovod
ARG HOROVOD_WITHOUT_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_VERSION=v0.21.1

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    software-properties-common

RUN cd /usr/lib/python3/dist-packages && \
    ln -sf apt_pkg.cpython-35m-x86_64-linux-gnu.so apt_pkg.so

RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    cmake \
    g++-8 \
    gcc-8 \
    git \
    ${PYTHON}-dev

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 500 --slave /usr/bin/g++ g++ /usr/bin/g++-5 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8

RUN ${PYTHON} -m pip install git+https://github.com/horovod/horovod.git@${HOROVOD_VERSION}
