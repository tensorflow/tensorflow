ARG IMAGE
FROM ${IMAGE}
ARG PYTHON_VERSION

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN apt-get update && \
    apt-get install -y \
      software-properties-common \
      debhelper \
      crossbuild-essential-armhf \
      crossbuild-essential-arm64 \
      zlib1g-dev  \
      zlib1g-dev:armhf \
      zlib1g-dev:arm64 \
      curl \
      unzip \
      git && \
    apt-get clean

RUN yes | add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y \
      python$PYTHON_VERSION \
      python$PYTHON_VERSION-dev \
      python$PYTHON_VERSION-distutils \
      libpython$PYTHON_VERSION-dev \
      libpython$PYTHON_VERSION-dev:armhf \
      libpython$PYTHON_VERSION-dev:arm64
RUN ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python3
RUN curl -OL https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py
RUN pip3 install --upgrade pip
RUN pip3 install numpy~=1.19.2 setuptools pybind11
RUN ln -sf /usr/include/python$PYTHON_VERSION /usr/include/python3
RUN ln -sf /usr/local/lib/python$PYTHON_VERSION/dist-packages/numpy/core/include/numpy /usr/include/python3/numpy
RUN curl -OL https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh cmake-3.16.8-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
