ARG IMAGE
FROM ${IMAGE}

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN apt-get update && \
    apt-get install -y \
      debhelper \
      dh-python \
      python-all \
      python-setuptools \
      python-wheel \
      python-numpy \
      python-pip \
      pybind11-dev \
      libpython-dev \
      libpython-dev:armhf \
      libpython-dev:arm64 \
      python3-all \
      python3-setuptools \
      python3-wheel \
      python3-numpy \
      python3-pip \
      libpython3-dev \
      libpython3-dev:armhf \
      libpython3-dev:arm64 \
      crossbuild-essential-armhf \
      crossbuild-essential-arm64 \
      zlib1g-dev  \
      zlib1g-dev:armhf \
      zlib1g-dev:arm64 \
      curl \
      unzip \
      git && \
    apt-get clean
RUN pip install pip --upgrade
RUN pip install pybind11
RUN pip3 install pip --upgrade
RUN pip3 install pybind11
