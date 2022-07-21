RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig

RUN apt-get update && apt-get install -y \
    gfortran \
    libblas-dev \
    liblapack-dev

RUN python3 -m pip --no-cache-dir install \
    Pillow \
    keras_preprocessing \
    tb-nightly \
    h5py \
    matplotlib \
    mock \
    'numpy<1.19.0' \
    scipy \
    sklearn \
    pandas \
    portpicker \
    enum34

# Installs bazelisk
RUN mkdir /bazel && \
    curl -fSsL -o /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    mkdir /bazelisk && \
    curl -fSsL -o /bazelisk/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazelisk/master/LICENSE" && \
    curl -fSsL -o /usr/bin/bazel "https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-arm64" && \
    chmod +x /usr/bin/bazel
