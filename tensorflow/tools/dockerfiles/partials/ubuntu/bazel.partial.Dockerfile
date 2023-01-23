RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install \
    Pillow \
    h5py \
    keras_preprocessing \
    tb-nightly \
    matplotlib \
    mock \
    'numpy<1.19.0' \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker \
    enum34

# Installs bazelisk
RUN mkdir /bazel && \
    curl -fSsL -o /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    mkdir /bazelisk && \
    curl -fSsL -o /bazelisk/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazelisk/master/LICENSE" && \
    curl -fSsL -o /usr/bin/bazel "https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64" && \
    chmod +x /usr/bin/bazel
