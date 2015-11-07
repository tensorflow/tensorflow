FROM ipython/notebook:latest

MAINTAINER Craig Citro <craigcitro@google.com>

# Set up Bazel.
# Install dependencies for bazel.
RUN apt-get update && apt-get install -y \
    pkg-config \
    zip \
    g++ \
    zlib1g-dev \
    unzip \
    swig \
    software-properties-common \
    wget

# We need to add a custom PPA to pick up JDK8, since trusty doesn't
# have an openjdk8 backport.  openjdk-r is maintained by a reliable contributor:
# Matthias Klose (https://launchpad.net/~doko).  It will do until
# we either update the base image beyond 14.04 or openjdk-8 is
# finally backported to trusty; see e.g.
#   https://bugs.launchpad.net/trusty-backports/+bug/1368094
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y openjdk-8-jdk openjdk-8-jre-headless

# Set up CUDA variables and symlinks
COPY cuda /usr/local/cuda
ENV CUDA_PATH /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
RUN ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/root/.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/root/.bazelrc
ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.1.1
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    wget -O /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE.txt
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
WORKDIR /tensorflow
# Pick up some TF dependencies
RUN apt-get update && \
    apt-get install -y python-numpy && \
    apt-get install -y libfreetype6-dev

# We can't clone the TF git repo yet, because of permissions issues.
# RUN git clone https://tensorflow.googlesource.com/
# Instead, we manually copy it in:
COPY tensorflow /tensorflow

# Set up the CUDA tensorflow directories
RUN rm -rf /tensorflow/third_party/gpus/cuda/lib64
RUN rm -rf /tensorflow/third_party/gpus/cuda/bin
RUN rm -rf /tensorflow/third_party/gpus/cuda/include
RUN rm -rf /tensorflow/third_party/gpus/cuda/nvvm
RUN ln -s /usr/local/cuda/lib64 /tensorflow/third_party/gpus/cuda/
RUN ln -s /usr/local/cuda/bin /tensorflow/third_party/gpus/cuda/
RUN ln -s /usr/local/cuda/include /tensorflow/third_party/gpus/cuda/
RUN ln -s /usr/local/cuda/nvvm /tensorflow/third_party/gpus/cuda/

# Now we build
RUN bazel clean && \
    bazel build -c opt --config=cuda tensorflow/tools/docker:simple_console

ENV PYTHONPATH=/tensorflow/bazel-bin/tensorflow/tools/docker/simple_console.runfiles/:$PYTHONPATH

# Add any notebooks in this directory.
COPY notebooks /notebooks

# Add variables for the local IPython. This sets a fixed password and
# switches to HTTP (to avoid self-signed certificate warnings in
# Chrome).
ENV PASSWORD=JustForNow
ENV USE_HTTP=1

RUN if [ -f /notebooks/requirements.txt ];\
        then pip install -r /notebooks/requirements.txt;\
    fi

# Set the workdir so we see notebooks on the IPython landing page.
WORKDIR /notebooks

# Remove CUDA libraries, headers, nvcc.  The user will have to
# provide this directly when running docker.
RUN rm -rf /usr/local/cuda
