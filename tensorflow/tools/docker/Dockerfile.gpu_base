FROM b.gcr.io/tensorflow-testing/tensorflow-full

MAINTAINER Craig Citro <craigcitro@google.com>

# Set up CUDA variables and symlinks
COPY cuda /usr/local/cuda
ENV CUDA_PATH /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64

RUN echo "CUDA_PATH=/usr/local/cuda" >>~/.bash_profile
RUN echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64" >>~/.bash_profile

# Set up to build TensorFlow with GPU support.
WORKDIR /tensorflow

# Configure the build for our CUDA configuration.
ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV CUDNN_INSTALL_PATH /usr/local/cuda
ENV TF_NEED_CUDA 1
RUN ./configure

# Now we build
RUN bazel clean && \
    bazel build -c opt --config=cuda tensorflow/tools/pip_package:build_pip_package

RUN rm -rf /tmp/pip && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --upgrade /tmp/pip/tensorflow-*.whl

RUN rm -rf /usr/local/cuda

RUN ["/bin/bash"]
