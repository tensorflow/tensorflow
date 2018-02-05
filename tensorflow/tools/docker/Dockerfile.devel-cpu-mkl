FROM tensorflow/tensorflow:latest-devel

LABEL maintainer="Clayne Robison<clayne.b.robison@intel.com>"

# These arguments are parameterized. Use --build-args to override.
ARG TF_BRANCH=r1.5
ARG WHL_DIR=/whl

RUN apt-get update && apt-get install -y --no-install-recommends \
        golang \
        vim \
        emacs \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install --upgrade \
        pip setuptools

RUN pip --no-cache-dir install wheel 

# Download and build TensorFlow.
WORKDIR /
RUN rm -rf tensorflow && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout ${TF_BRANCH}
WORKDIR /tensorflow

# Configure the build for CPU with MKL by accepting default build options and
# setting library locations
ENV CI_BUILD_PYTHON=python \
   LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
    PYTHON_BIN_PATH=/usr/bin/python \
    PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages \
    CC_OPT_FLAGS='-march=native' \
    TF_NEED_JEMALLOC=0 \
    TF_NEED_GCP=0 \
    TF_NEED_CUDA=0 \
    TF_NEED_HDFS=0 \
    TF_NEED_S3=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_GDR=0 \
    TF_ENABLE_XLA=0 \
    TF_NEED_VERBS=0 \
    TF_NEED_MPI=0
RUN ./configure

# Build and Install TensorFlow.
# The 'mkl' option builds with Intel(R) Math Kernel Library (MKL), which detects
# the platform it is currently running on and takes appropriately optimized 
# paths. The -march=native option is for code that is not in MKL, and assumes
# this container will be run on the same architecture on which it is built.
RUN LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
    bazel build --config=mkl \
                --config="opt" \
                --copt="-march=broadwell" \
                --copt="-O3" \
                //tensorflow/tools/pip_package:build_pip_package && \
    mkdir ${WHL_DIR} && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ${WHL_DIR}

# Clean up Bazel cache when done, but leave the whl.
# This will upgrade the default Tensorflow version with the Intel MKL version
RUN pip --no-cache-dir install --upgrade ${WHL_DIR}/tensorflow-*.whl && \
    rm -rf /root/.cache

WORKDIR /root

#add welcome message with instructions

RUN echo '[ ! -z "$TERM" -a -r /etc/motd ] && cat /etc/issue && cat /etc/motd' \
	>> /etc/bash.bashrc \
	; echo "\
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\
|								\n\
| Docker container running Ubuntu				\n\
| with TensorFlow ${TF_BRANCH} optimized for CPU		\n\
| with Intel(R) MKL						\n\
|								\n\
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\
\n "\
	> /etc/motd
