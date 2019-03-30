FROM ubuntu:18.04

LABEL maintainer="Clayne Robison <clayne.b.robison@intel.com>"

# This parameter MUST be set by parameterized_docker_build.sh
ARG TF_WHL_URL

# Optional parameters
ARG TF_BUILD_VERSION=r1.13
ARG PYTHON="python"
ARG PYTHON_DEV="python-dev"
ARG PIP="pip"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        ${PYTHON} \
        ${PYTHON}-dev \
        ${PYTHON}-pip \
        ${PYTHON}-setuptools \
        ${PYTHON}-wheel \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN ${PIP} --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        keras_applications \
        keras_preprocessing \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        && \
    ${PYTHON} -m ipykernel.kernelspec


COPY ${TF_WHL_URL} /
RUN ${PIP} install --no-cache-dir --force-reinstall /${TF_WHL_URL} && \
    rm -rf /${TF_WHL_URL}


# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--allow-root"]
