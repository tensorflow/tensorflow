FROM ubuntu:14.04

MAINTAINER Shanqing Cai <cais@google.com>

RUN apt-get update
RUN apt-get install -y \
    bc \
    curl \
    python \
    python-numpy \
    python-pip

# Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
RUN chmod +x install_google_cloud_sdk.bash
RUN ./install_google_cloud_sdk.bash --disable-prompts --install-dir=/var/gcloud

# Install kubectl
RUN /var/gcloud/google-cloud-sdk/bin/gcloud components install kubectl

# Install nightly TensorFlow pip
# TODO(cais): Should we build it locally instead?
RUN pip install \
    http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# Copy test files
COPY scripts /var/tf-dist-test/scripts
COPY python /var/tf-dist-test/python
