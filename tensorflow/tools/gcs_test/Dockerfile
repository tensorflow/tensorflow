FROM ubuntu:24.04@sha256:440dcf6a5640b2ae5c77724e68787a906afb8ddee98bf86db94eea8528c2c076

LABEL maintainer="Shanqing Cai <cais@google.com>"

RUN apt-get update
RUN apt-get install -y \
    curl \
    libcurl4-openssl-dev \
    python \
    python-pip

# Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
RUN chmod +x install_google_cloud_sdk.bash
RUN ./install_google_cloud_sdk.bash --disable-prompts --install-dir=/var/gcloud

# Install TensorFlow pip from build context.
COPY tensorflow-*.whl /
RUN pip install /tensorflow-*.whl

# Copy test files
RUN mkdir -p /gcs-smoke/python
COPY gcs_smoke_wrapper.sh /gcs-smoke/
COPY python/gcs_smoke.py /gcs-smoke/python/
