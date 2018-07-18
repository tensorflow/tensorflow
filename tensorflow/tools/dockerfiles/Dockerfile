FROM hadolint/hadolint:latest-debian
LABEL maintainer="Austin Anderson <angerson@google.com>"

RUN apt-get update && apt-get install -y python3 python3-pip bash
RUN pip3 install --upgrade pip setuptools pyyaml absl-py cerberus

WORKDIR /tf
VOLUME ["/tf"]

COPY bashrc /etc/bash.bashrc
RUN chmod 777 /etc/bash.bashrc
