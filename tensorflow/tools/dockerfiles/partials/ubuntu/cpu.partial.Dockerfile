FROM ubuntu:${UBUNTU_VERSION} as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*
