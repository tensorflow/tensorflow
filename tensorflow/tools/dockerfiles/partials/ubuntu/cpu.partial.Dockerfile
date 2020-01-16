FROM ubuntu:${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y curl
