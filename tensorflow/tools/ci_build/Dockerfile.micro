FROM ubuntu:16.04

LABEL maintainer="Pete Warden <petewarden@google.com>"

# We only need a few dependencies for Micro builds, so reduce the latency by skipping
# most of the packages used by other CI images.
RUN apt-get update && apt-get install -y sudo make curl build-essential zip tar python
