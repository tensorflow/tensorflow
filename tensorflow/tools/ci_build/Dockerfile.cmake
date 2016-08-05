FROM ubuntu:16.04

MAINTAINER Shanqing Cai <cais@google.com>

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN /install/install_deb_packages.sh
RUN /install/install_proto3_from_source.sh

RUN pip install --upgrade numpy

# Install golang
RUN add-apt-repository -y ppa:ubuntu-lxc/lxd-stable
RUN apt-get update
RUN apt-get install -y golang
