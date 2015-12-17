FROM ubuntu:14.04

MAINTAINER Jan Prach <jendap@google.com>

# Copy and run the install scripts.
COPY install/install_deb_packages.sh /install/install_deb_packages.sh
RUN /install/install_deb_packages.sh
COPY install/install_openjdk8_from_ppa.sh /install/install_openjdk8_from_ppa.sh
RUN /install/install_openjdk8_from_ppa.sh
COPY install/install_bazel.sh /install/install_bazel.sh
RUN /install/install_bazel.sh

# Set up bazelrc.
COPY install/.bazelrc /root/.bazelrc
ENV BAZELRC /root/.bazelrc
