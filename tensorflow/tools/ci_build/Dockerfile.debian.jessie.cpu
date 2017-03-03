FROM debian:jessie

MAINTAINER Jan Prach <jendap@google.com>

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN echo "deb http://http.debian.net/debian jessie-backports main" | tee -a /etc/apt/sources.list
RUN /install/install_deb_packages.sh
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh
RUN /install/install_golang.sh

# Fix a virtualenv install issue specific to Debian Jessie.
RUN pip install --upgrade virtualenv

# Set up bazelrc.
COPY install/.bazelrc /root/.bazelrc
ENV BAZELRC /root/.bazelrc
