FROM ubuntu:16.04

LABEL authors="Andrew Gibiansky <andrew.gibiansky@gmail.com>, Joel Hestness <jthestness@gmail.com>"

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    add-apt-repository -y ppa:mc3man/trusty-media && \
    add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN /install/install_deb_packages.sh
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh
RUN /install/install_proto3.sh
RUN /install/install_buildifier.sh
RUN /install/install_mpi.sh

# Set up bazelrc.
COPY install/.bazelrc /root/.bazelrc
ENV BAZELRC /root/.bazelrc

# Set up MPI
ENV TF_NEED_MPI 1
ENV MPI_HOME /usr/lib/openmpi
