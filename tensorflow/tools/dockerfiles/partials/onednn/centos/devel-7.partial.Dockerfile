FROM centos:${CENTOS_VERSION} AS base

# Enable both PowerTools and EPEL otherwise some packages like hdf5-devel fail to install
RUN yum clean all && \
    yum update -y && \
    yum install -y epel-release
