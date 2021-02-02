FROM centos:${CENTOS_VERSION} AS base

ARG CENTOS_VERSION=8

# Enable both PowerTools and EPEL otherwise some packages like hdf5-devel fail to install
RUN dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-"${CENTOS_VERSION}".noarch.rpm && \
    dnf clean all
