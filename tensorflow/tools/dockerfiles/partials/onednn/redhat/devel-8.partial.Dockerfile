FROM registry.access.redhat.com/ubi8/ubi:${REDHAT_VERSION} AS base

ARG REDHAT_VERSION=latest

# Enable EPEL otherwise some packages like hdf5-devel fail to install
RUN dnf install -y 'dnf-command(config-manager)' && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
    dnf clean all
