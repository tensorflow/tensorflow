FROM registry.access.redhat.com/ubi8/ubi:${REDHAT_VERSION} AS base

ARG REDHAT_VERSION=8

# Enable both PowerTools and EPEL otherwise some packages like hdf5-devel fail to install
RUN dnf --disableplugin=subscription-manager install -y 'dnf-command(config-manager)' && \
    dnf --disableplugin=subscription-manager config-manager --set-enabled powertools && \
    dnf --disableplugin=subscription-manager install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-"${REDHAT_VERSION}".noarch.rpm && \
    dnf --disableplugin=subscription-manager clean all
