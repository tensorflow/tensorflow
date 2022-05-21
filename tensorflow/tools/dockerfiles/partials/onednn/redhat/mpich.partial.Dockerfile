# Accessing Certain repos require RedHat subscription
ARG SUBSCRIPTION_ORG
ARG SUBSCRIPTION_KEY

RUN subscription-manager register --org=$SUBSCRIPTION_ORG --activationkey=$SUBSCRIPTION_KEY && \
    subscription-manager attach && \
    subscription-manager release --set=$(cat /etc/*release | grep VERSION_ID | cut -f2 -d'"')

# install mpich, openssh for MPI to communicate between containers
RUN INSTALL_PKGS="\
    mpich \
    mpich-devel \
    openssh \
    openssh-server" && \
    yum -y --setopt=tsflags=nodocs install $INSTALL_PKGS && \
    rpm -V $INSTALL_PKGS && \
    yum -y clean all --enablerepo='*'

RUN subscription-manager unregister

ENV PATH="/usr/lib64/mpich/bin:${PATH}"

# Create a wrapper for MPICH to allow running as root by default
RUN mv -f $(which mpirun) /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Set up SSH
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/sshd_config | grep -v StrictHostKeyChecking > /etc/ssh/sshd_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/sshd_config.new && \
    mv -f /etc/ssh/sshd_config.new /etc/ssh/sshd_config
