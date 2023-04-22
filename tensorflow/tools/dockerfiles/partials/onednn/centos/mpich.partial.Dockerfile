# install mpich, openssh for MPI to communicate between containers
RUN yum update -y && yum install -y \
    mpich \
    mpich-devel \
    openssh \
    openssh-server \
    redhat-rpm-config \
    which && \
    yum clean all

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
