ARG DEBIAN_FRONTEND="noninteractive"

# install mpich, openssh for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    mpich \
    libmpich-dev \
    openssh-client \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a wrapper for MPICH to allow running as root by default
RUN mv /usr/bin/mpirun /usr/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/bin/mpirun && \
    echo 'mpirun.real "$@"' >> /usr/bin/mpirun && \
    chmod a+x /usr/bin/mpirun

# Disable GCC noise for gcc newer than 5.x, otherwise Horovod installation fails
RUN sed -i 's/# if __GNUC__ > 5/# if __GNUC__ > 9/g' /usr/include/mpich/mpicxx.h


# Set up SSH
RUN mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
