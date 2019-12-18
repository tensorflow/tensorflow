# Install Horovod
ARG HOROVOD_VERSION=0.16.4
RUN ${PIP} install --no-cache-dir horovod==${HOROVOD_VERSION}
