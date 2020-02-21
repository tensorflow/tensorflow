# Install Horovod
ARG HOROVOD_VERSION=0.16.4
RUN pip3 install --no-cache-dir horovod==${HOROVOD_VERSION}
