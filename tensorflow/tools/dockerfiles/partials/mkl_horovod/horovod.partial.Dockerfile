# Install Horovod
ARG HOROVOD_VERSION=0.16.4
RUN python3 -m pip install --no-cache-dir horovod==${HOROVOD_VERSION}
