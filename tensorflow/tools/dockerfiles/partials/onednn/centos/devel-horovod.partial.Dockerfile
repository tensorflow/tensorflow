# Check out horovod source code if --build-arg CHECKOUT_HOROVOD_SRC=1
ARG CHECKOUT_HOROVOD_SRC=0
ARG HOROVOD_BRANCH=master
RUN test "${CHECKOUT_HOROVOD_SRC}" -eq 1 && git clone --branch "${HOROVOD_BRANCH}" --single-branch --recursive https://github.com/uber/horovod.git /horovod_src || true
