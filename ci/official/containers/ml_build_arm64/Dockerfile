################################################################################
FROM ubuntu:20.04@sha256:8e5c4f0285ecbb4ead070431d29b576a530d3166df73ec44affc1cd27555141b as devel
################################################################################

# Install devtoolset build dependencies
COPY setup.sources.sh /setup.sources.sh
COPY setup.packages.sh /setup.packages.sh
COPY builder.packages.txt /builder.packages.txt

RUN /setup.sources.sh && /setup.packages.sh /builder.packages.txt

RUN update-ca-certificates
# Install devtoolset-9 in /dt10 with glibc 2.17 and libstdc++ 4.8, for building
# manylinux2014-compatible packages.
COPY builder.devtoolset/fixlinks_aarch64.sh /fixlinks.sh
COPY builder.devtoolset/rpm-patch.sh /rpm-patch.sh
COPY builder.devtoolset/build_devtoolset.sh /build_devtoolset.sh
COPY builder.devtoolset/gcc9-fixups.patch /gcc9-fixups.patch
COPY builder.devtoolset/stringop_trunc.patch /stringop_trunc.patch

RUN /build_devtoolset.sh devtoolset-10 /dt10

# Build later version of patchelf that is not so buggy
RUN wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-aarch64.tar.gz && tar -zxvf patchelf-0.18.0-aarch64.tar.gz -C /usr && rm -rf patchelf-0.18.0-aarch64.tar.gz

RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-arm.tar.gz | \
    tar zxf - google-cloud-sdk && \
    google-cloud-sdk/install.sh --quiet
ENV PATH="$PATH:/google-cloud-sdk/bin/"

# Install various tools.
# - bats: bash unit testing framework
#         NOTE: v1.6.0 seems to have a bug that made "git" in setup_file break
# - bazelisk: always use the correct bazel version
# - buildifier: clean bazel build depshttps://github.com/bazelbuild/buildtools/releases/download/v7.3.1/buildifier-linux-arm64
# - buildozer: clean bazel build deps
RUN git clone --branch v1.11.0 https://github.com/bats-core/bats-core.git && bats-core/install.sh /usr/local && rm -rf bats-core
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.21.0/bazelisk-linux-arm64 -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v7.3.1/buildifier-linux-arm64 -O /usr/local/bin/buildifier && chmod +x /usr/local/bin/buildifier
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v7.3.1/buildozer-linux-arm64 -O /usr/local/bin/buildozer && chmod +x /usr/local/bin/buildozer

RUN groupadd -g 1001 buildslave && useradd -m -u 1001 -g buildslave buildslave
RUN mkdir -p /tf/venv
RUN chown -R buildslave:buildslave /tf

# All lines past this point are reset when $CACHEBUSTER is set. We need this
# for Python specifically because we install some nightly packages which are
# likely to change daily.
ARG CACHEBUSTER=0
RUN echo $CACHEBUSTER

# Setup build and environment
COPY devel.bashrc /root/.bashrc
COPY ld.so.conf /dt10/etc/

# Make sure clang is on the path
RUN ln -s /usr/lib/llvm-18/bin/clang /usr/bin/clang

# Setup JAX Python environment.
COPY requirements.txt /requirements.txt
COPY setup.python.sh /setup.python.sh
RUN /setup.python.sh python3.9 requirements.txt
RUN /setup.python.sh python3.10 requirements.txt
RUN /setup.python.sh python3.11 requirements.txt
RUN /setup.python.sh python3.12 requirements.txt
RUN /setup.python.sh python3.13 requirements.txt
# python3.13-nogil is a free-threaded build of python3.13.
RUN /setup.python.sh python3.13-nogil requirements.txt
RUN /setup.python.sh python3.14 requirements.txt

# Python commands by default run under 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/lib/python3.11 /usr/lib/tf_python
