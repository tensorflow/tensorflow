################################################################################
FROM ubuntu:22.04@sha256:4e0171b9275e12d375863f2b3ae9ce00a4c53ddda176bd55868df97ac6f21a6e as builder
################################################################################

# Install devtoolset build dependencies
COPY setup.packages.sh setup.packages.sh
COPY builder.packages.txt builder.packages.txt
RUN /setup.packages.sh /builder.packages.txt

# Install devtoolset-9 in /dt9 with glibc 2.17 and libstdc++ 4.8, for building
# manylinux2014-compatible packages.
COPY builder.devtoolset/fixlinks.sh /fixlinks.sh
COPY builder.devtoolset/rpm-patch.sh /rpm-patch.sh
COPY builder.devtoolset/build_devtoolset.sh /build_devtoolset.sh
COPY builder.devtoolset/glibc2.17-inline.patch /glibc2.17-inline.patch
RUN /build_devtoolset.sh devtoolset-9 /dt9

################################################################################
FROM nvidia/cuda:12.3.2-base-ubuntu22.04@sha256:8cecfe099315f73127d6d5cc43fce32c7ffff4ea0460eefac48f2b7d811ce857 as devel
################################################################################
COPY --from=builder /dt9 /dt9

# Install required development packages but delete unneeded CUDA bloat
# CUDA must be cleaned up in the same command to prevent Docker layer bloating
COPY setup.sources.sh /setup.sources.sh
COPY setup.packages.sh /setup.packages.sh
COPY setup.cuda.sh /setup.cuda.sh
COPY devel.packages.txt /devel.packages.txt
RUN /setup.sources.sh && /setup.packages.sh /devel.packages.txt && /setup.cuda.sh

# Make sure clang is on the path
RUN ln -s /usr/lib/llvm-18/bin/clang /usr/bin/clang

# Install various tools.
# - bats: bash unit testing framework
# - bazelisk: always use the correct bazel version
# - buildifier: clean bazel build deps
# - buildozer: clean bazel build deps
# - gcloud SDK: communicate with Google Cloud Platform (GCP) for RBE, CI
# - patchelf: Utility tool to modify existing ELF executables and libraries
RUN git clone --branch v1.7.0 https://github.com/bats-core/bats-core.git && bats-core/install.sh /usr/local && rm -rf bats-core
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel
RUN wget https://github.com/bazelbuild/buildtools/releases/download/3.5.0/buildifier -O /usr/local/bin/buildifier && chmod +x /usr/local/bin/buildifier
RUN wget https://github.com/bazelbuild/buildtools/releases/download/3.5.0/buildozer -O /usr/local/bin/buildozer && chmod +x /usr/local/bin/buildozer
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcloud && bash /tmp/gcloud --install-dir=~/usr/local/bin --disable-prompts
# Download and install patchelf v0.18.0 from GitHub. The default Ubuntu focal
# packages only provide the "0.10-2build1" version. We use patchelf to manipulate
# certain shared libraries during the wheel building process (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/build_pip_package.sh#L255-L262).
# When we use Patchelf versions <0.12, those shared libraries end up with a
# corrupted PT_NOTE program header. This was fixed in v0.12, see https://github.com/NixOS/patchelf/commit/43a33482b501b0f5ee9da312aabfca3806570cc9.
RUN wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz && tar -zxvf patchelf-0.18.0-x86_64.tar.gz -C /usr && rm -rf patchelf-0.18.0-x86_64.tar.gz

# All lines past this point are reset when $CACHEBUSTER is set. We need this
# for Python specifically because we install some nightly packages which are
# likely to change daily.
ARG CACHEBUSTER=0
RUN echo $CACHEBUSTER

# Setup Python environment. PYTHON_VERSION is e.g. "python3.8"
ARG PYTHON_VERSION
COPY setup.python.sh /setup.python.sh
COPY devel.requirements.txt /devel.requirements.txt
RUN /setup.python.sh $PYTHON_VERSION devel.requirements.txt

# Setup build and environment
COPY devel.usertools /usertools
COPY devel.bashrc /root/.bashrc

# Don't use the bazel cache when a new docker image is created.
RUN echo build --action_env=DOCKER_CACHEBUSTER=$(date +%s%N)$RANDOM >> /etc/bazel.bazelrc
RUN echo build --host_action_env=DOCKER_HOST_CACHEBUSTER=$(date +%s%N)$RANDOM >> /etc/bazel.bazelrc
