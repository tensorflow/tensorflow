FROM registry.access.redhat.com/ubi8/ubi:${REDHAT_VERSION} as base

### Required OpenShift Labels
LABEL name="Intel&#174; Optimizations for TensorFlow* with Open MPI* and Horovod*" \
      maintainer="Abolfazl Shahbazi <abolfazl.shahbazi@intel.com>" \
      vendor="Intel&#174; Corporation" \
      version="2.7.0" \
      release="2.7.0" \
      summary="Intel&#174; Optimizations for TensorFlow* with Open MPI* and Horovod* is a binary distribution of TensorFlow* with Intel&#174; oneAPI Deep Neural Network Library primitives." \
      description="Intel&#174; Optimizations for TensorFlow* with Open MPI* and Horovod* is a binary distribution of TensorFlow* with Intel&#174; oneAPI Deep Neural Network Library (Intel&#174; oneDNN) primitives, a popular performance library for deep learning applications. TensorFlow* is a widely-used machine learning framework in the deep learning arena, demanding efficient utilization of computational resources. In order to take full advantage of Intel&#174; architecture and to extract maximum performance, the TensorFlow* framework has been optimized using Intel&#174; oneDNN primitives."

# Licenses, Legal Notice and TPPs for older versions
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/LEGAL-NOTICE ./licenses/
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/LICENSE ./licenses/
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/third_party_programs_license/oneDNN-THIRD-PARTY-PROGRAMS ./licenses/third_party_programs_license/
ADD https://raw.githubusercontent.com/Intel-tensorflow/tensorflow/v2.7.0/third_party_programs_license/third-party-programs.txt ./licenses/third_party_programs_license/

ENV LANG C.UTF-8
ARG PYTHON=python3

### Add necessary updates here
RUN yum -y update-minimal --security --sec-severity=Important --sec-severity=Critical

RUN INSTALL_PKGS="\
    ${PYTHON}-pip \
    which" && \
    yum -y --setopt=tsflags=nodocs install $INSTALL_PKGS && \
    rpm -V $INSTALL_PKGS && \
    yum -y clean all --enablerepo='*'

# Intel Optimizations specific Envs
ENV KMP_AFFINITY='granularity=fine,verbose,compact,1,0' \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1
