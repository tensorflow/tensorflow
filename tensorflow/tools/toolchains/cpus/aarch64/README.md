# AARCH64 toolchain

Toolchain for performing TensorFlow AARCH64 builds such as used in Github
Actions ARM_CI and ARM_CD.

Maintainer: @elfringham (Linaro LDCG)

********************************************************************************

This repository contains a toolchain for use with the specially constructed
Docker containers that match those created by SIG Build for x86 architecture
builds, but modified for AARCH64 builds.

These Docker containers have been constructed to perform builds of TensorFlow
that are compatible with manylinux2014 requirements but in an environment that
has the C++11 Dual ABI enabled.

The Docker containers are available from
[Docker Hub](https://hub.docker.com/r/linaro/tensorflow-arm64-build/tags) The
source Dockerfiles are available from
[Linaro git](https://git.linaro.org/ci/dockerfiles.git/tree/tensorflow-arm64-build)
