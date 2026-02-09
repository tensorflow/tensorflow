#!/bin/bash
# This script builds a docker container, and pip wheels for all supported
# Python versions.  Run from the root tensorflow-text directory:
#
# ./oss_scripts/docker_builds.sh

set -e -x

# If a specific PYTHON_VERSION is specified, only build that one.
# Otherwisebuild all supported versions.
python_versions=("3.10" "3.11" "3.12")
if [[ ! -z ${PYTHON_VERSION+x} ]]; then
  python_versions=("$PYTHON_VERSION")
fi

# Clean previous images.
for python_version in ${python_versions[@]}
do
  docker rmi -f tensorflow_text:${python_version} || true
done

arch=$(uname -m)
build_args=()
if [ "$arch" == "x86_64" ]; then
  build_args+=("--config=release_cpu_linux")
  build_args+=("--platforms=@sigbuild-r2.17-clang_config_platform//:platform")
  auditwheel_platform="manylinux2014_x86_64"
elif [ "$arch" == "aarch64" ]; then
  build_args+=("--crosstool_top=@ml2014_aarch64_config_aarch64//crosstool:toolchain")
  auditwheel_platform="manylinux2014_aarch64"
fi

# Build wheel for each Python version.
for python_version in ${python_versions[@]}
do
  DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
    --build-arg HERMETIC_PYTHON_VERSION=${python_version} --build-arg PYTHON_VERSION=${python_version} \
    -t tensorflow_text:${python_version} - < "oss_scripts/build.Dockerfile.${arch}"

  docker run --rm -a stdin -a stdout -a stderr \
    --env PYTHON_VERSION=${python_version} \
    --env HERMETIC_PYTHON_VERSION=${python_version} \
    --env BUILD_ARGS=${build_args} \
    --env AUDITWHEEL_PLATFORM=${auditwheel_platform} \
    --env IS_NIGHTLY=${IS_NIGHTLY} \
    -v $PWD:/tmp/tensorflow_text \
    --name tensorflow_text tensorflow_text:${python_version} \
    bash oss_scripts/run_build.sh
done
