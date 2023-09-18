#!/bin/sh

set -e

export LANG=C

release_tag=2.15

docker build --pull \
        --tag=linaro/tensorflow-arm64-build:latest-multipython \
        --tag=linaro/tensorflow-arm64-build:${release_tag}-multipython .
mkdir -p tagdir-multipython
echo linaro/tensorflow-arm64-build:latest-multipython > tagdir-multipython/.docker-tag
mkdir -p tagdir-${release_tag}-multipython
echo linaro/tensorflow-arm64-build:${release_tag}-multipython > tagdir-${release_tag}-multipython/.docker-tag
