# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Values: debian:<version>, ubuntu:<version>
BASE_IMAGE ?= ubuntu:20.04
PYTHON_VERSION ?= 3.11
NUMPY_VERSION ?= 1.23.2

# Values: rpi, aarch64, native
TENSORFLOW_TARGET ?= native
WHEEL_PROJECT_NAME ?= tflite_runtime
# Values: according to https://www.python.org/dev/peps/pep-0440/
VERSION_SUFFIX ?=

MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TENSORFLOW_DIR := $(MAKEFILE_DIR)/../../../..
TAG_IMAGE := "tflite-runtime-builder-$(subst :,-,$(BASE_IMAGE))"

DOCKER_PARAMS := --pid=host \
    --env "CI_BUILD_USER=$(shell id -u -n)" \
    --env "CI_BUILD_UID=$(shell id -u)" \
    --env "CI_BUILD_GROUP=$(shell id -g -n)" \
    --env "CI_BUILD_GID=$(shell id -g)" \
    --env "CI_BUILD_HOME=$(TENSORFLOW_DIR)/bazel-ci_build-cache" \
    --env "WHEEL_PROJECT_NAME=$(WHEEL_PROJECT_NAME)" \
    --env "VERSION_SUFFIX=$(VERSION_SUFFIX)" \
    --volume $(TENSORFLOW_DIR):/tensorflow \
    --workdir /tensorflow

ifneq ($(WHEEL_PLATFORM_NAME),)
	DOCKER_PARAMS += --env WHEEL_PLATFORM_NAME=$(WHEEL_PLATFORM_NAME)
endif

.PHONY: help \
        docker-image \
        docker-shell \
        docker-build \
        clean

help:
	@echo "make docker-image -- build docker image"
	@echo "make docker-shell -- run shell inside the docker image"
	@echo "make docker-build -- build wheel and deb inside the docker image"
	@echo "make clean        -- remove wheel and deb files"

docker-image:
	docker build -t $(TAG_IMAGE) \
		--build-arg IMAGE=$(BASE_IMAGE) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--build-arg NUMPY_VERSION=$(NUMPY_VERSION) \
		-f $(MAKEFILE_DIR)/Dockerfile.py3 $(MAKEFILE_DIR)/.

docker-shell: docker-image
	mkdir -p $(TENSORFLOW_DIR)/bazel-ci_build-cache
	docker run --rm --interactive --tty \
		$(DOCKER_PARAMS) \
		$(TAG_IMAGE) /with_the_same_user /bin/bash

docker-build: docker-image
	mkdir -p $(TENSORFLOW_DIR)/bazel-ci_build-cache
	docker run \
		--rm --interactive $(shell tty -s && echo --tty) \
		$(DOCKER_PARAMS) \
		$(TAG_IMAGE) \
		/with_the_same_user /bin/bash -C /tensorflow/tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh $(TENSORFLOW_TARGET)
