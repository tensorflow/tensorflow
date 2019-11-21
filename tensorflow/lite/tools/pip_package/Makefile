# Values: debian:<version>, ubuntu:<version>
BASE_IMAGE ?= debian:buster
# Values: python, python3
PYTHON ?= python3
# Values: rpi, aarch64, native
TENSORFLOW_TARGET ?= native
# Values: n, y
BUILD_DEB ?= n
# Values: according to https://www.python.org/dev/peps/pep-0440/
VERSION_SUFFIX ?=

MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TENSORFLOW_DIR := $(MAKEFILE_DIR)/../../../..
OUT_DIR := $(CURDIR)/out/$(PYTHON)/$(subst :,-,$(BASE_IMAGE))
TAG_IMAGE := "tflite-runtime-builder-$(subst :,-,$(BASE_IMAGE))"

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
	docker build -t $(TAG_IMAGE) --build-arg IMAGE=$(BASE_IMAGE) .

docker-shell: docker-image
	docker run --rm --interactive --tty \
		--volume $(TENSORFLOW_DIR):/tensorflow \
		--workdir /tensorflow \
		$(TAG_IMAGE)

docker-build: docker-image
	mkdir -p $(OUT_DIR)
	docker run --user $(shell id -u):$(shell id -g) \
		--rm --interactive $(shell tty -s && echo --tty) \
		--env "PYTHON=$(PYTHON)" \
		--env "TENSORFLOW_TARGET=$(TENSORFLOW_TARGET)" \
		--env "BUILD_DEB=$(BUILD_DEB)" \
		--env "VERSION_SUFFIX=$(VERSION_SUFFIX)" \
		--volume $(TENSORFLOW_DIR):/tensorflow \
		--volume $(OUT_DIR):/out \
		$(TAG_IMAGE) \
		/bin/bash -c "tensorflow/tensorflow/lite/tools/pip_package/build_pip_package.sh && \
		              (cp /tmp/tflite_pip/*.deb /tmp/tflite_pip/$(PYTHON)/dist/{*.whl,*.tar.gz} /out 2>/dev/null || true)"

clean:
	rm -rf $(CURDIR)/out

