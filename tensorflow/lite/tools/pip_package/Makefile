# Values: debian:<version>, ubuntu:<version>
BASE_IMAGE ?= debian:buster
# Values: python, python3
PYTHON ?= python3
# Values: rpi, aarch64, native
TENSORFLOW_TARGET ?= native

ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TENSORFLOW_DIR := $(ROOT_DIR)/../../../..
NAME := $(word 1, $(subst :, ,$(BASE_IMAGE)))
VERSION := $(word 2, $(subst :, ,$(BASE_IMAGE)))
TAG_IMAGE := "tflite-builder-$(BASE_IMAGE)"

.PHONY: help \
        docker-image \
        docker-shell \
        docker-build \
        clean

help:
	@echo "make docker-image -- build docker image"
	@echo "make docker-shell -- run shell inside the docker image"
	@echo "make docker-build -- build wheel inside the docker image"
	@echo "make clean        -- remove built wheel files"

docker-image:
	docker build -t $(TAG_IMAGE) --build-arg VERSION=$(VERSION) -f Dockerfile.$(NAME) .

docker-shell: docker-image
	docker run --rm -it $(TAG_IMAGE)

docker-build: docker-image
	docker run \
		-e "PYTHON=$(PYTHON)" \
		-e "TENSORFLOW_TARGET=$(TENSORFLOW_TARGET)" \
		-v $(TENSORFLOW_DIR):/tensorflow \
		-v $(CURDIR):/out \
		--rm -it $(TAG_IMAGE) \
		/bin/bash -c "bash /tensorflow/tensorflow/lite/tools/pip_package/build_pip_package.sh && cp /tmp/tflite_pip/$(PYTHON)/dist/*.whl /out"

clean:
	rm -f *.whl
