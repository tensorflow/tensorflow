#!/bin/bash

set -ex

apt-get -qq update

apt-get -qq -y install \
	realpath \
	git \
	build-essential \
	python-virtualenv \
	python-dev \
	libblas-dev \
	liblapack-dev \
	gfortran \
	swig \
	wget \
	software-properties-common \
	pixz \
	zip \
	zlib1g-dev \
	unzip

add-apt-repository --yes ppa:openjdk-r/ppa && apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq -y --force-yes install openjdk-8-jdk
