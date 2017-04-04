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
	zip \
	zlib1g-dev \
	unzip

echo 'oracle-java8-installer shared/accepted-oracle-license-v1-1 select true' | debconf-set-selections
add-apt-repository --yes ppa:webupd8team/java && apt-get -qq update
DEBIAN_FRONTEND=noninteractive apt-get -qq -y --force-yes --no-install-recommends install oracle-java8-installer
