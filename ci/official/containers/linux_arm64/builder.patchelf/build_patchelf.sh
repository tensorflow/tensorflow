#!/bin/bash -eu

# Need a newer version of patchelf as the installed version is buggy in 20.04
# so get patchelf source from 22.04 ie 'jammy' and build it to avoid dependency
# problems that would occur with a binary package

mkdir -p /patchelf
cd /patchelf
echo deb-src http://ports.ubuntu.com/ubuntu-ports/ jammy universe>>/etc/apt/sources.list
apt-get update
apt-get -y build-dep patchelf/jammy
apt-get -b source patchelf/jammy

# This will leave a .deb file for installation in a later stage
