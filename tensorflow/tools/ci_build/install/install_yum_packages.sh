#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#

set -e

yum install -y  epel-release \
                centos-release-scl \
                sudo

yum install -y  atlas-devel \
                bzip2-devel \
                curl-devel \
                devtoolset-7 \
                expat-devel \
                gdbm-devel \
                gettext-devel \
                java-1.8.0-openjdk \
                java-1.8.0-openjdk-devel \
                libffi-devel \
                libtool \
                libuuid-devel \
                ncurses-devel \
                openssl-devel \
                patch \
                patchelf \
                perl-core \
                python27 \
                readline-devel \
                sqlite-devel \
                wget \
                xz-devel \
                zlib-devel

# Install latest git.
yum install -y http://opensource.wandisco.com/centos/6/git/x86_64/wandisco-git-release-6-1.noarch.rpm
yum install -y git
