#!/bin/bash
#
# Copyright 2018 Google Inc. All rights reserved.
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
set -e

# build flatc on debian once to speed up the test loop below
docker build -t build_flatc_debian_stretch -f tests/docker/Dockerfile.testing.build_flatc_debian_stretch .
BUILD_CONTAINER_ID=$(docker create --read-only build_flatc_debian_stretch)
docker cp ${BUILD_CONTAINER_ID}:/code/flatc flatc_debian_stretch

for f in $(ls tests/docker/languages | sort)
do
        # docker pull sometimes fails for unknown reasons, probably travisci-related. this retries the pull we need a few times.
        REQUIRED_BASE_IMAGE=$(cat tests/docker/languages/${f} | head -n 1  | awk ' { print $2 } ')

        set +e
        n=0
        until [ $n -ge 5 ]
        do
           docker pull $REQUIRED_BASE_IMAGE && break
           n=$[$n+1]
           sleep 1
        done
        set -e

        docker build -t $(echo ${f} | cut -f 3- -d .) -f tests/docker/languages/${f} .
        echo "TEST OK: ${f}"
done
