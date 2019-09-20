#!/bin/bash
#
# Copyright 2014 Google Inc. All rights reserved.
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

set -o errexit

echo Compile then run the Java test.

java -version

testdir=$(dirname $0)

targetdir="${testdir}/target"

if [[ -e "${targetdir}" ]]; then
    echo "cleaning target"
    rm -rf "${targetdir}"
fi

mkdir -v "${targetdir}"

if ! find "${testdir}/../java" -type f -name "*.class" -delete; then
    echo "failed to clean .class files from java directory" >&2
    exit 1
fi

javac -d "${targetdir}" -classpath "${testdir}/../java:${testdir}:${testdir}/namespace_test:${testdir}/union_vector" "${testdir}/JavaTest.java"

(cd "${testdir}" && java -classpath "${targetdir}" JavaTest )

rm -rf "${targetdir}"
