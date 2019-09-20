#!/bin/sh

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

echo Compile then run the Kotlin test.

testdir=$(dirname $0)
targetdir="${testdir}/kotlin"

if [[ -e "${targetdir}" ]]; then
    echo "cleaning target"
    rm -rf "${targetdir}"
fi

mkdir -v "${targetdir}"

if ! find "${testdir}/../java" -type f -name "*.class" -delete; then
    echo "failed to clean .class files from java directory" >&2
    exit 1
fi

all_kt_files=`find . -name "*.kt" -print`

# Compile java FlatBuffer library 
javac ${testdir}/../java/com/google/flatbuffers/*.java -d $targetdir
# Compile Kotlin files
kotlinc $all_kt_files -classpath $targetdir -include-runtime -d $targetdir
# Make jar
jar cvf ${testdir}/kotlin_test.jar -C $targetdir . > /dev/null
# Run test
kotlin -cp ${testdir}/kotlin_test.jar KotlinTest
# clean up
rm -rf $targetdir
rm ${testdir}/kotlin_test.jar
