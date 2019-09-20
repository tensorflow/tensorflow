#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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
#
# Note: This script runs on Mac and Linux. It requires `kotlin` to be installed
# and `flatc` to be built (using `cmake` in the root directory).

sampledir=$(cd $(dirname $BASH_SOURCE) && pwd)
rootdir=$(cd $sampledir/.. && pwd)
currentdir=$(pwd)

if [[ "$sampledir" != "$currentdir" ]]; then
  echo Error: This script must be run from inside the $sampledir directory.
  echo You executed it from the $currentdir directory.
  exit 1
fi

# Run `flatc`. Note: This requires you to compile using `cmake` from the
# root `/flatbuffers` directory.
if [ -e ../flatc ]; then
  echo "compiling now"
  ../flatc --kotlin --gen-mutable monster.fbs
elif [ -e ../Debug/flatc ]; then
  ../Debug/flatc --kotlin --gen-mutable monster.fbs
else
  echo 'flatc' could not be found. Make sure to build FlatBuffers from the \
       $rootdir directory.
  exit 1
fi

echo Compiling and running the Kotlin sample

all_kt_files=`find $sampledir -name "*.kt" -print`
# Run test
mkdir -v "${sampledir}/kotlin"
echo Compiling Java Runtime
javac ${rootdir}/java/com/google/flatbuffers/*.java -d ${sampledir}/kotlin
echo Compiling Kotlin source
kotlinc -classpath ${sampledir}/../java:${sampledir}/kotlin $all_kt_files SampleBinary.kt -include-runtime -d ${sampledir}/kotlin
# Make jar
echo Making jar
jar cvf ${sampledir}/kotlin/kotlin_sample.jar -C ${sampledir}/kotlin . > /dev/null
echo Running test
kotlin -cp ${sampledir}/kotlin/kotlin_sample.jar SampleBinary

# Cleanup temporary files.
rm -rf MyGame/
# rm -rf ${sampledir}/kotlin
