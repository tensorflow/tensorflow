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
# Note: This script runs on Mac and Linux. It requires `go` to be installed
# and 'flatc' to be built (using `cmake` in the root directory).

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
  ../flatc --go monster.fbs
elif [ -e ../Debug/flatc ]; then
  ../Debug/flatc --go monster.fbs
else
  echo 'flatc' could not be found. Make sure to build FlatBuffers from the \
       $rootdir directory.
 exit 1
fi

echo Compiling and running the Go sample.

# Go requires a particular layout of files in order to link the necessary
# packages. Copy these files to the respective directores to compile the
# sample.
mkdir -p ${sampledir}/go_gen/src/MyGame/Sample
mkdir -p ${sampledir}/go_gen/src/github.com/google/flatbuffers/go
cp MyGame/Sample/*.go ${sampledir}/go_gen/src/MyGame/Sample/
cp ${sampledir}/../go/* ${sampledir}/go_gen/src/github.com/google/flatbuffers/go

# Export the `GOPATH`, so that `go` will know which directories to search for
# the libraries.
export GOPATH=${sampledir}/go_gen/

# Compile and execute the sample.
go build -o go_sample sample_binary.go
./go_sample

# Clean up the temporary files.
rm -rf MyGame/
rm -rf ${sampledir}/go_gen/
rm go_sample
