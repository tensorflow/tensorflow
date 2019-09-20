#!/bin/bash -eu
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

pushd "$(dirname $0)" >/dev/null
test_dir="$(pwd)"
go_path=${test_dir}/go_gen
go_src=${go_path}/src

# Emit Go code for the example schema in the test dir:
../flatc -g -I include_test monster_test.fbs

# Go requires a particular layout of files in order to link multiple packages.
# Copy flatbuffer Go files to their own package directories to compile the
# test binary:
mkdir -p ${go_src}/MyGame/Example
mkdir -p ${go_src}/MyGame/Example2
mkdir -p ${go_src}/github.com/google/flatbuffers/go
mkdir -p ${go_src}/flatbuffers_test

cp -a MyGame/*.go ./go_gen/src/MyGame/
cp -a MyGame/Example/*.go ./go_gen/src/MyGame/Example/
cp -a MyGame/Example2/*.go ./go_gen/src/MyGame/Example2/
# do not compile the gRPC generated files, which are not tested by go_test.go
# below, but have their own test.
rm ./go_gen/src/MyGame/Example/*_grpc.go
cp -a ../go/* ./go_gen/src/github.com/google/flatbuffers/go
cp -a ./go_test.go ./go_gen/src/flatbuffers_test/

# Run tests with necessary flags.
# Developers may wish to see more detail by appending the verbosity flag
# -test.v to arguments for this command, as in:
#   go -test -test.v ...
# Developers may also wish to run benchmarks, which may be achieved with the
# flag -test.bench and the wildcard regexp ".":
#   go -test -test.bench=. ...
GOPATH=${go_path} go test flatbuffers_test \
                     --test.coverpkg=github.com/google/flatbuffers/go \
                     --cpp_data=${test_dir}/monsterdata_test.mon \
                     --out_data=${test_dir}/monsterdata_go_wire.mon \
                     --test.bench=. \
                     --test.benchtime=3s \
                     --fuzz=true \
                     --fuzz_fields=4 \
                     --fuzz_objects=10000

GO_TEST_RESULT=$?
rm -rf ${go_path}/{pkg,src}
if [[ $GO_TEST_RESULT  == 0 ]]; then
    echo "OK: Go tests passed."
else
    echo "KO: Go tests failed."
    exit 1
fi

NOT_FMT_FILES=$(gofmt -l MyGame)
if [[ ${NOT_FMT_FILES} != "" ]]; then
    echo "These files are not well gofmt'ed:"
    echo
    echo "${NOT_FMT_FILES}"
    # enable this when enums are properly formated
    # exit 1
fi
