#!/bin/sh
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
#
# Note to pub consumers: this file is used to assist with publishing the
# pub package from the flatbuffers repository and is not meant for general use.
# As pub does not currently provide a way to exclude files, it is included here.

command -v pub >/dev/null 2>&1 || { echo >&2 "Require `pub` but it's not installed.  Aborting."; exit 1; }

cp ../samples/monster.fbs example/
cp ../tests/monster_test.fbs test/
pub publish

rm example/monster.fbs
rm test/monster_test.fbs