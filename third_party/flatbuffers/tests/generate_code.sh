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
set -e

../flatc --cpp --java --kotlin  --csharp --dart --go --binary --lobster --lua --python --js --ts --php --rust --grpc --gen-mutable --reflect-names --gen-object-api --gen-compare --no-includes --cpp-ptr-type flatbuffers::unique_ptr  --no-fb-import -I include_test monster_test.fbs monsterdata_test.json
../flatc --cpp --java --kotlin --csharp --dart --go --binary --lobster --lua --python --js --ts --php --rust --gen-mutable --reflect-names --no-fb-import --cpp-ptr-type flatbuffers::unique_ptr  -o namespace_test namespace_test/namespace_test1.fbs namespace_test/namespace_test2.fbs
../flatc --cpp --java --kotlin --csharp --js --ts --php --gen-mutable --reflect-names --gen-object-api --gen-compare --cpp-ptr-type flatbuffers::unique_ptr -o union_vector ./union_vector/union_vector.fbs
../flatc -b --schema --bfbs-comments --bfbs-builtins -I include_test monster_test.fbs
../flatc -b --schema --bfbs-comments --bfbs-builtins -I include_test arrays_test.fbs
../flatc --jsonschema --schema -I include_test monster_test.fbs
../flatc --cpp --java --kotlin --csharp --python --gen-mutable --reflect-names --gen-object-api --gen-compare --no-includes monster_extra.fbs monsterdata_extra.json
../flatc --cpp --java --csharp --python --gen-mutable --reflect-names --gen-object-api --gen-compare --no-includes --scoped-enums --jsonschema --cpp-ptr-type flatbuffers::unique_ptr arrays_test.fbs
cd ../samples
../flatc --cpp --lobster --gen-mutable --reflect-names --gen-object-api --gen-compare --cpp-ptr-type flatbuffers::unique_ptr monster.fbs
../flatc -b --schema --bfbs-comments --bfbs-builtins monster.fbs
cd ../reflection
./generate_code.sh
