#!/bin/sh
#
# Copyright 2016 Google Inc. All rights reserved.
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

npm install @types/flatbuffers

../flatc --ts --no-fb-import --gen-mutable -o ts -I include_test monster_test.fbs
../flatc -b -I include_test monster_test.fbs unicode_test.json
tsc --strict --noUnusedParameters --noUnusedLocals --noImplicitReturns --strictNullChecks ts/monster_test_generated.ts
node JavaScriptTest ./ts/monster_test_generated

../flatc --ts --js --no-fb-import -o ts union_vector/union_vector.fbs

# test JS version first, then transpile and rerun for TS
node JavaScriptUnionVectorTest ./ts/union_vector_generated
tsc --strict --noUnusedParameters --noUnusedLocals --noImplicitReturns --strictNullChecks ts/union_vector_generated.ts
node JavaScriptUnionVectorTest ./ts/union_vector_generated

npm uninstall @types/flatbuffers
