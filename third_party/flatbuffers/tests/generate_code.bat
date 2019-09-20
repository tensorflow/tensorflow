:: Copyright 2015 Google Inc. All rights reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

set buildtype=Release
if "%1"=="-b" set buildtype=%2

..\%buildtype%\flatc.exe --cpp --java --csharp --dart --go --binary --lobster --lua --python --js --ts --php --rust --grpc --gen-mutable --reflect-names --gen-object-api --gen-compare --no-includes --cpp-ptr-type flatbuffers::unique_ptr  --no-fb-import -I include_test monster_test.fbs monsterdata_test.json || goto FAIL
..\%buildtype%\flatc.exe --cpp --java --csharp --dart --go --binary --lobster --lua --python --js --ts --php --rust --gen-mutable --reflect-names --no-fb-import --cpp-ptr-type flatbuffers::unique_ptr  -o namespace_test namespace_test/namespace_test1.fbs namespace_test/namespace_test2.fbs || goto FAIL
..\%buildtype%\flatc.exe --cpp --java --csharp --js --ts --php --gen-mutable --reflect-names --gen-object-api --gen-compare --cpp-ptr-type flatbuffers::unique_ptr -o union_vector ./union_vector/union_vector.fbs || goto FAIL
..\%buildtype%\flatc.exe -b --schema --bfbs-comments --bfbs-builtins -I include_test monster_test.fbs || goto FAIL
..\%buildtype%\flatc.exe -b --schema --bfbs-comments --bfbs-builtins -I include_test arrays_test.fbs || goto FAIL
..\%buildtype%\flatc.exe --jsonschema --schema -I include_test monster_test.fbs || goto FAIL
..\%buildtype%\flatc.exe --cpp --java --csharp --python --gen-mutable --reflect-names --gen-object-api --gen-compare --no-includes --scoped-enums --jsonschema --cpp-ptr-type flatbuffers::unique_ptr arrays_test.fbs || goto FAIL
..\%buildtype%\flatc.exe --cpp --gen-mutable --gen-object-api --reflect-names --cpp-ptr-type flatbuffers::unique_ptr native_type_test.fbs || goto FAIL

IF NOT "%MONSTER_EXTRA%"=="skip" (
  @echo Generate MosterExtra
  ..\%buildtype%\flatc.exe --cpp --java --csharp --python --gen-mutable --reflect-names --gen-object-api --gen-compare --no-includes monster_extra.fbs monsterdata_extra.json || goto FAIL
) else (
  @echo monster_extra.fbs skipped (the strtod function from MSVC2013 or older doesn't support NaN/Inf arguments)
)

cd ../samples
..\%buildtype%\flatc.exe --cpp --lobster --gen-mutable --reflect-names --gen-object-api --gen-compare --cpp-ptr-type flatbuffers::unique_ptr monster.fbs || goto FAIL
..\%buildtype%\flatc.exe -b --schema --bfbs-comments --bfbs-builtins monster.fbs || goto FAIL
cd ../reflection
call generate_code.bat %1 %2 || goto FAIL

set EXITCODE=0
goto SUCCESS
:FAIL
set EXITCODE=1
:SUCCESS
cd ../tests
EXIT /B %EXITCODE%
