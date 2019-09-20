@echo off
rem Copyright 2018 Google Inc. All rights reserved.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem Compile then run the Rust test.

rem TODO(rw): how do we make this script abort the calling script in appveyor?

cd rust_usage_test
cargo test -- --quiet || exit /b 1
cargo run --bin=alloc_check || exit /b 1
cd ..
