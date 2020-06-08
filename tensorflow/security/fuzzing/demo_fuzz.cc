/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdint>
#include <cstdlib>

// This is a demo fuzzer to test that the entire framework functions correctly.
// Once we start moving the existing fuzzers to this framework we will delete
// this.
// TODO(mihaimaruseac): Delete this when no longer needed
void DemoFuzzer(const uint8_t* data, size_t size) {
  // Trigger a small bug that should be found by the fuzzer quite quickly
  if (size > 10 && size % 3 == 2)
    if (data[0] > data[1])
      if (data[5] % data[2] == data[3]) abort();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  DemoFuzzer(data, size);
  return 0;
}
