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
#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <cstdlib>

#include "tensorflow/core/platform/tstring.h"

// This is a fuzzer for tensorflow::tstring

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);

  tensorflow::tstring base = fuzzed_data.ConsumeRandomLengthString(10);

  while (fuzzed_data.remaining_bytes() > 0) {
    tensorflow::tstring pair = fuzzed_data.ConsumeRandomLengthString(10);
    base.append(pair);
    assert(base.size() <= base.capacity());
  }

  return 0;
}

}  // namespace
