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

#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"

// This is a fuzzer for tensorflow::str_util::StringReplace

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);

  if (size < 1) return 0;

  bool all_flag = fuzzed_data.ConsumeBool();

  std::string s = fuzzed_data.ConsumeRandomLengthString(10);
  std::string oldsub = fuzzed_data.ConsumeRandomLengthString(5);
  std::string newsub = fuzzed_data.ConsumeRemainingBytesAsString();

  tensorflow::StringPiece sp(s);
  tensorflow::StringPiece oldsubp(oldsub);
  tensorflow::StringPiece newsubp(newsub);

  std::string subbed =
      tensorflow::str_util::StringReplace(sp, oldsubp, newsubp, all_flag);

  return 0;
}

}  // namespace
