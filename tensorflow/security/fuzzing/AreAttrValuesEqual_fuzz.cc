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

#include "absl/strings/match.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"

// This is a fuzzer for AreAttrValuesEqual.

namespace {

// A few helpers to construct AttrValue protos.
template <typename T>
tensorflow::AttrValue createAttrValue(T value) {
  tensorflow::AttrValue ret;
  SetAttrValue(value, &ret);
  return ret;
}

// A helper to do the comparison asserts.
template <typename T>
void compareValues(T value, T value_2) {
  const tensorflow::AttrValue proto = createAttrValue(value);
  const tensorflow::AttrValue proto_same = createAttrValue(value);
  const tensorflow::AttrValue proto2 = createAttrValue(value_2);

  // Assert that AreAttrValuesEqual are same with or without allow false
  // negatives.
  assert(tensorflow::AreAttrValuesEqual(proto, proto_same,
                                        /*allow_false_negatives=*/false));
  assert(tensorflow::AreAttrValuesEqual(proto, proto_same,
                                        /*allow_false_negatives=*/true));
  // Assert that AreAttrValuesEqual are same with or without allow false
  // negatives.
  assert(tensorflow::AreAttrValuesEqual(proto, proto2,
                                        /*allow_false_negatives=*/false) ==
         tensorflow::AreAttrValuesEqual(proto, proto2,
                                        /*allow_false_negatives=*/true));
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);

  // Choose random integers.
  const int random_int = fuzzed_data.ConsumeIntegralInRange(1, 100);
  const int random_int2 = fuzzed_data.ConsumeIntegralInRange(1, 1000);
  compareValues(random_int, random_int2);

  // Choose random floats.
  const float random_float =
      fuzzed_data.ConsumeFloatingPointInRange(1.0f, 1000.0f);
  const float random_float2 =
      fuzzed_data.ConsumeFloatingPointInRange(1.0f, 1000.0f);
  compareValues(random_float, random_float2);

  // Choose random strings.
  const int content_size = fuzzed_data.ConsumeIntegralInRange(10, 300);
  const std::string test_string =
      fuzzed_data.ConsumeRandomLengthString(content_size);
  const std::string test_string2 = fuzzed_data.ConsumeRemainingBytesAsString();
  compareValues(test_string, test_string2);

  return 0;
}

}  // namespace
