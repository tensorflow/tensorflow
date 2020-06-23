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

#include "tensorflow/core/platform/status.h"

// This is a fuzzer for `tensorflow::Status`. Since `Status` is used almost
// everywhere, we need to ensure that the common functionality is safe. We don't
// expect many crashes from this fuzzer since we only create a status and then
// look at the error message from it but this is a good test of the fuzzing
// infrastructure, with minimal dependencies (thus, it is a good test to weed
// out linker bloat and other linker issues).

namespace {

tensorflow::error::Code BuildRandomErrorCode(uint32_t code) {
  // We cannot build a `Status` with error_code of 0 and a message, so force
  // error code to be non-zero.
  if (code == 0) {
    return tensorflow::error::UNKNOWN;
  }

  return static_cast<tensorflow::error::Code>(code);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  tensorflow::error::Code error_code;
  std::string error_message;

  FuzzedDataProvider fuzzed_data(data, size);

  uint32_t code = fuzzed_data.ConsumeIntegral<uint32_t>();
  error_code = BuildRandomErrorCode(code);

  error_message = fuzzed_data.ConsumeRemainingBytesAsString();

  tensorflow::Status s = tensorflow::Status(error_code, error_message);
  const std::string actual_message = s.ToString();
  const std::size_t pos = actual_message.rfind(error_message);
  assert(pos != std::string::npos);  // Suffix is error message
  assert(pos > 0);                   // Prefix is error code

  // In some build configurations `assert` is a no-op. This causes `pos` to be
  // unused and then produces an error if also compiling with `-Werror`.
  (void)pos;

  return 0;
}

}  // namespace
