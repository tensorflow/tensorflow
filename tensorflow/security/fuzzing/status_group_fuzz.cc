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

// This is a fuzzer for `tensorflow::StatusGroup`. Since `Status` is used almost
// everywhere, we need to ensure that the common functionality is safe. We don't
// expect many crashes from this fuzzer

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
  const std::string error_message = "ERROR";
  tensorflow::StatusGroup sg;
  FuzzedDataProvider fuzzed_data(data, size);

  while (fuzzed_data.remaining_bytes() > 0) {
    uint32_t code = fuzzed_data.ConsumeIntegral<uint32_t>();
    tensorflow::error::Code error_code = BuildRandomErrorCode(code);
    bool is_derived = fuzzed_data.ConsumeBool();

    tensorflow::Status s = tensorflow::Status(error_code, error_message);

    if (is_derived) {
      tensorflow::Status derived_s = tensorflow::StatusGroup::MakeDerived(s);
      sg.Update(derived_s);
    } else {
      sg.Update(s);
    }
  }

  // Ignore warnings that these values are unused
  sg.as_summary_status().IgnoreError();
  sg.as_concatenated_status().IgnoreError();
  sg.AttachLogMessages();

  return 0;
}

}  // namespace
