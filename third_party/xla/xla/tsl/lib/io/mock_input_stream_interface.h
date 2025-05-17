/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TSL_LIB_IO_MOCK_INPUT_STREAM_INTERFACE_H_
#define XLA_TSL_LIB_IO_MOCK_INPUT_STREAM_INTERFACE_H_

#include <cstdint>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "xla/tsl/lib/io/inputstream_interface.h"
#include "tsl/platform/tstring.h"

namespace tsl::io {

class MockInputStreamInterface : public InputStreamInterface {
 public:
  MOCK_METHOD(absl::Status, ReadNBytes, (int64_t, tstring*), (override));
  MOCK_METHOD(absl::Status, SkipNBytes, (int64_t bytes_to_skip), (override));
  MOCK_METHOD(int64_t, Tell, (), (const, override));
  MOCK_METHOD(absl::Status, Reset, (), (override));
};

}  // namespace tsl::io

#endif  // XLA_TSL_LIB_IO_MOCK_INPUT_STREAM_INTERFACE_H_
