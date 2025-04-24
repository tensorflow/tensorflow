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

#ifndef XLA_TSL_PLATFORM_MOCK_WRITABLE_FILE_H_
#define XLA_TSL_PLATFORM_MOCK_WRITABLE_FILE_H_

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {

class MockWritableFile : public WritableFile {
 public:
  MOCK_METHOD(absl::Status, Append, (absl::string_view data), (override));
  MOCK_METHOD(absl::Status, Close, (), (override));
  MOCK_METHOD(absl::Status, Flush, (), (override));
  MOCK_METHOD(absl::Status, Sync, (), (override));
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_MOCK_WRITABLE_FILE_H_
