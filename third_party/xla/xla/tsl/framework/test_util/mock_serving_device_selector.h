/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_FRAMEWORK_TEST_UTIL_MOCK_SERVING_DEVICE_SELECTOR_H_
#define XLA_TSL_FRAMEWORK_TEST_UTIL_MOCK_SERVING_DEVICE_SELECTOR_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "tsl/platform/test.h"
namespace tsl {

namespace test_util {

class MockServingDeviceSelector : public tsl::ServingDeviceSelector {
 public:
  MOCK_METHOD(tsl::DeviceReservation, ReserveDevice, (absl::string_view),
              (override));
  MOCK_METHOD(void, Enqueue, (int32_t, absl::string_view), (override));
  MOCK_METHOD(void, Completed, (int32_t, bool), (override));
  MOCK_METHOD(void, FreeDeviceReservation, (const tsl::DeviceReservation&),
              (override));
};

}  // namespace test_util
}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_TEST_UTIL_MOCK_SERVING_DEVICE_SELECTOR_H_
