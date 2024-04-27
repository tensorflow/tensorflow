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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tsl/framework/serving_device_selector.h"
#include "tsl/framework/test_util/mock_serving_device_selector.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

class IfrtServingCoreSelectorTest : public ::testing::Test {
 protected:
  explicit IfrtServingCoreSelectorTest() {
    core_selector_ =
        std::make_unique<IfrtServingCoreSelector>(&serving_device_selector_);
  }

  tsl::test_util::MockServingDeviceSelector serving_device_selector_;
  std::unique_ptr<IfrtServingCoreSelector> core_selector_;
};

TEST_F(IfrtServingCoreSelectorTest, ReservedDevicesReturns) {
  int64_t program_id1 = 111111;
  EXPECT_CALL(serving_device_selector_,
              ReserveDevice(absl::StrCat(program_id1)))
      .WillOnce([this](::testing::Unused) {
        return tsl::DeviceReservation(0, &serving_device_selector_);
      });
  tsl::DeviceReservation reservation =
      core_selector_->ReserveDevice(program_id1);
  EXPECT_THAT(reservation.device_index(), 0);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
