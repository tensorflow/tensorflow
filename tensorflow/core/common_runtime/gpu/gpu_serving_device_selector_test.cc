/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/gpu/gpu_serving_device_selector.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/serving_device_selector.h"
#include "tensorflow/core/common_runtime/serving_device_selector_policies.h"

namespace tensorflow {
namespace gpu {
namespace {

TEST(GpuServingDeviceSelector, Basic) {
  // Create a selector with two devices and round-robin policy.
  GpuServingDeviceSelector selector(/*num_devices=*/2,
                                    std::make_unique<RoundRobinPolicy>());

  const std::string program_fingerprint = "TensorFlow";
  DeviceReservation reservation = selector.ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), 0);

  reservation = selector.ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), 1);

  reservation = selector.ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace tensorflow
