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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "xla/tsl/framework/serving_device_selector_policies.h"
#include "tensorflow/core/common_runtime/gpu/gpu_scheduling_metrics_storage.h"

namespace tensorflow {
namespace gpu {
class ServingDeviceSelectorTestHelper {
 public:
  ServingDeviceSelectorTestHelper() {
    GpuServingDeviceSelector::OverwriteNowNsFunctionForTest(NowNs);
    now_ns_ = 0;
  }

  ~ServingDeviceSelectorTestHelper() {
    GpuServingDeviceSelector::OverwriteNowNsFunctionForTest(
        absl::GetCurrentTimeNanos);
  }

  static void ElapseNs(int64_t ns) { now_ns_ += ns; }

  static int64_t NowNs() { return now_ns_; }

 private:
  static int64_t now_ns_;
};

int64_t ServingDeviceSelectorTestHelper::now_ns_ = 0;
namespace {

TEST(GpuServingDeviceSelector, Basic) {
  // Create a selector with two devices and round-robin policy.
  GpuServingDeviceSelector selector(/*num_devices=*/2,
                                    std::make_unique<tsl::RoundRobinPolicy>());

  const std::string program_fingerprint = "TensorFlow";
  tsl::DeviceReservation reservation =
      selector.ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), 0);

  reservation = selector.ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), 1);

  reservation = selector.ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), 0);
}

TEST(GpuServingDeviceSelector, DefaultPolicyOnlyEnqueueCall) {
  ServingDeviceSelectorTestHelper helper;
  auto policy = std::make_unique<tsl::RoundRobinPolicy>();
  auto serving_device_selector =
      std::make_unique<tensorflow::gpu::GpuServingDeviceSelector>(
          4, std::move(policy));
  serving_device_selector->Enqueue(3, "16ms");
  serving_device_selector->Enqueue(2, "8ms");
  serving_device_selector->Enqueue(1, "4ms");
  serving_device_selector->Enqueue(0, "2ms");
  // Nothing is completed yet, we don't have any estimated execution time, and
  // we don't know what programs we are enqueueing.
  serving_device_selector->Enqueue(3, "16ms");
  serving_device_selector->Enqueue(2, "8ms");
  serving_device_selector->Enqueue(1, "4ms");
  serving_device_selector->Enqueue(0, "2ms");
  helper.ElapseNs(2e6);
  serving_device_selector->Completed(0, false);
  helper.ElapseNs(2e6);
  serving_device_selector->Completed(0, false);
  serving_device_selector->Completed(1, false);
  helper.ElapseNs(4e6);
  serving_device_selector->Completed(1, false);
  serving_device_selector->Completed(2, false);
  helper.ElapseNs(8e6);
  serving_device_selector->Completed(2, false);
  serving_device_selector->Completed(3, false);
  helper.ElapseNs(16e6);
  serving_device_selector->Completed(3, false);

  serving_device_selector->Enqueue(3, "16ms");
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      16e6);
  serving_device_selector->Enqueue(2, "8ms");
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      24e6);
  serving_device_selector->Enqueue(1, "4ms");
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      28e6);
  serving_device_selector->Enqueue(0, "2ms");
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      30e6);
  helper.ElapseNs(2e6);
  serving_device_selector->Completed(0, false);
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      22e6);
  helper.ElapseNs(2e6);
  serving_device_selector->Completed(1, false);
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      16e6);
  helper.ElapseNs(4e6);
  serving_device_selector->Completed(2, false);
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      8e6);
  helper.ElapseNs(8e6);
  serving_device_selector->Completed(3, false);
  EXPECT_EQ(
      GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Get(),
      0e6);
}

}  // namespace
}  // namespace gpu
}  // namespace tensorflow
