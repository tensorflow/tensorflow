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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SERVING_DEVICE_SELECTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SERVING_DEVICE_SELECTOR_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "tensorflow/core/framework/resource_base.h"

namespace tensorflow {
namespace gpu {
class GpuServingDeviceSelector;
const char kGpuServingDeviceSelectorResourceName[] =
    "gpu_serving_device_selector";

class GpuServingDeviceSelectorResource : public ResourceBase {
 public:
  explicit GpuServingDeviceSelectorResource(
      int num_devices, std::unique_ptr<tsl::ServingDeviceSelector::Policy>
                           device_selector_policy)
      : selector_(std::make_unique<GpuServingDeviceSelector>(
            num_devices, std::move(device_selector_policy))) {}

  std::string DebugString() const override {
    return "GpuServingDeviceSelectorResource";
  };

  GpuServingDeviceSelector* selector() const { return selector_.get(); }

 private:
  std::unique_ptr<GpuServingDeviceSelector> selector_;
};

class GpuServingDeviceSelector : public tsl::ServingDeviceSelector {
 public:
  GpuServingDeviceSelector(
      int num_devices,
      std::unique_ptr<ServingDeviceSelector::Policy> device_selector_policy);

  tsl::DeviceReservation ReserveDevice(
      absl::string_view program_fingerprint) override;

  // Enqueues the program on the stream of index `index_on_host`.
  void Enqueue(int32_t index_on_host, absl::string_view fingerprint) override;

  // Marks the completion of a program on the given stream.
  // If `had_error` is true, this function doesn't update program's execution
  // time stats to avoid incorrect estimates.
  void Completed(int32_t index_on_host, bool had_error) override;

 private:
  friend class ServingDeviceSelectorTestHelper;
  static void OverwriteNowNsFunctionForTest(int64_t (*now_ns)());

  void FreeDeviceReservation(
      const tsl::DeviceReservation& reservation) override;

  // Only for metrics reporting purposes.
  int64_t TotalEstimatedTimeTillIdleNs() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex mu_;
  absl::FixedArray<DeviceState, 8> device_states_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<ServingDeviceSelector::Policy> device_selector_policy_;
  int64_t req_id_counter_ ABSL_GUARDED_BY(mu_);
  // Map from program fingerprint to execution info.
  absl::node_hash_map<std::string, ExecutionInfo> execution_info_
      ABSL_GUARDED_BY(mu_);
  std::optional<int64_t> min_exec_time_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SERVING_DEVICE_SELECTOR_H_
