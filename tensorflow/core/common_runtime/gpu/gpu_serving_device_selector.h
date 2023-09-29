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

#include <memory>

#include "absl/container/fixed_array.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {
namespace gpu {

class GpuServingDeviceSelector : public ServingDeviceSelector {
 public:
  GpuServingDeviceSelector(
      int num_devices,
      std::unique_ptr<ServingDeviceSelector::Policy> device_selector_policy);

  DeviceReservation ReserveDevice(
      absl::string_view program_fingerprint) override;

 private:
  void FreeDeviceReservation(const DeviceReservation& reservation) override;

  absl::Mutex mu_;
  absl::FixedArray<DeviceState, 8> device_states_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<ServingDeviceSelector::Policy> device_selector_policy_;
  int64_t req_id_counter_ ABSL_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SERVING_DEVICE_SELECTOR_H_
