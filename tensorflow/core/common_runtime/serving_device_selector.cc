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

#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {

DeviceReservation::DeviceReservation(int device_index,
                                     ServingDeviceSelector* device_selector)
    : device_index_(device_index), device_selector_(device_selector) {}

DeviceReservation::~DeviceReservation() { reset(); }

void DeviceReservation::reset() {
  if (device_selector_) device_selector_->FreeDeviceReservation(*this);
  device_selector_ = nullptr;
}

DeviceReservation::DeviceReservation(DeviceReservation&& r)
    : device_index_{r.device_index_}, device_selector_{r.device_selector_} {
  r.device_selector_ = nullptr;
}

DeviceReservation& DeviceReservation::operator=(DeviceReservation&& r) {
  if (this == &r) return *this;

  if (device_selector_) device_selector_->FreeDeviceReservation(*this);

  device_index_ = r.device_index_;
  device_selector_ = r.device_selector_;
  r.device_selector_ = nullptr;
  return *this;
}

}  // namespace tensorflow
