/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_OP_HANDLER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_OP_HANDLER_H_

#include <memory>
#include <unordered_map>

#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/lib/core/status.h"
namespace tensorflow {

// TODO(tfrt-devs): Figure out a way to unify it with OpHandler in TFRT.
class CustomDeviceOpHandler {
 public:
  ~CustomDeviceOpHandler() = default;
  // Register a new custom device.
  Status RegisterCustomDevice(const string& device_name,
                              std::unique_ptr<CustomDevice> device);

  // Find the custom device from given name. Return true if it finds one.
  bool FindCustomDeviceFromName(const string& name,
                                CustomDevice** device) const;

  Status Execute(ImmediateExecutionOperation* op,
                 ImmediateExecutionTensorHandle** retvals, int* num_retvals);

  ImmediateExecutionTensorHandle* CopyTensorHandleToDevice(
      ImmediateExecutionContext* context,
      ImmediateExecutionTensorHandle* handle, const char* device_name,
      Status* status);

  // Determine whether to place an op on a custom device. This method is
  // exposed as public for test only.
  Status MaybePinToCustomDevice(CustomDevice** device,
                                const ImmediateExecutionOperation& op) const;

  void Clear();

 private:
  std::unordered_map<string, std::unique_ptr<CustomDevice>> custom_devices_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_OP_HANDLER_H_
