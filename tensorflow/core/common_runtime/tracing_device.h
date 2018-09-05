/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_TRACING_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_TRACING_DEVICE_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

namespace test {
class Benchmark;
}
struct SessionOptions;

// This class implements tracing functionality that is shared by its subclasses
// (including ThreadPoolDevice and XlaDevice).
class TracingDevice : public Device {
 public:
  TracingDevice(Env* env, const DeviceAttributes& attributes)
      : Device(env, attributes) {}

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override {
    if (TF_PREDICT_FALSE(
            tracing::GetTraceCollector() ||
            tracing::GetEventCollector(tracing::EventCategory::kCompute))) {
      const string& op_name = op_kernel->name();
      tracing::ScopedActivity activity(op_name, op_kernel->type_string(),
                                       op_kernel->IsExpensive());
      tracing::ScopedRegion region(tracing::EventCategory::kCompute, op_name);
      op_kernel->Compute(context);
    } else {
      op_kernel->Compute(context);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TracingDevice);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_TRACING_DEVICE_H_
