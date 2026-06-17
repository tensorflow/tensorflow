/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_QCOM_THIN_FILTER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_QCOM_THIN_FILTER_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/common/task/qcom_thin_filter_desc.h"

namespace tflite {
namespace gpu {
namespace cl {

class QcomThinFilter : public GPUObject {
 public:
  QcomThinFilter() {}
  explicit QcomThinFilter(cl_mem filter) : filter_(filter) {}
  ~QcomThinFilter() override;

  // Move only
  QcomThinFilter(QcomThinFilter&& filter);
  QcomThinFilter& operator=(QcomThinFilter&& filter);
  QcomThinFilter(const QcomThinFilter&) = delete;
  QcomThinFilter& operator=(const QcomThinFilter&) = delete;

  cl_mem GetMemoryPtr() const { return filter_; }

  absl::Status GetGPUResources(const GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  absl::Status CreateFromDescriptor(const QcomThinFilterDescriptor& desc,
                                    CLContext* context);

 private:
  void Release();
  cl_mem filter_ = nullptr;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_QCOM_THIN_FILTER_H_
