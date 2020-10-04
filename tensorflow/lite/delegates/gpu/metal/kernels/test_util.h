/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_

#include <map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

class SingleOpModel {
 public:
  SingleOpModel() = delete;
  SingleOpModel(Operation&& operation,
                const std::vector<TensorRef<BHWC>>& inputs,
                const std::vector<TensorRef<BHWC>>& outputs);
  virtual ~SingleOpModel() = default;

  bool PopulateTensor(int index, std::vector<float>&& data) {
    inputs_[index].data = data;
    return true;
  }

  absl::Status Invoke();

  const std::vector<float>& GetOutput(int index) const {
    return outputs_[index].data;
  }

 protected:
  GraphFloat32 graph_;
  std::vector<TensorFloat32> inputs_;
  std::vector<TensorFloat32> outputs_;
};

absl::Status CompareVectors(const std::vector<float>& reference,
                            const std::vector<float>& output, float max_error);

/// Helper function that compiles previously configured graph (with added
/// tasks), initializes graph with specified inputs, invokes and fills specified
/// outputs
absl::Status RunGraph(const std::vector<ComputeTaskDescriptorPtr>& graph,
                      id<MTLDevice> device,
                      const std::map<ValueId, TensorFloat32>& inputs,
                      std::map<ValueId, TensorFloat32>* outputs);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_TEST_UTIL_H_
