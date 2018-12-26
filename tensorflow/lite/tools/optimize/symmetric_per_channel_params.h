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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_SYMMETRIC_PER_CHANNEL_PARAMS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_SYMMETRIC_PER_CHANNEL_PARAMS_H_

#include <memory>
#include <vector>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace internal {

// Helper to read/write symmetric per channel parameters to tensors.
//
// TODO(shashishekhar): This maybe useful for kernels as well, refactor to
// kernel utils.
class SymmetricPerChannelParams {
 public:
  SymmetricPerChannelParams(const std::vector<float> scales,
                            int channel_dim_index)
      : scales_(scales), channel_dim_index_(channel_dim_index) {}

  // Creates a new instance by reading the information from existing tensor.
  static TfLiteStatus ReadFromTensor(
      const TensorT& tensor,
      std::unique_ptr<SymmetricPerChannelParams>* params);

  // Add the symmetric per channel info to the tensor.
  TfLiteStatus AddToTensor(TensorT* tensor) const;

  const std::vector<float>& scales() const { return scales_; }

  int channel_dim_index() const { return channel_dim_index_; }

 private:
  std::vector<float> scales_;
  int channel_dim_index_;
};
}  // namespace internal
}  // namespace optimize
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_SYMMETRIC_PER_CHANNEL_PARAMS_H_
