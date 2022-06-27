/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_WEIGHTS_CONVERTER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_WEIGHTS_CONVERTER_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class ConverterToConvWeights : public GPUOperation {
 public:
  ConverterToConvWeights(const OperationDef& definition,
                         const WeightsDescription& weights_desc,
                         Layout input_layout);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConverterToConvWeights(ConverterToConvWeights&& operation) = default;
  ConverterToConvWeights& operator=(ConverterToConvWeights&& operation) =
      default;
  ConverterToConvWeights(const ConverterToConvWeights&) = delete;
  ConverterToConvWeights& operator=(const ConverterToConvWeights&) = delete;

 private:
  std::string GetConverterToConvWeightsCode();

  OHWI GetWeightsSize() const;

  WeightsDescription weights_desc_;

  Layout input_layout_;  // Can be only OHWI or HWIO
  // if input_layout_ is OHWI: reinterpreting weights as OHWI-BHWC tensor
  // if input_layout_ is HWIO: reinterpreting weights as HWIO-BHWC tensor
};

ConverterToConvWeights CreateConverterToConvWeights(
    const OperationDef& definition, const WeightsDescription& weights_desc,
    Layout input_layout);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONV_WEIGHTS_CONVERTER_H_
