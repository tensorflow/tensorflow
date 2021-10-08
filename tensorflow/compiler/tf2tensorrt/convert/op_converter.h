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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class Converter;

// Parameters for each op converter.
struct OpConverterParams {
  // Constructor used for validation only.
  OpConverterParams(const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store,
                    TrtPrecisionMode precision_mode, bool use_calibration,
                    bool use_implicit_batch);

  // Constructor used for conversion.
  OpConverterParams(Converter* converter, const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store);

  Converter* converter = nullptr;
  const NodeDef& node_def;
  const std::vector<TRT_TensorOrWeights>& inputs;
  std::vector<TRT_TensorOrWeights>* outputs;
  const bool validation_only;
  TrtWeightStore* weight_store;
  const TrtPrecisionMode precision_mode;
  const bool use_calibration;
  const bool use_implicit_batch;
};

using OpConverter = std::function<Status(OpConverterParams*)>;

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
