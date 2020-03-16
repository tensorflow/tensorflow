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
#include "tensorflow/compiler/mlir/lite/quantization/xla/quantize.h"

#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

namespace mlir {
namespace xla_hlo {

// Quantizes the model in the computation.
tensorflow::Status XlaQuantize(const tensorflow::tf2xla::Config& config,
                               xla::XlaComputation* computation) {
  return tensorflow::Status::OK();
}

}  // namespace xla_hlo
}  // namespace mlir
