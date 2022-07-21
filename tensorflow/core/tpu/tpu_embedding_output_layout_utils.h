/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OUTPUT_LAYOUT_UTILS_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OUTPUT_LAYOUT_UTILS_H_

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {
namespace tpu {

// Computes the shape of the output tensors from an embedding configuration.
Status ComputeOutputTensorShapes(
    const TPUEmbeddingConfiguration& config,
    std::vector<tensorflow::TensorShapeProto>* shapes);

// Computes the shape of the output tensors based on the number of input
// features.
Status ComputeOutputTensorShapesFromFeature(
    const TPUEmbeddingConfiguration& config,
    std::vector<tensorflow::TensorShapeProto>* shapes);

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OUTPUT_LAYOUT_UTILS_H_
