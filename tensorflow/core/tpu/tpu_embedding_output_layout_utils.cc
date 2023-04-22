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

#include "tensorflow/core/tpu/tpu_embedding_output_layout_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_output_layout.pb.h"

namespace tensorflow {
namespace tpu {

Status ComputeOutputTensorShapes(const TPUEmbeddingConfiguration& config,
                                 std::vector<TensorShapeProto>* shapes) {
  int batch_size = config.batch_size_per_tensor_core();

  for (const TPUEmbeddingConfiguration::TableDescriptor& table :
       config.table_descriptor()) {
    TensorShapeProto shape;
    auto* dim0 = shape.add_dim();
    dim0->set_size(table.num_features() * batch_size);
    auto* dim1 = shape.add_dim();
    dim1->set_size(table.dimension());
    shapes->push_back(shape);
  }
  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
