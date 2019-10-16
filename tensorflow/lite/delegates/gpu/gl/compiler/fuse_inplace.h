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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_INPLACE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_INPLACE_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

namespace tflite {
namespace gpu {
namespace gl {

// Fuse two shaders where second shader is inline shader with the first.
// First shader should have a special symbol that defines a place where such
// fusion should be made and what variable needs to be changed.
// Second shader needs to operation with 'value_0' variable.
// Example:
//
//  First shader:
//   vec4 result = input_data_0.data[gid.x, gid.y, gid.z];
//   $inplace_update:result$
//   ...
//   output_data_0.data[1,2,3] = result;
//
//  Second shader:
//   value_0 = max(value_0, 0);
//
//  Fused shader:
//   vec4 result = input_data_0.data[gid.x, gid.y, gid.z];
//   result = max(result, 0);
//   ...
//   output_data_0.data[1,2,3] = result;
//
class FuseInplaceUpdate : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final;
};

// Removes all %inplace_update:XXX% strings from the code.
class RemoveUnusedInplaceUpdates : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_INPLACE_H_
