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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_INLINE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_INLINE_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

namespace tflite {
namespace gpu {
namespace gl {

// Fuses every two nodes where first node does default output and second node
// is INLINE.
//
// Generates code as follows:
//   1. all uniforms are inlined
//   2. source code is wrapped into {}
// For example:
//  value = clamp(value, 0.0, clip);
//  +
//  value = 1.0 / (1.0 + exp(-1.0 * value));
// will turn into:
//  {
//    value = clamp(value, 0.0, clip);
//  }
//  {
//    value = 1.0 / (1.0 + exp(-1.0 * value));
//  }
class FuseAutoOutputWithInline : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final { return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_FUSE_INLINE_H_
