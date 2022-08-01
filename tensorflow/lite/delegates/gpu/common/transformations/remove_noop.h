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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_REMOVE_NOOP_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_REMOVE_NOOP_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

namespace tflite {
namespace gpu {

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputConcat();

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputAdd();

std::unique_ptr<SequenceTransformation> NewRemoveDegenerateUpsampling();

// Removes reshape with input shape == output shape
std::unique_ptr<NodeTransformation> NewRemoveIdentityReshape();

// Removes strided slice with input tensor == output tensor
std::unique_ptr<NodeTransformation> NewRemoveIdentityStridedSlice();

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_REMOVE_NOOP_H_
