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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_SHAPE_INFERENCE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_SHAPE_INFERENCE_H_

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace eager {

absl::Status RunShapeInference(
    const NodeDef& ndef, const FunctionLibraryDefinition& lib_def,
    const absl::InlinedVector<TensorHandle*, 4UL>& inputs,
    const absl::InlinedVector<TensorHandle*, 2UL>& retvals);

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_SHAPE_INFERENCE_H_
