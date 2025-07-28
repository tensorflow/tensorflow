/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_TENSOR_IDENTIFIER_H_
#define TENSORFLOW_LITE_CORE_TENSOR_IDENTIFIER_H_

namespace tflite {

// Uniquely identifies a tensor in the model by storing the subgraph index and
// the tensor index within that subgraph. This is a stable way to reference a
// tensor, as the pointer to the tensor itself may be invalidated by operations
// that resize the tensor vector.
struct TfLiteTensorIdentifier {
  int subgraph_idx;
  int tensor_idx;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_TENSOR_IDENTIFIER_H_
