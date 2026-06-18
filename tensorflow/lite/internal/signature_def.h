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
#ifndef TENSORFLOW_LITE_INTERNAL_SIGNATURE_DEF_H_
#define TENSORFLOW_LITE_INTERNAL_SIGNATURE_DEF_H_

#include <map>
#include <string>

namespace tflite {
namespace internal {

// Structure representing SignatureDef inputs/outputs.
struct SignatureDef {
  // Maps name in signature def as key to index of the tensor in the model.
  std::map<std::string, uint32_t> inputs;
  // Maps name in signature def as key to index of the tensor in the model.
  std::map<std::string, uint32_t> outputs;
  // The key of this SignatureDef in the SavedModel signature def map.
  std::string signature_key;
  // The subgraph index of the signature in the model.
  uint32_t subgraph_index;
};

}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_INTERNAL_SIGNATURE_DEF_H_
