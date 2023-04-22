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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_REVIVAL_STATE_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_REVIVAL_STATE_H_

#include <string>

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tf_concrete_function_revival_state.h"

namespace tensorflow {

// All "Resources" should have these 3 saved functions:
// https://github.com/tensorflow/tensorflow/blob/86dc281333d7d277ddc1882f2bca4b17e7ec40e5/tensorflow/python/training/tracking/tracking.py#L277-L281
struct RestoredResourceRevivalState {
  std::string device;
  TFConcreteFunctionRevivalState* create_resource = nullptr;
  TFConcreteFunctionRevivalState* initialize = nullptr;
  TFConcreteFunctionRevivalState* destroy_resource = nullptr;
  ImmediateTensorHandlePtr resource_handle = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_RESTORED_RESOURCE_REVIVAL_STATE_H_
