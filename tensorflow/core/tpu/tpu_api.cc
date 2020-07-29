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

#include "tensorflow/core/tpu/tpu_api.h"

namespace tensorflow {
namespace tpu {

TfTpu_BaseFn* InitializeApiFn() {
  static TfTpu_BaseFn base_fn;
  return &base_fn;
}

TfTpu_ConfigApiFn* ConfigApiFn() {
  static TfTpu_ConfigApiFn config_api_fn;
  return &config_api_fn;
}

TfTpu_MeshStateApiFn* MeshStateApiFn() {
  static TfTpu_MeshStateApiFn mesh_state_api_fn;
  return &mesh_state_api_fn;
}

TfTpu_CompileApiFn* CompileApiFn() {
  static TfTpu_CompileApiFn compile_api_fn;
  return &compile_api_fn;
}

TfTpu_ExecutorApiFn* ExecutorApiFn() {
  static TfTpu_ExecutorApiFn executor_api_fn;
  return &executor_api_fn;
}

TfTpu_NodeContextApiFn* NodeContextApiFn() {
  static TfTpu_NodeContextApiFn node_context_api_fn;
  return &node_context_api_fn;
}

TfTpu_UtilApiFn* UtilApiFn() {
  static TfTpu_UtilApiFn util_api_fn;
  return &util_api_fn;
}

}  // namespace tpu
}  // namespace tensorflow
