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

#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_api.h"

namespace stream_executor {
namespace tpu {

TfTpu_BaseFn* InitializeApiFn() {
  static TfTpu_BaseFn base_fn;
  return &base_fn;
}

const TfTpu_OpsApiFn* OpsApiFn() {
  static TfTpu_OpsApiFn ops_api_fn;
  return &ops_api_fn;
}

}  // namespace tpu
}  // namespace stream_executor
