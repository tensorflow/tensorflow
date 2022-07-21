/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/pjrt_api.h"

namespace tensorflow {
namespace tpu {

static const PJRT_Api* pjrt_api;

const PJRT_Api* PjrtApi() { return pjrt_api; }

void SetPjrtApi(const PJRT_Api* api) { pjrt_api = api; }

}  // namespace tpu
}  // namespace tensorflow
