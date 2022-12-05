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

#include "tensorflow/compiler/xla/pjrt/pjrt_client_test.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace xla {
namespace {

// Register CPU as the backend for tests in pjrt_client_test.cc.
const bool kUnused =
    (RegisterTestClientFactory([]() {
       return GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/4);
     }),
     true);

}  // namespace
}  // namespace xla
