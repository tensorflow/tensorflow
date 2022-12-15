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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"

namespace xla {
namespace ifrt {
namespace {

const bool kUnused =
    (test_util::RegisterClientFactory(
         []() -> StatusOr<std::unique_ptr<Client>> {
           TF_ASSIGN_OR_RETURN(auto pjrt_client,
                               xla::GetTfrtCpuClient(/*asynchronous=*/true,
                                                     /*cpu_device_count=*/2));
           return PjRtClient::Create(std::move(pjrt_client));
         }),
     true);

}  // namespace
}  // namespace ifrt
}  // namespace xla
