/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_SHARDING_UTILS_H_
#define TENSORFLOW_CORE_TFRT_IFRT_SHARDING_UTILS_H_

#include <memory>

#include "xla/executable_run_options.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/sharding.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

// Sharded the given `data` by the `sharding` specification.
// It currently supports even sharding, replication and partial replication.
StatusOr<tsl::RCReference<xla::ifrt::Array>> MakeAssembledArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    std::shared_ptr<xla::ifrt::Sharding> sharding,
    const Eigen::ThreadPoolDevice& thread_pool_device);
}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  //  TENSORFLOW_CORE_TFRT_IFRT_SHARDING_UTILS_H_
