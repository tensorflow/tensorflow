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

#include <cstdint>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ifrt_serving {

// Create a tensor from the given host tensor based on given device ids and
// sharding information.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromTensor(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    absl::Span<const int> device_ids, const xla::HloSharding& hlo_sharding,
    const tsl::thread::ThreadPool& thread_pool);

// A variant of the above api. The difference is that the user passes in
// device_list directly instead of a list of device_ids.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromTensor(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    const xla::ifrt::DeviceListRef& device_list,
    const xla::HloSharding& hlo_sharding,
    const tsl::thread::ThreadPool& thread_pool);

// Reshard an disassembled array list back to one single tensor
// based on given sharding spec.
//
// input_array: the input device buffers.
//
// hlo_sharding: sharding spec that describes how the input device buffers are
// sharded.
//
// device_list: list of devices that is aligned with the order of device buffers
// in the `input_array`.
//
xla::ifrt::Future<tensorflow::Tensor> MakeTensorFromArray(
    xla::ifrt::Client& ifrt_client, xla::ifrt::Array& input_array,
    const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list,
    tsl::thread::ThreadPool& thread_pool);

// A wrapper around xla::ShapeUtil::ByteStrides to get the byte strides of a
// TensorFlow tensor.
std::optional<absl::InlinedVector<int64_t, 4>> GetByteStrides(
    tensorflow::DataType dtype, const tensorflow::TensorShape& shape);

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  //  TENSORFLOW_CORE_TFRT_IFRT_SHARDING_UTILS_H_
