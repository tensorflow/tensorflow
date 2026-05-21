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
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace ifrt_serving {

// A handle that bundles necessary information for transferring a single input
// tensor to devices.
struct InputHandle {
  // The input tensor to be transferred.
  tensorflow::Tensor tensor;
  // The IFRT dtype of the input tensor.
  xla::ifrt::DType ifrt_dtype;
  // The IFRT shape of the input tensor.
  std::shared_ptr<const xla::ifrt::Shape> ifrt_shape;
  // The XLA shape of the input tensor.
  std::shared_ptr<const xla::Shape> input_xla_shape;
  // The devices to transfer the tensor to.
  xla::ifrt::DeviceListRef device_list;
  // The sharding of the tensor.
  xla::ifrt::ShardingRef ifrt_sharding;
  // The layout of the input tensor.
  xla::ifrt::LayoutRef xla_input_layout;
  // The byte strides of the input tensor.
  absl::Span<const int64_t> byte_strides;

  // IFRT pack-inputs H2D fusion metadata.
  // -1  : transfer this operand individually (default).
  // >=0 : pack into the named transfer group.
  int64_t pack_group_id = -1;
  // Byte offset within the group's coalesced host scratch buffer. Meaningful
  // only when pack_group_id >= 0.
  int64_t pack_offset = 0;
  // If true, the resulting ifrt::Array for this handle represents the entire
  // coalesced group (used for the rewritten executable parameter). Exactly one
  // handle per group must have this set.
  bool is_pack_group_representative = false;
  // The expected shape of the packed buffer for representative handles.
  // Meaningful only when is_pack_group_representative is true.
  std::shared_ptr<const xla::Shape> expected_packed_xla_shape = nullptr;
};

// A per-request H2D transfer executor. The caller should call
// `RegisterH2DTransfer` to register tensors to be transferred, and then call
// `RunH2DTransfers` to start the transfers. The futures returned by
// `RegisterH2DTransfer` will only be guaranteed to be fulfilled after
// `RunH2DTransfers` returns OK.
class H2DTransferExecutor {
 public:
  // TODO(b/445201291): Make the constructor private once the
  // H2DTransferExecutorFactory is plumbed through the stack.
  explicit H2DTransferExecutor(xla::ifrt::Client& ifrt_client);
  virtual ~H2DTransferExecutor() = default;

  // Registers a list of tensors to be transferred to devices.
  // This should be called only before `RunH2DTransfers` once.
  virtual absl::StatusOr<tsl::Future<std::vector<xla::ifrt::ArrayRef>>>
  ScheduledH2DTransfers(absl::Span<const InputHandle> handles,
                        tsl::thread::ThreadPool& thread_pool);

  // Executes the H2D transfers for all registered tensors.
  virtual absl::Status RunH2DTransfers();

 protected:
  xla::ifrt::Client& ifrt_client_;
};

class H2DTransferExecutorFactory {
 public:
  virtual ~H2DTransferExecutorFactory() = default;
  virtual absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
  CreateH2DTransferExecutor(xla::ifrt::Client& ifrt_client);
};

// Create a tensor from the given host tensor based on given device ids and
// sharding information.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromTensor(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    absl::Span<const int> device_ids, xla::ifrt::ShardingRef sharding,
    const tsl::thread::ThreadPool& thread_pool,
    const xla::ifrt::LayoutRef& xla_input_layout);

// A variant of the above api. The difference is that the user passes in
// device_list directly instead of a list of device_ids.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromTensor(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    const xla::ifrt::DeviceListRef& device_list,
    xla::ifrt::ShardingRef sharding, const tsl::thread::ThreadPool& thread_pool,
    const xla::ifrt::LayoutRef& xla_input_layout);

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
tsl::Future<tensorflow::Tensor> MakeTensorFromArray(
    xla::ifrt::Client& ifrt_client, xla::ifrt::Array& input_array,
    const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list,
    tsl::thread::ThreadPool& thread_pool);

// A wrapper around xla::ShapeUtil::ByteStrides to get the byte strides of a
// TensorFlow tensor.
std::optional<absl::InlinedVector<int64_t, 4>> GetByteStrides(
    tensorflow::DataType dtype, const tensorflow::TensorShape& shape);

// Description of a single H2D-transfer input together with its pack-plan
// metadata, used by MakeArraysFromTensorsPacked.
struct PackedTensorInput {
  // Source host tensor.
  tensorflow::Tensor tensor;
  // Sharding to apply to this operand's typed ifrt::Array (used only when
  // pack_group_id == -1, i.e., this operand is transferred individually).
  xla::HloSharding hlo_sharding;
  // -1 = transfer individually; >= 0 = fuse into the named pack group.
  int64_t pack_group_id = -1;
  // Byte offset within the pack group's host scratch buffer. Meaningful only
  // when pack_group_id >= 0.
  int64_t pack_offset = 0;
};

// Builds the list of ifrt::Array operands that LoadedExecutable::Execute()
// expects when its callee was rewritten by PackInputsPass.
//
// Output ordering, matching the post-rewrite signature
// `[kept individuals, in original order] ++ [packed buffer]`:
//   1. For each input with pack_group_id == -1, in the order they appear in
//      `inputs`: one ifrt::Array built from its native tensor.
//   2. For each unique pack_group_id >= 0 in ascending order: one ifrt::Array
//      built from a packed tensor<Nxi8> host buffer assembled by memcpying
//      each member's bytes at its pack_offset.
//
// Lifetime: the packed Tensor is held alive by the returned ifrt::Array via
// MakeArrayFromHostBuffer's `on_done_with_host_buffer` capture, so the caller
// need not extend it. (Defends against the async-execution case where the
// packed bytes would otherwise be freed before the H2D completes.)
//
// `packed_sharding` is the HloSharding from the executable's compile metadata
// for the packed parameter — derived from the executable, NOT from any input's
// sharding (which would be wrong for the rank-1 INT8 packed buffer).
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> MakeArraysFromTensorsPacked(
    xla::ifrt::Client& ifrt_client, absl::Span<const PackedTensorInput> inputs,
    const xla::ifrt::DeviceListRef& device_list,
    const xla::HloSharding& packed_sharding,
    const tsl::thread::ThreadPool& thread_pool);

// Converts `hlo_sharding` to `xla::ifrt::Sharding`.
//
// Returns `xla::ifrt::SingleDeviceSharding` if `device_list` has only one
// device or `hlo_sharding` is maximal and not replicated (i.e. the entire
// tensor is on a single device). Otherwise returns `xla::ifrt::HloSharding`.
//
// Returns error if `hlo_sharding` is not one of following supported cases:
// * Tiled: The tensor is split into pieces, each assigned to a device.
// * Replicated: The entire tensor is copied to every device.
// * TileMaximal: The entire tensor is on a single device (when not replicated).
absl::StatusOr<xla::ifrt::ShardingRef> ToIfrtSharding(
    xla::ifrt::Client& ifrt_client, const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list);

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  //  TENSORFLOW_CORE_TFRT_IFRT_SHARDING_UTILS_H_
