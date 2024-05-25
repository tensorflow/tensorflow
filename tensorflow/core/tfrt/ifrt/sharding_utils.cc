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
// Enable definition of Eigen::ThreadPoolDevice instead of just declaration.
#define EIGEN_USE_THREADS

#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tensorflow/core/tpu/kernels/sharding_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

// Shard the given `input_tensor` into equal shapes of slices.
//
// `num_paritions_per_axis` specifies the number of partitions along
// each axis (dimension).
//
// `num_replicas` specifies the number of replication for each partitioned
// sliced buffer.
//
// `devices` contains a list of devices flattend into the following
// order: [slice0][replicate0], [slice0][replicate1], ..., [slice1][replicate0],
// [slice1][replicate1], ...
absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
SplitAndCreateArraysFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    const std::vector<int32_t>& num_partitions_per_axis, int num_replicas,
    const std::vector<xla::ifrt::Device*>& devices,
    const tsl::thread::ThreadPool& thread_pool) {
  Eigen::ThreadPoolDevice thread_pool_device(thread_pool.AsEigenThreadPool(),
                                             thread_pool.NumThreads());

  int64_t num_slices = 1;
  for (auto k : num_partitions_per_axis) {
    num_slices *= k;
  }

  tensorflow::DataType tensor_data_type = input_tensor.dtype();
  std::vector<int32_t> paddings(num_partitions_per_axis.size(), 0);
  std::vector<tensorflow::Tensor> split_tensors;
  split_tensors.resize(num_slices);

  auto allocate_output_fn =
      [&](int i, const tensorflow::TensorShape& output_slice_shape,
          tensorflow::Tensor** tensor) {
        if (i < 0 || i >= split_tensors.size()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Index ", i, " out of range [0, ", split_tensors.size(), "]"));
        }
        split_tensors[i] =
            tensorflow::Tensor(tensor_data_type, output_slice_shape);
        *tensor = &split_tensors[i];
        return absl::OkStatus();
      };

  // Fast path for output in the simple no split case.
  auto assign_or_copy_value_fn =
      [&](const tensorflow::Tensor& input) -> Status {
    split_tensors[0] = input;
    return absl::OkStatus();
  };

  // XlaNDSplitter only support rank (0, 8] as there is no concept of split for
  // rank 0 tensor.
  if (input_tensor.shape().dims() == 0) {
    if (split_tensors.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Rank 0 tensor only expects 1 slice but got ", split_tensors.size()));
    }
    split_tensors[0] = input_tensor;
  } else {
    switch (input_tensor.dtype()) {
#define CASE(type)                                                             \
  case tensorflow::DataTypeToEnum<type>::value: {                              \
    TF_ASSIGN_OR_RETURN(auto splitter,                                         \
                        (XlaNDSplitter<Eigen::ThreadPoolDevice, type>::Create( \
                            num_partitions_per_axis, num_slices, paddings,     \
                            /*has_paddings=*/false)));                         \
    TF_RETURN_IF_ERROR(                                                        \
        splitter.Split(&input_tensor, "input tensor", assign_or_copy_value_fn, \
                       allocate_output_fn, thread_pool_device));               \
  } break;
      TF_CALL_ALL_TYPES(CASE);
      TF_CALL_quint8(CASE);
#undef CASE
      default:
        return absl::InvalidArgumentError("Unsupported data type");
    }
  }

  if (split_tensors.size() * num_replicas != devices.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expect ", devices.size(), " but got ",
                     split_tensors.size(), " x ", num_replicas));
  }

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  arrays.reserve(devices.size());
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype, ToIfrtDType(tensor_data_type));
  auto device_iter = devices.begin();
  for (int slice_idx = 0; slice_idx < split_tensors.size(); ++slice_idx) {
    auto& tensor = split_tensors[slice_idx];

    for (int i = 0; i < num_replicas; ++i) {
      VLOG(2) << "Make array for buffer slice " << slice_idx << " at "
              << tensor.data();
      if (device_iter == devices.end()) {
        return absl::InternalError(
            absl::StrCat("Missing Device ", i, " for slice ", slice_idx));
      }
      auto single_device_sharding = xla::ifrt::SingleDeviceSharding::Create(
          *device_iter, xla::ifrt::MemoryKind());

      TF_ASSIGN_OR_RETURN(
          auto array,
          ifrt_client.MakeArrayFromHostBuffer(
              tensor.data(), dtype,
              xla::ifrt::Shape(tensor.shape().dim_sizes()),
              /*byte_strides=*/{}, std::move(single_device_sharding),
              xla::ifrt::Client::HostBufferSemantics::
                  kImmutableUntilTransferCompletes,
              [tensor, slice_idx]() {
                // Keep tensor alive
                VLOG(2) << "Done with host buffer for slice " << slice_idx
                        << " at " << tensor.data();
              }));
      arrays.push_back(std::move(array));
      device_iter++;
    }
  }
  return arrays;
}

// Reassembles split tensors back into one tensor.
//
// `num_concats` specifies the number of split tensors along
// each axis (dimension).
//
// `disassembled_tensors` contains a list of tensor flattened into the
// minor-to-major order.
//
absl::StatusOr<tensorflow::Tensor> MakeTensorFromDisassembledTensors(
    xla::ifrt::Client& ifrt_client,
    absl::Span<const tensorflow::Tensor> disassembled_tensors,
    const std::vector<int>& num_concats,
    tensorflow::DataType output_tensor_type,
    const tensorflow::TensorShape& output_tensor_shape,
    const tsl::thread::ThreadPool& thread_pool) {
  Eigen::ThreadPoolDevice thread_pool_device(thread_pool.AsEigenThreadPool(),
                                             thread_pool.NumThreads());

  int num_slices = 1;
  for (int i = 0; i < num_concats.size(); ++i) {
    num_slices *= num_concats[i];
  }

  if (num_slices != disassembled_tensors.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expect number of slices is ", disassembled_tensors.size(),
                     " but got ", num_slices, ""));
  }
  std::vector<int32_t> paddings(num_concats.size(), 0);

  tensorflow::Tensor output_tensor(output_tensor_type, output_tensor_shape);

  auto assign_or_copy_value_fn = [&](const Tensor& tensor) {
    output_tensor = tensor;
    return absl::OkStatus();
  };
  auto get_output_fn = [&]() -> absl::StatusOr<tensorflow::Tensor*> {
    return &output_tensor;
  };

  switch (output_tensor.dtype()) {
#define CASE(type)                                                        \
  case tensorflow::DataTypeToEnum<type>::value: {                         \
    TF_ASSIGN_OR_RETURN(                                                  \
        auto concatenator,                                                \
        (XlaNDConcatenator<Eigen::ThreadPoolDevice, type>::Create(        \
            num_concats, num_slices, paddings, /*has_paddings=*/false))); \
    TF_RETURN_IF_ERROR(concatenator.ComputeInternal(                      \
        disassembled_tensors, assign_or_copy_value_fn, get_output_fn,     \
        thread_pool_device));                                             \
  } break;
    TF_CALL_ALL_TYPES(CASE);
    TF_CALL_quint8(CASE);
#undef CASE
    default:
      return absl::InvalidArgumentError("Unsupported data type");
  }

  return output_tensor;
}

absl::StatusOr<int> VerifyIndexDomainsAndGetReplicas(
    absl::Span<xla::ifrt::IndexDomain> index_domains,
    const tensorflow::TensorShape& tensor_shape) {
  if (index_domains.size() <= 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expect multiple index domains but got ", index_domains.size()));
  }

  for (auto index_domain = index_domains.begin();
       index_domain < index_domains.end(); ++index_domain) {
    if (index_domain->shape().dims().size() != tensor_shape.dims()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expect equal rank of ", tensor_shape.dims(),
                       " but got ", index_domain->shape().dims().size()));
    }
  }

  // Only support equal shape for all index domains
  auto first_index_domain = index_domains.begin();
  for (auto index_domain = index_domains.begin() + 1;
       index_domain < index_domains.end(); ++index_domain) {
    if (first_index_domain->shape() != index_domain->shape()) {
      return absl::UnimplementedError(absl::StrCat(
          "Expect equal shape of ", first_index_domain->shape().DebugString(),
          " but got ", index_domain->shape().DebugString()));
    }
  }

  // Verify that each `IndexDomain` appear the same `num_replica` times. Since
  // shapes are the same for all `IndexDomain`, this also implies each `origin`
  // appear `num_replica` times.
  struct IndexDomainLexicographicalComparator {
    bool operator()(const xla::ifrt::IndexDomain& a,
                    const xla::ifrt::IndexDomain& b) const {
      return std::lexicographical_compare(
          a.origin().elements().begin(), a.origin().elements().end(),
          b.origin().elements().begin(), b.origin().elements().end());
    }
  };
  absl::btree_map<xla::ifrt::IndexDomain, int,
                  IndexDomainLexicographicalComparator>
      index_domain_counts;
  for (const auto& index_domain : index_domains) {
    index_domain_counts[index_domain]++;
  }

  std::vector<xla::ifrt::IndexDomain> unique_index_domains;
  unique_index_domains.reserve(index_domain_counts.size());
  int num_replicas = index_domain_counts.begin()->second;
  for (const auto& [index_domain, count] : index_domain_counts) {
    if (count != num_replicas) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Expected ", num_replicas, " replicas for ",
          index_domain.DebugString(), " but got ", count, " replicas"));
    }
    unique_index_domains.push_back(index_domain);
  }

  // Verify that distances of between origins of neighbouring `IndexDomain`
  // bounded by shape. Note that unique_indexx_domains are already in sorted
  // order.
  auto prev_iter = unique_index_domains.begin();
  auto next_iter = unique_index_domains.begin() + 1;
  const auto& bounded_box = first_index_domain->shape();
  while (prev_iter != unique_index_domains.end() &&
         next_iter != unique_index_domains.end()) {
    xla::ifrt::Index offset = next_iter->origin() - prev_iter->origin();
    for (int dim = 0; dim < bounded_box.dims().size(); ++dim) {
      if (std::abs(offset.elements()[dim]) != bounded_box.dims()[dim] &&
          offset.elements()[dim] != 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "IndexDomains should not have gap or overlap, but got ",
            prev_iter->DebugString(), " and ", next_iter->DebugString(),
            " that have offset of ", offset.DebugString()));
      }
    }
    prev_iter = next_iter;
    next_iter++;
  }

  // Verify the last `IndexDomain`'s upper end of the bound matches with the
  // tensor shape. Together with the above check, this provides an approximation
  // to the following two assumptions:
  // 1. the union of all IndexDomain covers the entire global shape array with
  // no gaps.
  // 2. no two index_domain have any overlap.
  std::vector<int64_t> bounded_shape;
  const auto& last_index_domain = unique_index_domains.back();
  bounded_shape.reserve(last_index_domain.shape().dims().size());
  for (int d = 0; d < last_index_domain.shape().dims().size(); ++d) {
    bounded_shape.push_back(last_index_domain.origin().elements()[d] +
                            last_index_domain.shape().dims()[d]);
  }

  if (xla::ifrt::Shape(bounded_shape) !=
      xla::ifrt::Shape(tensor_shape.dim_sizes())) {
    return absl::FailedPreconditionError(absl::StrCat(
        "IndexDomain ", last_index_domain.DebugString(),
        " does not overlap with tensor shape ", tensor_shape.DebugString()));
  }

  return num_replicas;
}

// A simple wrapper function to create ifrt array for one single device.
absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
CreateArrayFromHostTensorForSingleDevice(xla::ifrt::Client& ifrt_client,
                                         const tensorflow::Tensor& tensor,
                                         xla::ifrt::Device* device) {
  TF_ASSIGN_OR_RETURN(auto dtype, ToIfrtDType(tensor.dtype()));

  VLOG(2) << "Make single device array for buffer slice at " << tensor.data();
  auto single_device_sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  return ifrt_client.MakeArrayFromHostBuffer(
      tensor.data(), dtype, ToIfrtShape(tensor.shape()),
      /*byte_strides=*/{}, std::move(single_device_sharding),
      xla::ifrt::Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
      [tensor]() {
        // Keep tensor alive
        VLOG(2) << "Done with single device host buffer for slice " << " at "
                << tensor.data();
      });
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
MakeAssembledArrayFromHostBuffer(xla::ifrt::Client& ifrt_client,
                                 const tensorflow::Tensor& input_tensor,
                                 const xla::HloSharding& hlo_sharding,
                                 const xla::ifrt::DeviceList& device_list,
                                 const tsl::thread::ThreadPool& thread_pool) {
  // TODO(b/316959894): use xla::HloSharding to identifying sharding axis.
  auto sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), hlo_sharding);

  VLOG(2) << "Assembling arrays by sharding " << sharding->DebugString();

  TF_ASSIGN_OR_RETURN(auto index_domains,
                      sharding->IndexDomains(
                          xla::ifrt::Shape(input_tensor.shape().dim_sizes())));

  TF_ASSIGN_OR_RETURN(int index_domain_replicas,
                      VerifyIndexDomainsAndGetReplicas(
                          absl::MakeSpan(index_domains), input_tensor.shape()));

  const auto& first_index_domain = index_domains.begin();
  std::vector<int32_t> num_partitions_per_axis;
  int total_num_partitions = 1;
  num_partitions_per_axis.reserve(input_tensor.shape().dims());
  for (int dim = 0; dim < input_tensor.shape().dims(); ++dim) {
    int target_size = first_index_domain->shape().dims()[dim];
    if (input_tensor.shape().dim_size(dim) % target_size != 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Only support even sharding, but input tensor shape ",
          input_tensor.shape().DebugString(), " not even splittable to ",
          first_index_domain->shape().DebugString()));
    }
    int num_partitions = input_tensor.shape().dim_size(dim) / target_size;
    total_num_partitions *= num_partitions;
    num_partitions_per_axis.push_back(num_partitions);
  }

  if (total_num_partitions > sharding->devices().size() ||
      sharding->devices().size() % total_num_partitions != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Number of devices ", sharding->devices().size(),
        " not a multiple of number of partitions", total_num_partitions));
  }

  // Assume index domains are non-overlapping and each index domain appears
  // exactly num_replicates times. This allows us to rely on
  // lexicographical sorting to replicate slices in the correct order.
  int num_replicas = sharding->devices().size() / total_num_partitions;
  if (index_domain_replicas != num_replicas) {
    return absl::FailedPreconditionError(
        absl::StrCat("IndexDomain indicates ", index_domain_replicas,
                     " replicas, but got ", num_replicas, " replicas"));
  }

  // Sorted the IndexDomain and devices from major to minor dimenson. For
  // example, a two dimension IndexDomain will be ordered by [0, 0], [0, 1], [1,
  // 0], [1, 1].
  // This is O(n*log(n)) vs looking for devices individually which is O(n^2).
  struct IndexDomainDevice {
    xla::ifrt::IndexDomain index_domain;
    xla::ifrt::Device* device;
    // The index of this `device`/`index_domain` in the
    // sharding.devices/index_domains.
    int original_shard_index;
  };
  std::vector<IndexDomainDevice> index_domain_devices;
  index_domain_devices.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    index_domain_devices.push_back(
        {index_domains[i], sharding->devices()[i], i});
  }
  std::sort(index_domain_devices.begin(), index_domain_devices.end(),
            [](const IndexDomainDevice& a, const IndexDomainDevice& b) {
              return std::lexicographical_compare(
                  a.index_domain.origin().elements().begin(),
                  a.index_domain.origin().elements().end(),
                  b.index_domain.origin().elements().begin(),
                  b.index_domain.origin().elements().end());
            });
  // Now the devices is in order.
  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(index_domain_devices.size());
  std::vector<int> original_device_indices;
  original_device_indices.reserve(index_domain_devices.size());
  for (auto& [index_domain, device, original_device_index] :
       index_domain_devices) {
    devices.push_back(device);
    original_device_indices.push_back(original_device_index);
    VLOG(3) << "Device " << device->ToString();
  }

  TF_ASSIGN_OR_RETURN(auto arrays,
                      SplitAndCreateArraysFromHostBuffer(
                          ifrt_client, input_tensor, num_partitions_per_axis,
                          num_replicas, devices, thread_pool));

  // Re-arranged arrays back to original device order
  std::vector<tsl::RCReference<xla::ifrt::Array>> rearranged_arrays;
  rearranged_arrays.resize(arrays.size());
  for (int i = 0; i < arrays.size(); ++i) {
    rearranged_arrays[original_device_indices[i]] = std::move(arrays[i]);
  }

  return ifrt_client.AssembleArrayFromSingleDeviceArrays(
      xla::ifrt::Shape(input_tensor.shape().dim_sizes()), std::move(sharding),
      absl::MakeSpan(rearranged_arrays),
      xla::ifrt::ArrayCopySemantics::kDonateInput);
}

}  // namespace

absl::StatusOr<tensorflow::Tensor> MakeTensorFromArray(
    xla::ifrt::Client& ifrt_client, xla::ifrt::Array& input_array,
    const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceList& device_list,
    const tsl::thread::ThreadPool& thread_pool) {
  TF_ASSIGN_OR_RETURN(tensorflow::DataType data_type,
                      ToTensorDataType(input_array.dtype()));
  tensorflow::TensorShape tensor_shape = ToTensorShape(input_array.shape());

  VLOG(2) << "Create tensor from array based on sharding: "
          << hlo_sharding.ToString();

  if (hlo_sharding.IsReplicated()) {
    VLOG(1) << "Fast path for replication";
    // fast path for replication.
    TF_ASSIGN_OR_RETURN(auto fully_replicated_array,
                        input_array.FullyReplicatedShard(
                            xla::ifrt::ArrayCopySemantics::kDonateInput));

    if (fully_replicated_array->shape() != ToIfrtShape(tensor_shape)) {
      return absl::InternalError(absl::StrCat(
          "Not fully replicated output. Expected ", tensor_shape.DebugString(),
          " but got ", fully_replicated_array->shape().DebugString()));
    }
    tensorflow::Tensor output_tensor(data_type, tensor_shape);
    TF_RETURN_IF_ERROR(
        fully_replicated_array
            ->CopyToHostBuffer(output_tensor.data(),
                               /*byte_strides=*/std::nullopt,
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());
    return output_tensor;
  } else if (hlo_sharding.IsTileMaximal()) {
    // Maximal implies single device
    VLOG(1) << "Fast path for maximal";
    TF_ASSIGN_OR_RETURN(
        std::vector<tsl::RCReference<xla::ifrt::Array>> disassembled_array,
        input_array.DisassembleIntoSingleDeviceArrays(
            xla::ifrt::ArrayCopySemantics::kDonateInput));

    int64_t device_id = hlo_sharding.GetUniqueDevice();

    tensorflow::Tensor output_tensor(data_type, tensor_shape);
    TF_RETURN_IF_ERROR(
        disassembled_array[device_id]
            ->CopyToHostBuffer(output_tensor.data(),
                               /*byte_strides=*/std::nullopt,
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());
    return output_tensor;
  }

  auto ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), hlo_sharding);

  TF_ASSIGN_OR_RETURN(auto index_domains,
                      ifrt_sharding->IndexDomains(ToIfrtShape(tensor_shape)));

  TF_ASSIGN_OR_RETURN(int index_domain_replicas,
                      VerifyIndexDomainsAndGetReplicas(
                          absl::MakeSpan(index_domains), tensor_shape));

  if (index_domain_replicas != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Subgroup replication is not supported at output. Number "
        "of unique index main ",
        index_domain_replicas, " is not equal to number of index domains",
        index_domains.size()));
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<tsl::RCReference<xla::ifrt::Array>> disassembled_array,
      input_array.DisassembleIntoSingleDeviceArrays(
          xla::ifrt::ArrayCopySemantics::kDonateInput));

  if (index_domains.size() != disassembled_array.size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Number of index domains ", index_domains.size(),
        " not equal to number of arrays ", disassembled_array.size()));
  }

  for (int i = 0; i < index_domains.size(); ++i) {
    if (index_domains[i].shape() != disassembled_array[i]->shape()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Index domain ", index_domains[i].shape().DebugString(),
                       " not equal to array shape ",
                       disassembled_array[i]->shape().DebugString()));
    }
  }

  // `VerifyIndexDomainsAndGetReplicas` already verify all index domains are of
  // equal shape.
  std::vector<int> num_concats;
  num_concats.reserve(tensor_shape.dims());
  const xla::ifrt::Shape& per_split_shape = index_domains[0].shape();
  int num_slices = 1;
  for (int i = 0; i < per_split_shape.dims().size(); ++i) {
    int dim_num_concats = tensor_shape.dim_size(i) / per_split_shape.dims()[i];
    num_slices *= dim_num_concats;
    num_concats.push_back(dim_num_concats);
  }
  if (num_slices != index_domains.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expect number of slices is ", index_domains.size(),
                     " but got ", num_slices));
  }

  VLOG(2) << "Index domains: ";
  for (const auto& index_domain : index_domains) {
    VLOG(2) << index_domain.DebugString();
  }

  // disassembled array is in device order.
  struct IndexDomainDeviceArray {
    xla::ifrt::IndexDomain index_domain;
    tsl::RCReference<xla::ifrt::Array> array;
  };
  std::vector<IndexDomainDeviceArray> index_domain_device_arrays;
  index_domain_device_arrays.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    index_domain_device_arrays.push_back(
        {index_domains[i], disassembled_array[i]});
  }

  std::sort(
      index_domain_device_arrays.begin(), index_domain_device_arrays.end(),
      [](const IndexDomainDeviceArray& a, const IndexDomainDeviceArray& b) {
        return std::lexicographical_compare(
            a.index_domain.origin().elements().begin(),
            a.index_domain.origin().elements().end(),
            b.index_domain.origin().elements().begin(),
            b.index_domain.origin().elements().end());
      });

  std::vector<xla::ifrt::Future<>> arrays_copy_status;
  std::vector<tensorflow::Tensor> input_tensors;
  input_tensors.reserve(index_domain_device_arrays.size());
  arrays_copy_status.reserve(index_domain_device_arrays.size());
  for (const auto& [index_domain, array] : index_domain_device_arrays) {
    tensorflow::TensorShape tensor_shape = ToTensorShape(index_domain.shape());
    TF_ASSIGN_OR_RETURN(tensorflow::DataType dtype,
                        ToTensorDataType(array->dtype()));
    tensorflow::Tensor tensor(dtype, tensor_shape);
    input_tensors.push_back(tensor);
    xla::ifrt::Future<> copy_status =
        array->CopyToHostBuffer(tensor.data(), /*byte_strides=*/{},
                                xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
    copy_status.OnReady([tensor](absl::Status status) {
      VLOG(1) << "Copy of tensor " << tensor.data() << " done with status "
              << status;
    });
    arrays_copy_status.push_back(std::move(copy_status));
  }

  TF_RETURN_IF_ERROR(
      xla::ifrt::JoinFutures(absl::MakeSpan(arrays_copy_status)).Await());

  return MakeTensorFromDisassembledTensors(
      ifrt_client, absl::MakeSpan(input_tensors), num_concats, data_type,
      tensor_shape, thread_pool);
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> MakeArrayFromTensor(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    const xla::ifrt::DeviceList& device_list,
    const xla::HloSharding& hlo_sharding,
    const tsl::thread::ThreadPool& thread_pool) {
  VLOG(3) << "IsTiled: " << hlo_sharding.IsTiled();
  VLOG(3) << "IsReplicated: " << hlo_sharding.IsReplicated();
  VLOG(3) << "IsTileMaximal: " << hlo_sharding.IsTileMaximal();
  if (!hlo_sharding.IsTiled() && !hlo_sharding.IsReplicated() &&
      !hlo_sharding.IsTileMaximal()) {
    return absl::UnimplementedError(absl::StrCat(
        "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
        hlo_sharding.ToString()));
  }

  VLOG(1) << "Hlo sharding: " << hlo_sharding.ToString();
  VLOG(1) << "Device list size: " << device_list.size();

  if (device_list.size() == 1) {
    return CreateArrayFromHostTensorForSingleDevice(ifrt_client, input_tensor,
                                                    device_list[0]);
  }

  // IsTileMaximal() also returns true for a replicate sharding created by
  // xla::HloSharding::Replicate().
  if (!hlo_sharding.IsReplicated() && hlo_sharding.IsTileMaximal()) {
    VLOG(1) << "Single device fast path for Maximal tiled tensor";
    xla::ifrt::Device* device;
    int unique_device_id = hlo_sharding.GetUniqueDevice();
    TF_ASSIGN_OR_RETURN(device, ifrt_client.LookupDevice(
                                    xla::ifrt::DeviceId(unique_device_id)));
    return CreateArrayFromHostTensorForSingleDevice(ifrt_client, input_tensor,
                                                    device);
  }

  return MakeAssembledArrayFromHostBuffer(ifrt_client, input_tensor,
                                          std::move(hlo_sharding), device_list,
                                          thread_pool);
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> MakeArrayFromTensor(
    xla::ifrt::Client& ifrt_client, const tensorflow::Tensor& input_tensor,
    absl::Span<const int> device_ids, const xla::HloSharding& hlo_sharding,
    const tsl::thread::ThreadPool& thread_pool) {
  if (device_ids.empty()) {
    return absl::InvalidArgumentError("device_ids cannot be empty");
  }
  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(device_ids.size());
  for (auto device_id : device_ids) {
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Device * device,
        ifrt_client.LookupDevice(xla::ifrt::DeviceId(device_id)));
    devices.push_back(device);
  }
  xla::ifrt::DeviceList device_list(
      xla::ifrt::DeviceList::Devices(devices.begin(), devices.end()));

  return MakeArrayFromTensor(ifrt_client, input_tensor, device_list,
                             hlo_sharding, thread_pool);
}

}  // namespace ifrt_serving
}  // namespace tensorflow
