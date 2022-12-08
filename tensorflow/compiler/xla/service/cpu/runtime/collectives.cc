// Copyright 2022 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/xla/service/cpu/runtime/collectives.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace cpu {

using mlir::succeeded;

using ::xla::runtime::CustomCall;
using ::xla::runtime::Executable;
using ::xla::runtime::MemrefView;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

static std::string ReplicaGroupsToString(
    CustomCall::TensorRef<int64_t> replica_groups) {
  if (replica_groups.shape[0] == 0) {
    return "{}";
  }
  std::string result;

  const auto& shape = replica_groups.shape;
  size_t stride = replica_groups.data.size() / shape[0];

  absl::StrAppend(&result, "{");
  for (size_t i = 0; i < replica_groups.data.size(); i += stride) {
    if (i > 0) {
      absl::StrAppend(&result, ", ");
    }

    auto start = replica_groups.data.begin() + i;
    llvm::ArrayRef<int64_t> inner_data(start, start + stride);

    absl::StrAppend(&result, "{");
    absl::StrAppend(
        &result,
        // The replica groups can have different sizes. Smaller groups are
        // padded with -1.
        absl::StrJoin(llvm::make_filter_range(
                          inner_data, [](int64_t id) { return id >= 0; }),
                      ", "));
    absl::StrAppend(&result, "}");
  }
  absl::StrAppend(&result, "}");

  return result;
}

static std::string SourceTargetPairsToString(
    CustomCall::TensorRef<int64_t> source_target_pairs) {
  std::string result;
  for (size_t i = 0; i < source_target_pairs.data.size(); i += 2) {
    if (i > 0) {
      absl::StrAppend(&result, ",");
    }
    absl::StrAppend(&result, source_target_pairs.data[i], "=",
                    source_target_pairs.data[i + 1]);
  }
  return result;
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaPartitionId {
  absl::StatusOr<int32_t> operator()(
      const ExecutableRunOptions* run_options) const;
  static XlaPartitionId Handler() { return XlaPartitionId(); }
};
}  // namespace

absl::StatusOr<int32_t> XlaPartitionId::operator()(
    const ExecutableRunOptions* run_options) const {
  int32_t result;
  __xla_cpu_runtime_PartitionId(run_options, &result);
  return result;
}

static bool PartitionId(xla::runtime::ExecutionContext* ctx, void** args,
                        void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.partition_id")
                             .Ret<int32_t>()
                             .UserData<const ExecutableRunOptions*>()
                             .To<RuntimeChecks()>(XlaPartitionId::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaReplicaId {
  absl::StatusOr<int32_t> operator()(
      const ExecutableRunOptions* run_options) const;
  static XlaReplicaId Handler() { return XlaReplicaId(); }
};
}  // namespace

absl::StatusOr<int32_t> XlaReplicaId::operator()(
    const ExecutableRunOptions* run_options) const {
  int32_t result;
  __xla_cpu_runtime_ReplicaId(run_options, &result);
  return result;
}

static bool ReplicaId(xla::runtime::ExecutionContext* ctx, void** args,
                      void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.replica_id")
                             .Ret<int32_t>()
                             .UserData<const ExecutableRunOptions*>()
                             .To<RuntimeChecks()>(XlaReplicaId::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaAllReduce {
  absl::Status operator()(const ExecutableRunOptions* run_options,
                          CustomCall::RemainingArgs buffers,
                          CustomCall::TensorRef<int64_t> replica_groups,
                          int64_t channel_id, int32_t use_global_device_ids,
                          int64_t op_id, int32_t reduction_kind) const;
  static XlaAllReduce Handler() { return XlaAllReduce(); }
};
}  // namespace

absl::Status XlaAllReduce::operator()(
    const ExecutableRunOptions* run_options, CustomCall::RemainingArgs buffers,
    CustomCall::TensorRef<int64_t> replica_groups, int64_t channel_id,
    int32_t use_global_device_ids, int64_t op_id,
    int32_t reduction_kind) const {
  if (replica_groups.shape.size() != 2) {
    return absl::InvalidArgumentError("replica_groups must be a 2d tensor.");
  }

  if (buffers.size() % 2) {
    return absl::InvalidArgumentError(
        "number of input buffers and output buffers must be equal.");
  }

  std::string replica_groups_str = ReplicaGroupsToString(replica_groups);
  int64_t num_buffers = static_cast<int64_t>(buffers.size()) / 2;

  llvm::SmallVector<void*> input_buffers, output_buffers;
  ShapeProto shape;
  for (int i = 0; i < num_buffers; ++i) {
    auto input = buffers.get<MemrefView>(i);
    auto output = buffers.get<MemrefView>(i + num_buffers);
    if (!succeeded(input) || !succeeded(output)) {
      return absl::InvalidArgumentError("all arguments must be memrefs.");
    }

    *shape.add_tuple_shapes() =
        ShapeUtil::MakeShapeWithDescendingLayout(input->dtype, input->sizes)
            .ToProto();
    input_buffers.push_back(input->data);
    output_buffers.push_back(output->data);
  }
  std::string shape_str =
      (shape.tuple_shapes().size() == 1 ? shape.tuple_shapes(0) : shape)
          .SerializeAsString();

  __xla_cpu_runtime_AllReduce(
      run_options, replica_groups_str.c_str(),
      static_cast<int32_t>(replica_groups_str.size()),
      static_cast<int32_t>(channel_id), use_global_device_ids, op_id,
      reduction_kind, shape_str.c_str(), static_cast<int32_t>(shape_str.size()),
      static_cast<int32_t>(num_buffers), input_buffers.data(),
      output_buffers.data());

  return absl::OkStatus();
}

static bool AllReduce(xla::runtime::ExecutionContext* ctx, void** args,
                      void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.cpu.all_reduce")
          .UserData<const ExecutableRunOptions*>()
          .RemainingArgs()
          .Attr<CustomCall::TensorRef<int64_t>>("replica_groups")
          .Attr<int64_t>("channel_handle")
          .Attr<int32_t>("use_global_device_ids")
          .Attr<int64_t>("op_id")
          .Attr<int32_t>("reduction_kind")
          .To<RuntimeChecks()>(XlaAllReduce::Handler())
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaTupleAllToAll {
  absl::Status operator()(const ExecutableRunOptions* run_options,
                          CustomCall::RemainingArgs buffers,
                          CustomCall::TensorRef<int64_t> replica_groups) const;
  static XlaTupleAllToAll Handler() { return XlaTupleAllToAll(); }
};
}  // namespace

absl::Status XlaTupleAllToAll::operator()(
    const ExecutableRunOptions* run_options, CustomCall::RemainingArgs buffers,
    CustomCall::TensorRef<int64_t> replica_groups) const {
  if (replica_groups.shape.size() != 2) {
    return absl::InvalidArgumentError("replica_groups must be a 2d tensor.");
  }

  if (buffers.size() % 2) {
    return absl::InvalidArgumentError(
        "number of input buffers and output buffers must be equal.");
  }

  std::string replica_groups_str = ReplicaGroupsToString(replica_groups);
  int64_t num_buffers = static_cast<int64_t>(buffers.size()) / 2;

  llvm::SmallVector<void*> input_buffers, output_buffers;
  for (int i = 0; i < num_buffers; ++i) {
    auto input = buffers.get<MemrefView>(i);
    auto output = buffers.get<MemrefView>(i + num_buffers);
    if (!succeeded(input) || !succeeded(output)) {
      return absl::InvalidArgumentError("all arguments must be memrefs.");
    }

    input_buffers.push_back(input->data);
    output_buffers.push_back(output->data);
  }

  auto first_input = *buffers.get<MemrefView>(0);
  size_t buffer_size = ShapeUtil::ByteSizeOfElements(
      ShapeUtil::MakeShape(first_input.dtype, first_input.sizes));

  __xla_cpu_runtime_AllToAll(run_options, 0, 0, replica_groups_str.c_str(),
                             static_cast<int32_t>(replica_groups_str.size()),
                             static_cast<int32_t>(num_buffers),
                             static_cast<int64_t>(buffer_size),
                             input_buffers.data(), output_buffers.data());

  return absl::OkStatus();
}

static bool TupleAllToAll(xla::runtime::ExecutionContext* ctx, void** args,
                          void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.cpu.all_reduce")
          .UserData<const ExecutableRunOptions*>()
          .RemainingArgs()
          .Attr<CustomCall::TensorRef<int64_t>>("replica_groups")
          .To<RuntimeChecks()>(XlaTupleAllToAll::Handler())
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaCollectivePermute {
  absl::Status operator()(const ExecutableRunOptions* run_options,
                          MemrefView input, MemrefView output,
                          CustomCall::TensorRef<int64_t> source_target_pairs,
                          int64_t channel_id) const;
  static XlaCollectivePermute Handler() { return XlaCollectivePermute(); }
};
}  // namespace

absl::Status XlaCollectivePermute::operator()(
    const ExecutableRunOptions* run_options, MemrefView input,
    MemrefView output, CustomCall::TensorRef<int64_t> source_target_pairs,
    int64_t channel_id) const {
  if (source_target_pairs.shape.size() != 2 ||
      source_target_pairs.shape[1] != 2) {
    return absl::InvalidArgumentError(
        "source_target_pairs must be a ?x2 tensor.");
  }
  size_t byte_size = ShapeUtil::ByteSizeOfElements(
      ShapeUtil::MakeShape(input.dtype, input.sizes));
  std::string source_target_pairs_str =
      SourceTargetPairsToString(source_target_pairs);

  __xla_cpu_runtime_CollectivePermute(
      run_options, static_cast<int32_t>(channel_id), 0,
      static_cast<int32_t>(byte_size), input.data, output.data,
      source_target_pairs_str.c_str(),
      static_cast<int32_t>(source_target_pairs_str.size()));

  return absl::OkStatus();
}

static bool CollectivePermute(xla::runtime::ExecutionContext* ctx, void** args,
                              void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla.cpu.collective_permute")
          .UserData<const ExecutableRunOptions*>()
          .Arg<MemrefView>()  // input
          .Arg<MemrefView>()  // output
          .Attr<CustomCall::TensorRef<int64_t>>("source_target_pairs")
          .Attr<int64_t>("channel_handle")
          .To<RuntimeChecks()>(XlaCollectivePermute::Handler())
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateXlaCpuCollectivesCall(
    xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.cpu.all_reduce", &xla::cpu::AllReduce);
  registry.Register("xla.cpu.tuple_all_to_all", &xla::cpu::TupleAllToAll);
  registry.Register("xla.cpu.collective_permute", &xla::cpu::CollectivePermute);
  registry.Register("xla.cpu.partition_id", &xla::cpu::PartitionId);
  registry.Register("xla.cpu.replica_id", &xla::cpu::ReplicaId);
}

}  // namespace cpu
}  // namespace xla
