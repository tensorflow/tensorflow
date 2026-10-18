/* Copyright 2026 The OpenXLA Authors.

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

// Multi-GPU integration tests for RaggedAllToAllThunk command-buffer Record().
// Requires two GPUs and CUDA 12.9+ driver/toolkit for CreateChildCommand /
// UpdateChildCommand support.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kNumUpdates = 4;
static constexpr int64_t kNumInputRows = 4;
static constexpr int64_t kNumRowElements = 1;
static constexpr int64_t kNumElements = kNumInputRows * kNumRowElements;
static constexpr int64_t kNumBuffers = 6;

static RaggedAllToAllConfig MakeConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  RaggedAllToAllConfig config;
  config.config.operand_element_type = {F32, F32, S64, S64, S64, S64};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.num_total_updates = kNumUpdates;
  config.num_input_rows = kNumInputRows;
  config.num_row_elements = kNumRowElements;
  config.one_shot_kernel_enabled = true;
  config.allow_fallback_to_nccl = false;
  return config;
}

static CollectiveThunk::Buffer MakeBuffer(const BufferAllocation& allocation,
                                          PrimitiveType element_type,
                                          int64_t element_count) {
  Shape shape = ShapeUtil::MakeShape(element_type, {element_count});
  ShapedSlice slice{BufferAllocation::Slice(
                        &allocation, 0, ShapeUtil::ByteSizeOfElements(shape)),
                    shape};
  return CollectiveThunk::Buffer{.element_count = element_count,
                                 .source_buffer = slice,
                                 .destination_buffer = slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
}

static std::vector<BufferAllocation> MakeThunkBufferAllocations() {
  std::vector<BufferAllocation> allocations;
  allocations.reserve(kNumBuffers);
  allocations.emplace_back(/*index=*/0, sizeof(float) * kNumElements,
                           /*color=*/0);
  allocations.emplace_back(/*index=*/1, sizeof(float) * kNumElements,
                           /*color=*/0);
  for (int i = 2; i < kNumBuffers; ++i) {
    allocations.emplace_back(i, sizeof(int64_t) * kNumUpdates, /*color=*/0);
  }
  return allocations;
}

static RaggedAllToAllThunk MakeThunk(
    const std::vector<BufferAllocation>& allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(kNumBuffers);
  buffers.push_back(MakeBuffer(allocations[0], F32, kNumElements));
  buffers.push_back(MakeBuffer(allocations[1], F32, kNumElements));
  for (int i = 2; i < kNumBuffers; ++i) {
    buffers.push_back(MakeBuffer(allocations[i], S64, kNumUpdates));
  }
  return RaggedAllToAllThunk(Thunk::ThunkInfo(), MakeConfig(),
                             std::move(buffers), kNumDevices);
}

using DeviceTestSlot = CollectiveThunkMultiGpuTestState;

static std::vector<int64_t> DeviceBufferSizes() {
  std::vector<int64_t> buffer_sizes;
  buffer_sizes.reserve(kNumBuffers);
  buffer_sizes.push_back(static_cast<int64_t>(sizeof(float)) * kNumElements);
  buffer_sizes.push_back(static_cast<int64_t>(sizeof(float)) * kNumElements);
  for (int i = 2; i < kNumBuffers; ++i) {
    buffer_sizes.push_back(static_cast<int64_t>(sizeof(int64_t)) * kNumUpdates);
  }
  return buffer_sizes;
}

static std::vector<BufferAllocation::Index> AllAllocationIndices() {
  std::vector<BufferAllocation::Index> indices;
  indices.reserve(kNumBuffers);
  for (int i = 0; i < kNumBuffers; ++i) {
    indices.push_back(i);
  }
  return indices;
}

static std::vector<float> InputValues(int device_ordinal, int phase) {
  std::vector<float> values(kNumElements);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(phase * 100 + device_ordinal * 10 + i);
  }
  return values;
}

static std::vector<int64_t> OutputOffsets(int device_ordinal) {
  int64_t base = 2 * device_ordinal;
  return {base, base + 1, base, base + 1};
}

static std::vector<float> ExpectedValues(int device_ordinal, int phase) {
  std::vector<float> rank0 = InputValues(/*device_ordinal=*/0, phase);
  std::vector<float> rank1 = InputValues(/*device_ordinal=*/1, phase);
  if (device_ordinal == 0) {
    return {rank0[0], rank0[1], rank1[0], rank1[1]};
  }
  return {rank0[2], rank0[3], rank1[2], rank1[3]};
}

static absl::Status WriteBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const std::vector<int64_t>& data) {
  ABSL_RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), data.size() * sizeof(int64_t)));
  return absl::OkStatus();
}

static absl::Status PrepareInputs(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> buffers,
    int device_ordinal, int phase) {
  ABSL_RETURN_IF_ERROR(
      FillDeviceBuffer(stream, buffers[0], InputValues(device_ordinal, phase)));
  ABSL_RETURN_IF_ERROR(FillDeviceBuffer(stream, buffers[1],
                                   std::vector<float>(kNumElements, -1.0f)));
  ABSL_RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[2], std::vector<int64_t>{0, 1, 2, 3}));
  ABSL_RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[3], std::vector<int64_t>{1, 1, 1, 1}));
  ABSL_RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[4], OutputOffsets(device_ordinal)));
  ABSL_RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[5], std::vector<int64_t>{1, 1, 1, 1}));
  return stream.BlockHostUntilDone();
}

static absl::Status VerifyOutput(se::Stream& stream,
                                 se::DeviceAddressBase buffer,
                                 int device_ordinal, int phase) {
  ABSL_ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadDeviceBuffer(stream, buffer, kNumElements));
  std::vector<float> expected = ExpectedValues(device_ordinal, phase);
  for (int i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    RaggedAllToAllThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  std::vector<int64_t> buffer_sizes = DeviceBufferSizes();
  return SetupCollectiveThunkDevice(device_ordinal, kNumDevices, buffer_sizes,
                                    thunk, device_assignment, slot);
}

static absl::Status RunExecuteOnStreamPhase(DeviceTestSlot& slot,
                                            RaggedAllToAllThunk& thunk,
                                            BufferAllocations& allocations,
                                            int device_ordinal, int phase) {
  ABSL_RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  ABSL_RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  return VerifyOutput(*slot.stream, slot.create_buffers[1], device_ordinal,
                      phase);
}

static absl::Status RunCreatePhase(DeviceTestSlot& slot,
                                   RaggedAllToAllThunk& thunk,
                                   BufferAllocations& allocations,
                                   int device_ordinal, int phase) {
  ABSL_RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  ABSL_RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  ABSL_RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  ABSL_RETURN_IF_ERROR(RecordCommandBufferCreate(slot, thunk, execute_params));
  ABSL_RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, slot.create_buffers[1], device_ordinal,
                      phase);
}

static absl::Status RunUpdatePhase(DeviceTestSlot& slot,
                                   RaggedAllToAllThunk& thunk,
                                   BufferAllocations& allocations,
                                   int device_ordinal, int phase) {
  ABSL_RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.update_buffers, device_ordinal, phase));

  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  ABSL_RETURN_IF_ERROR(RecordCommandBufferUpdate(slot, thunk, execute_params,
                                            AllAllocationIndices()));
  ABSL_RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, slot.update_buffers[1], device_ordinal,
                      phase);
}

TEST(RaggedAllToAllThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  std::vector<DeviceTestSlot> slots(kNumDevices);
  // Make sure allocations outlive the thunk.
  std::vector<std::unique_ptr<BufferAllocations>> allocs(kNumDevices);
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "ragged_execute", [&](int d) -> absl::Status {
        ABSL_RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        allocs[d] = std::make_unique<BufferAllocations>(
            MakeBufferAllocations(slots[d], slots[d].create_buffers));
        return RunExecuteOnStreamPhase(slots[d], thunk, *allocs[d], d,
                                       /*phase=*/1);
      }));
}

TEST(RaggedAllToAllThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  std::vector<DeviceTestSlot> slots(kNumDevices);
  // Make sure allocations outlive the thunk.
  std::vector<std::unique_ptr<BufferAllocations>> allocs(kNumDevices);
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "ragged_create", [&](int d) -> absl::Status {
        ABSL_RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        allocs[d] = std::make_unique<BufferAllocations>(
            MakeBufferAllocations(slots[d], slots[d].create_buffers));
        return RunCreatePhase(slots[d], thunk, *allocs[d], d, /*phase=*/1);
      }));
}

TEST(RaggedAllToAllThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  std::vector<DeviceTestSlot> slots(kNumDevices);
  // Make sure allocations outlive the thunk.
  std::vector<std::unique_ptr<BufferAllocations>> create_allocs(kNumDevices);
  std::vector<std::unique_ptr<BufferAllocations>> update_allocs(kNumDevices);
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "ragged_create", [&](int d) -> absl::Status {
        ABSL_RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        create_allocs[d] = std::make_unique<BufferAllocations>(
            MakeBufferAllocations(slots[d], slots[d].create_buffers));
        return RunCreatePhase(slots[d], thunk, *create_allocs[d], d,
                              /*phase=*/1);
      }));

  ASSERT_OK(RunOnDevices(kNumDevices, "ragged_update", [&](int d) {
    update_allocs[d] = std::make_unique<BufferAllocations>(
        MakeBufferAllocations(slots[d], slots[d].update_buffers));
    return RunUpdatePhase(slots[d], thunk, *update_allocs[d], d, /*phase=*/2);
  }));
}

TEST(RaggedAllToAllThunkMultiGpuTest,
     SingletonRaggedAllToAllDoesNotRequestBarrierOrSymmetricMemory) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  constexpr absl::string_view kModuleStr = R"hlo(
HloModule module, replica_count=2

ENTRY entry {
  input = bf16[4,16] parameter(0)
  output = bf16[4,16] parameter(1)
  input_offsets = s64[1] parameter(2)
  send_sizes = s64[1] parameter(3)
  output_offsets = s64[1] parameter(4)
  recv_sizes = s64[1] parameter(5)
  ROOT ra2a = bf16[4,16] ragged-all-to-all(
      input, output, input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0},{1}}
}
)hlo";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      true);
  debug_options
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(true);
  config.set_debug_options(debug_options);
  config.set_replica_count(kNumDevices);

  VerifiedHloModule module("test_module", config,
                           /*verifier_layout_sensitive=*/false,
                           /*allow_mixed_precision_in_hlo_verifier=*/false,
                           ShapeUtil::ByteSizeOfElements);
  ASSERT_OK(module.ParseHloStringAndVerifyModule(kModuleStr));

  const HloInstruction* root = module.entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kRaggedAllToAll);
  const auto* ra2a_instr =
      absl::down_cast<const HloRaggedAllToAllInstruction*>(root);

  std::vector<BufferAllocation> allocations;
  allocations.reserve(ra2a_instr->operand_count());
  for (int i = 0; i < ra2a_instr->operand_count(); ++i) {
    allocations.emplace_back(
        /*index=*/i, ShapeUtil::ByteSizeOf(ra2a_instr->operand(i)->shape()),
        /*color=*/0);
  }

  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(ra2a_instr->operand_count());
  for (int i = 0; i < ra2a_instr->operand_count(); ++i) {
    const Shape& shape = ra2a_instr->operand(i)->shape();
    int64_t byte_size = ShapeUtil::ByteSizeOf(shape);
    ShapedSlice slice{BufferAllocation::Slice(&allocations[i], 0, byte_size),
                      shape};
    buffers.push_back(CollectiveThunk::Buffer{
        /*element_count=*/ShapeUtil::ElementsIn(shape),
        /*source_buffer=*/slice,
        /*destination_buffer=*/slice,
        /*source_memory_space=*/0,
        /*destination_memory_space=*/0,
    });
  }

  RaggedAllToAllThunk thunk(Thunk::ThunkInfo(), ra2a_instr, std::move(buffers),
                            /*p2p_memcpy_enabled=*/true,
                            /*devices_per_host=*/kNumDevices);

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  GpuExecutableRunOptions::DeviceIdMap id_map;
  for (int i = 0; i < kNumDevices; ++i) {
    id_map[LocalDeviceId(i)] = GlobalDeviceId(i);
  }

  ASSERT_OK(RunOnDevices(
      kNumDevices, "singleton_ra2a_prepare", [&](int d) -> absl::Status {
        se::StreamExecutor* executor = GetGpuExecutor(d);
        ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                         executor->CreateStream());
        se::StreamExecutorAddressAllocator allocator(executor);

        GpuExecutableRunOptions gpu_options;
        gpu_options.set_gpu_global_device_ids(id_map);

        ServiceExecutableRunOptions run_options;
        run_options.mutable_run_options()->set_stream(stream.get());
        run_options.mutable_run_options()->set_device_assignment(
            &device_assignment);
        run_options.mutable_run_options()->set_gpu_executable_run_options(
            &gpu_options);
        run_options.mutable_run_options()->set_local_device_count(kNumDevices);

        ABSL_ASSIGN_OR_RETURN(
            CollectiveParams collective_params,
            CollectiveParams::Create(run_options, /*async_streams=*/{},
                                     LocalDeviceId(d)));

        std::vector<se::DeviceAddressBase> allocated_buffers(
            allocations.size());
        BufferAllocations buffer_allocs(allocated_buffers, d, &allocator);
        CollectiveCliqueRequests clique_requests;
        CollectiveMemoryRequests memory_requests(buffer_allocs);
        Thunk::PrepareParams prepare_params{&collective_params,
                                            &clique_requests, &memory_requests,
                                            executor, &buffer_allocs};

        ABSL_RETURN_IF_ERROR(thunk.Prepare(prepare_params));

        std::vector<CollectiveCliqueRequests::CliqueRequest> requested_cliques =
            clique_requests.OrderedRequestedCliques();
        EXPECT_EQ(requested_cliques.size(), 1);
        if (!requested_cliques.empty()) {
          EXPECT_EQ(requested_cliques[0].key.num_devices(), 1);
          EXPECT_FALSE(requested_cliques[0].use_cross_device_barrier_requested);
        }
        EXPECT_EQ(memory_requests.symmetric_size(), 0);
        return absl::OkStatus();
      }));
}

}  // namespace
}  // namespace xla::gpu
