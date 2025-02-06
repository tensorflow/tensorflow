/* Copyright 2025 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/all_gather_thunk.h"
#include "xla/backends/cpu/runtime/all_reduce_thunk.h"
#include "xla/backends/cpu/runtime/all_to_all_thunk.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/collective_permute_thunk.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/convolution_thunk.h"
#include "xla/backends/cpu/runtime/convolution_thunk_test_util.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/dot_thunk.h"
#include "xla/backends/cpu/runtime/fft_thunk.h"
#include "xla/backends/cpu/runtime/infeed_thunk.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/logical_id_thunk.h"
#include "xla/backends/cpu/runtime/outfeed_thunk.h"
#include "xla/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "xla/backends/cpu/runtime/resource_use.h"
#include "xla/backends/cpu/runtime/rng_state_thunk.h"
#include "xla/backends/cpu/runtime/serdes_base.h"
#include "xla/backends/cpu/runtime/sort_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/backends/cpu/runtime/topk_thunk.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_convolution_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

// Thunk sequence serdes test base.
// This is independent of the serialization format.
template <typename T>
class ThunkSequenceSerdesTest : public ::testing::Test {
 protected:
  explicit ThunkSequenceSerdesTest() = default;

  absl::StatusOr<ThunkSequence> CreateThunkSequenceFromAllThunkTypes() {
    // NOTE create buffer allocations using thunk_testlib
    ThunkSequence thunk_sequence;

    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateAllGatherThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateAllReduceThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateAllToAllThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateReduceScatterThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateCallThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateCollectivePermuteThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateCopyThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateConditionalThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateCustomCallThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateDotThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateFftThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateInfeedThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateOutfeedThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreatePartitionIdThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateReplicaIdThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateRngGetAndUpdateStateThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateTopKThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateWhileThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateXnnDotThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateXnnConvolutionThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateKernelThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateConvolutionThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateSortThunk());
    return thunk_sequence;
  }

  absl::StatusOr<std::string> Serialize(const ThunkSequence& thunk_sequence) {
    return thunk_sequence_serdes_->Serialize(thunk_sequence);
  }

  absl::StatusOr<std::unique_ptr<ThunkSequence>> Deserialize(
      const std::string& serialized) {
    return thunk_sequence_serdes_->Deserialize(serialized);
  }

  bool VerifyThunkSequenceEquality(const ThunkSequence& thunk_sequence_1,
                                   const ThunkSequence& thunk_sequence_2) {
    if (thunk_sequence_1.size() != thunk_sequence_2.size()) {
      return false;
    }
    for (int i = 0; i < thunk_sequence_1.size(); ++i) {
      if (!VerifyThunkEquality(*thunk_sequence_1[i], *thunk_sequence_2[i])) {
        return false;
      }
    }
    return true;
  }

 public:
  void SetUp() override {
    // HACK(basioli): allocations are created on thunk creation and are pushed
    // back into this vector. If we don't reserve enough space, reallocation
    // will get triggered which will invalidate the pointers to the allocations
    // owned by the thunks.
    buffer_allocations_.reserve(10000);
    thunk_sequence_serdes_ = std::make_unique<T>(&buffer_allocations_);
  }

 private:
  void AddBufferAllocations(const size_t no_of_allocations_to_add) {
    for (size_t i = 0; i < no_of_allocations_to_add; ++i) {
      literals_.push_back(LiteralUtil::CreateFull<float>({2, 4}, 0.0));
      buffer_allocations_.push_back(
          CreateBufferAllocation(buffer_allocations_.size(), literals_.back()));
    }
  }
  // Thunk creation helper functions.
  absl::StatusOr<std::unique_ptr<Thunk>> CreateAllGatherThunk() {
    AddBufferAllocations(2);

    return AllGatherThunk::Create(
        Thunk::Info(),
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        });
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateAllReduceThunk() {
    AddBufferAllocations(2);

    return AllReduceThunk::Create(
        Thunk::Info(), ReductionKind::SUM,
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        },
        /*single_replica=*/false);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateAllToAllThunk() {
    AddBufferAllocations(2);

    return AllToAllThunk::Create(
        Thunk::Info(),
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        });
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateReduceScatterThunk() {
    AddBufferAllocations(2);

    return ReduceScatterThunk::Create(
        Thunk::Info(), ReductionKind::SUM,
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        });
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateCallThunk() {
    ThunkSequence called_sequence;
    TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(), CreateAllGatherThunk());
    TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(), CreateAllReduceThunk());
    TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(), CreateAllToAllThunk());
    return CallThunk::Create(Thunk::Info(),
                             /*called_sequence=*/
                             std::move(called_sequence));
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateCollectivePermuteThunk() {
    AddBufferAllocations(2);

    return CollectivePermuteThunk::Create(
        Thunk::Info(),
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        },
        {{0, 0}});
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateCopyThunk() {
    AddBufferAllocations(2);

    return CopyThunk::Create(
        Thunk::Info(),
        /*src_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*src_shape=*/literals_[buffer_allocations_.size() - 2].shape(),
        /*dst_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*dst_shape=*/literals_[buffer_allocations_.size() - 1].shape());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateConditionalThunk() {
    std::vector<ThunkSequence> branch_sequences;
    for (int i = 0; i < 2; ++i) {
      ThunkSequence called_sequence;
      TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(),
                          CreateAllGatherThunk());
      TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(),
                          CreateAllReduceThunk());
      TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(),
                          CreateAllToAllThunk());
      branch_sequences.push_back(std::move(called_sequence));
    }

    AddBufferAllocations(1);

    return ConditionalThunk::Create(
        Thunk::Info(),
        /*branch_index_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        std::move(branch_sequences));
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateCustomCallThunk() {
    AddBufferAllocations(2);

    return CustomCallThunk::Create(
        Thunk::Info(), "custom_call_thunk_test",
        {
            /*arguments_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*arguments_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*results_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*results_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
            /*is_tuple_result=*/false,
        },
        /*backend_config=*/"", CustomCallApiVersion::API_VERSION_ORIGINAL);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateDotThunk() {
    AddBufferAllocations(3);
    DotDimensionNumbers dot_dimensions;
    dot_dimensions.add_lhs_contracting_dimensions(1);
    dot_dimensions.add_rhs_contracting_dimensions(0);
    return DotThunk::Create(
        Thunk::Info(),
        /*dot_dimensions=*/dot_dimensions,
        /*lhs_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 3]),
        /*lhs_shape=*/literals_[buffer_allocations_.size() - 3].shape(),
        /*rhs_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*rhs_shape=*/literals_[buffer_allocations_.size() - 2].shape(),
        /*out_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*out_shape=*/literals_[buffer_allocations_.size() - 1].shape());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateFftThunk() {
    AddBufferAllocations(2);

    return FftThunk::Create(
        Thunk::Info(),
        /*is_multi_thread_eigen=*/false, /*fft_type=*/0,
        /*fft_length*/ {1},
        /*input_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*input_shape=*/literals_[buffer_allocations_.size() - 2].shape(),
        /*output_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*output_shape=*/literals_[buffer_allocations_.size() - 1].shape());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateInfeedThunk() {
    AddBufferAllocations(2);

    return InfeedThunk::Create(
        Thunk::Info(),
        /*infeed_buffers=*/
        {{
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 2]),
             literals_[buffer_allocations_.size() - 2].shape(),
         },
         {
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 1]),
             literals_[buffer_allocations_.size() - 1].shape(),
         }},
        InfeedThunk::InfeedResources());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateOutfeedThunk() {
    AddBufferAllocations(2);

    return OutfeedThunk::Create(
        Thunk::Info(),
        /*outfeed_buffers=*/
        {{
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 2]),
             literals_[buffer_allocations_.size() - 2].shape(),
         },
         {
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 1]),
             literals_[buffer_allocations_.size() - 1].shape(),
         }},
        OutfeedThunk::OutfeedResources());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreatePartitionIdThunk() {
    AddBufferAllocations(1);
    return PartitionIdThunk::Create(
        Thunk::Info(),
        /*logical_id_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]));
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateReplicaIdThunk() {
    AddBufferAllocations(1);
    return ReplicaIdThunk::Create(
        Thunk::Info(),
        /*logical_id_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]));
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateRngGetAndUpdateStateThunk() {
    AddBufferAllocations(1);
    return RngGetAndUpdateStateThunk::Create(
        Thunk::Info(),
        /*state_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*delta=*/0);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateTopKThunk() {
    AddBufferAllocations(3);
    return TopKThunk::Create(
        Thunk::Info(),
        /*values=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 3]),
        /*output=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*indices=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*batch_size=*/1,
        /*input_size=*/1,
        /*k=*/2

    );
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateWhileThunk() {
    ThunkSequence cond_sequence;
    TF_ASSIGN_OR_RETURN(cond_sequence.emplace_back(), CreateAllGatherThunk());
    ThunkSequence body_sequence;
    TF_ASSIGN_OR_RETURN(body_sequence.emplace_back(), CreateAllGatherThunk());
    TF_ASSIGN_OR_RETURN(body_sequence.emplace_back(), CreateAllReduceThunk());
    TF_ASSIGN_OR_RETURN(body_sequence.emplace_back(), CreateAllToAllThunk());

    AddBufferAllocations(1);
    return WhileThunk::Create(
        Thunk::Info(),
        /*cond_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*cond_sequence=*/std::move(cond_sequence),
        /*body_sequence=*/std::move(body_sequence),
        /*trip_count=*/1);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateXnnDotThunk() {
    AddBufferAllocations(3);
    DotDimensionNumbers dot_dimensions;
    dot_dimensions.add_lhs_contracting_dimensions(1);
    dot_dimensions.add_rhs_contracting_dimensions(0);
    return XnnDotThunk::Create(
        XnnFusionThunk::Options(), Thunk::Info(),
        /*dot_dimensions=*/dot_dimensions,
        /*lhs_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 3]),
        /*lhs_shape=*/literals_[buffer_allocations_.size() - 3].shape(),
        /*rhs_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*rhs_shape=*/literals_[buffer_allocations_.size() - 2].shape(),
        /*out_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*out_shape=*/literals_[buffer_allocations_.size() - 1].shape());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateXnnConvolutionThunk() {
    std::vector<int64_t> input_dims = {1, 8, 8, 16};
    std::vector<int64_t> kernel_dims = {32, 1, 1, 16};
    std::vector<int64_t> output_dims = {1, 8, 8, 32};

    // Convolution rank inferred from the input dimensions.
    int convolution_rank = input_dims.size() - 2;

    // Convolution parameters.
    ConvolutionDimensionNumbers conv_dims =
        MakeConvolutionDimensionNumbers(convolution_rank);
    Window window = MakeWindow(convolution_rank);

    // Adjust kernel dimensions for XNNPACK.
    conv_dims.set_kernel_input_feature_dimension(3);
    conv_dims.set_kernel_output_feature_dimension(0);
    conv_dims.set_kernel_spatial_dimensions(0, 1);
    conv_dims.set_kernel_spatial_dimensions(1, 2);

    // Actual data.
    literals_.push_back(
        LiteralUtil::CreateFull<float>(input_dims, 0.0));  // input
    literals_.push_back(
        LiteralUtil::CreateFull<float>(kernel_dims, 0.0));  // kernel
    literals_.push_back(
        LiteralUtil::CreateFull<float>(output_dims, 0.0));  // output

    buffer_allocations_.push_back(CreateBufferAllocation(
        buffer_allocations_.size(), literals_[literals_.size() - 3]));
    buffer_allocations_.push_back(CreateBufferAllocation(
        buffer_allocations_.size(), literals_[literals_.size() - 2]));
    buffer_allocations_.push_back(CreateBufferAllocation(
        buffer_allocations_.size(), literals_[literals_.size() - 1]));

    return XnnConvolutionThunk::Create(
        XnnFusionThunk::Options(), Thunk::Info(),
        /*input_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 3]),
        /*input_shape=*/literals_[buffer_allocations_.size() - 3].shape(),
        /*kernel_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*kernel_shape=*/literals_[buffer_allocations_.size() - 2].shape(),
        /*output_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*output_shape=*/literals_[buffer_allocations_.size() - 1].shape(),
        conv_dims, window, /*feature_group_count=*/1);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateKernelThunk() {
    AddBufferAllocations(2);
    return KernelThunk::Create(
        Thunk::Info(),
        /*arguments_buffers=*/
        {CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2])},
        /*results_buffers=*/
        {CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1])},
        /*kernel_name=*/"test",
        /*thread_dim=*/se::ThreadDim(1),
        /*invariant_arguments=*/{{0}},
        /*min_alignment=*/8);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateConvolutionThunk() {
    std::vector<int64_t> input_dims = {2, 2, 1};
    std::vector<int64_t> kernel_dims = {1, 1, 1};
    std::vector<int64_t> output_dims = {2, 2, 1};
    // Convolution rank inferred from the input dimensions.
    int convolution_rank = input_dims.size() - 2;

    // Convolution parameters.
    ConvolutionDimensionNumbers dnums =
        MakeConvolutionDimensionNumbers(convolution_rank);
    Window window = MakeWindow(convolution_rank);

    // Actual data.
    literals_.push_back(
        LiteralUtil::CreateFull<float>(input_dims, 0.0));  // input
    literals_.push_back(
        LiteralUtil::CreateFull<float>(kernel_dims, 0.0));  // kernel
    literals_.push_back(
        LiteralUtil::CreateFull<float>(output_dims, 0.0));  // output

    buffer_allocations_.push_back(CreateBufferAllocation(
        buffer_allocations_.size(), literals_[literals_.size() - 3]));
    buffer_allocations_.push_back(CreateBufferAllocation(
        buffer_allocations_.size(), literals_[literals_.size() - 2]));
    buffer_allocations_.push_back(CreateBufferAllocation(
        buffer_allocations_.size(), literals_[literals_.size() - 1]));

    ConvolutionThunk::Options options;

    return ConvolutionThunk::Create(
        {"convolution"}, options,
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 3]),
        literals_[buffer_allocations_.size() - 3].shape(),
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        literals_[buffer_allocations_.size() - 2].shape(),
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        literals_[buffer_allocations_.size() - 1].shape(), dnums, window,
        /*feature_group_count=*/1);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateSortThunk() {
    AddBufferAllocations(2);
    return SortThunk::Create(
        Thunk::Info(),
        /*inputs=*/
        {{
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 2]),
             literals_[buffer_allocations_.size() - 2].shape(),
         },
         {
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 1]),
             literals_[buffer_allocations_.size() - 1].shape(),
         }},
        /*dimension=*/0,
        /*is_stable=*/false,
        /*comparator_name=*/"test",
        /*direction=*/SortThunk::SortDirection::kAscending);
  }

  bool VerifySliceEquality(const BufferAllocation::Slice& slice_1,
                           const BufferAllocation::Slice& slice_2) {
    return slice_1.offset() == slice_2.offset() &&
           slice_1.size() == slice_2.size() &&
           slice_1.allocation() == slice_2.allocation();
  }

  bool VerifySliceShapeEquality(const BufferAllocation::Slice& slice_1,
                                const Shape& shape_1,
                                const BufferAllocation::Slice& slice_2,
                                const Shape& shape_2) {
    return VerifySliceEquality(slice_1, slice_2) &&
           ShapeUtil::Equal(shape_1, shape_2);
  }

  bool VerifyShapesEquality(absl::Span<const Shape> shapes_1,
                            absl::Span<const Shape> shapes_2) {
    if (shapes_1.size() != shapes_2.size()) {
      return false;
    }

    for (size_t i = 0; i < shapes_1.size(); ++i) {
      if (!ShapeUtil::Equal(shapes_1[i], shapes_2[i])) {
        return false;
      }
    }
    return true;
  }

  bool VerifySlicesEquality(
      absl::Span<const BufferAllocation::Slice> slices_1,
      absl::Span<const BufferAllocation::Slice> slices_2) {
    if (slices_1.size() != slices_2.size()) {
      return false;
    }

    for (size_t i = 0; i < slices_1.size(); ++i) {
      if (!VerifySliceEquality(slices_1[i], slices_2[i])) {
        return false;
      }
    }
    return true;
  }

  bool VerifyResourceEquality(const std::shared_ptr<Resource>& resource_1,
                              const std::shared_ptr<Resource>& resource_2) {
    if ((resource_1 == nullptr) ^ (resource_2 == nullptr)) {
      return false;
    }

    if (resource_1 && resource_1->kind() != resource_2->kind()) {
      return false;
    }

    return true;
  }

  bool VerifyCollectiveThunkEquality(const CollectiveThunk& thunk_1,
                                     const CollectiveThunk& thunk_2) {
    const auto& op_params_1 = thunk_1.op_params();
    const auto& op_params_2 = thunk_2.op_params();

    bool are_replica_groups_equal = absl::c_equal(
        op_params_1.group, op_params_2.group,
        [](const ReplicaGroup& group_1, const ReplicaGroup& group_2) {
          return absl::c_equal(group_1.replica_ids(), group_2.replica_ids());
        });

    if (op_params_1.op_id != op_params_2.op_id ||
        op_params_1.has_channel_id != op_params_2.has_channel_id ||
        op_params_1.use_global_device_ids !=
            op_params_2.use_global_device_ids ||
        !are_replica_groups_equal) {
      return false;
    }

    const auto& op_buffers_1 = thunk_1.op_buffers();
    const auto& op_buffers_2 = thunk_2.op_buffers();

    if (!VerifySlicesEquality(op_buffers_1.source_buffers,
                              op_buffers_2.source_buffers) ||
        !VerifySlicesEquality(op_buffers_1.destination_buffers,
                              op_buffers_2.destination_buffers) ||
        !VerifyShapesEquality(op_buffers_1.source_shapes,
                              op_buffers_2.source_shapes) ||
        !VerifyShapesEquality(op_buffers_1.destination_shapes,
                              op_buffers_2.destination_shapes)) {
      return false;
    }

    const auto& op_resources_1 = thunk_1.op_resources();
    const auto& op_resources_2 = thunk_2.op_resources();

    if (!VerifyResourceEquality(op_resources_1.communicator_resource,
                                op_resources_2.communicator_resource)) {
      return false;
    }

    return true;
  }

  bool VerifyAllGatherThunkEquality(const AllGatherThunk& thunk_1,
                                    const AllGatherThunk& thunk_2) {
    return VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyAllReduceThunkEquality(const AllReduceThunk& thunk_1,
                                    const AllReduceThunk& thunk_2) {
    return thunk_1.single_replica() == thunk_2.single_replica() &&
           thunk_1.reduction_kind() == thunk_2.reduction_kind() &&
           VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyAllToAllThunkEquality(const AllToAllThunk& thunk_1,
                                   const AllToAllThunk& thunk_2) {
    return VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyCallThunkEquality(const CallThunk& thunk_1,
                               const CallThunk& thunk_2) {
    return absl::c_equal(thunk_1.called_executor().thunk_sequence(),
                         thunk_2.called_executor().thunk_sequence(),
                         [this](const std::unique_ptr<Thunk>& thunk_1,
                                const std::unique_ptr<Thunk>& thunk_2) {
                           return VerifyThunkEquality(*thunk_1, *thunk_2);
                         });
  }

  bool VerifyCollectivePermuteThunkEquality(
      const CollectivePermuteThunk& thunk_1,
      const CollectivePermuteThunk& thunk_2) {
    return absl::c_equal(thunk_1.source_target_pairs(),
                         thunk_2.source_target_pairs()) &&
           VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyCopyThunkEquality(const CopyThunk& thunk_1,
                               const CopyThunk& thunk_2) {
    return VerifySliceShapeEquality(thunk_1.src_buffer(), thunk_1.src_shape(),
                                    thunk_2.src_buffer(),
                                    thunk_2.src_shape()) &&
           VerifySliceShapeEquality(thunk_1.dst_buffer(), thunk_1.dst_shape(),
                                    thunk_2.dst_buffer(), thunk_2.dst_shape());
  }

  bool VerifyConditionalThunkEquality(const ConditionalThunk& thunk_1,
                                      const ConditionalThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.branch_index_buffer(),
                               thunk_2.branch_index_buffer()) &&
           absl::c_equal(thunk_1.branch_executors(), thunk_2.branch_executors(),
                         [this](const ThunkExecutor& executor_1,
                                const ThunkExecutor& executor_2) {
                           return absl::c_equal(
                               executor_1.thunk_sequence(),
                               executor_2.thunk_sequence(),
                               [this](const std::unique_ptr<Thunk>& thunk_1,
                                      const std::unique_ptr<Thunk>& thunk_2) {
                                 return VerifyThunkEquality(*thunk_1, *thunk_2);
                               });
                         });
  }

  bool VerifyCustomCallThunkEquality(const CustomCallThunk& thunk_1,
                                     const CustomCallThunk& thunk_2) {
    bool are_op_buffers_equal =
        absl::c_equal(thunk_1.op_buffers().arguments_buffers,
                      thunk_2.op_buffers().arguments_buffers,
                      [this](const BufferAllocation::Slice& slice_1,
                             const BufferAllocation::Slice& slice_2) {
                        return VerifySliceEquality(slice_1, slice_2);
                      });

    are_op_buffers_equal &=
        absl::c_equal(thunk_1.op_buffers().results_buffers,
                      thunk_2.op_buffers().results_buffers,
                      [this](const BufferAllocation::Slice& slice_1,
                             const BufferAllocation::Slice& slice_2) {
                        return VerifySliceEquality(slice_1, slice_2);
                      });

    are_op_buffers_equal &=
        absl::c_equal(thunk_1.op_buffers().arguments_shapes,
                      thunk_2.op_buffers().arguments_shapes,
                      [](const Shape& shape_1, const Shape& shape_2) {
                        return ShapeUtil::Equal(shape_1, shape_2);
                      });

    are_op_buffers_equal &=
        absl::c_equal(thunk_1.op_buffers().results_shapes,
                      thunk_2.op_buffers().results_shapes,
                      [](const Shape& shape_1, const Shape& shape_2) {
                        return ShapeUtil::Equal(shape_1, shape_2);
                      });
    return thunk_1.target_name() == thunk_2.target_name() &&
           thunk_1.api_version() == thunk_2.api_version() &&
           thunk_1.backend_config() == thunk_2.backend_config() &&
           are_op_buffers_equal;
  }

  bool VerifyDotThunkEquality(const DotThunk& thunk_1,
                              const DotThunk& thunk_2) {
    bool are_dot_dimensions_equal =
        absl::c_equal(thunk_1.dot_dimensions().lhs_batch_dimensions(),
                      thunk_2.dot_dimensions().lhs_batch_dimensions()) &&
        absl::c_equal(thunk_1.dot_dimensions().rhs_batch_dimensions(),
                      thunk_2.dot_dimensions().rhs_batch_dimensions()) &&
        absl::c_equal(thunk_1.dot_dimensions().lhs_contracting_dimensions(),
                      thunk_2.dot_dimensions().lhs_contracting_dimensions()) &&
        absl::c_equal(thunk_1.dot_dimensions().rhs_contracting_dimensions(),
                      thunk_2.dot_dimensions().rhs_contracting_dimensions());

    return are_dot_dimensions_equal &&
           VerifySliceShapeEquality(thunk_1.dot_slices().lhs_buffer,
                                    thunk_1.dot_slices().lhs_shape,
                                    thunk_2.dot_slices().lhs_buffer,
                                    thunk_2.dot_slices().lhs_shape) &&
           VerifySliceShapeEquality(thunk_1.dot_slices().rhs_buffer,
                                    thunk_1.dot_slices().rhs_shape,
                                    thunk_2.dot_slices().rhs_buffer,
                                    thunk_2.dot_slices().rhs_shape) &&
           VerifySliceShapeEquality(
               thunk_1.dot_slices().out_buffer, thunk_1.dot_slices().out_shape,
               thunk_2.dot_slices().out_buffer, thunk_2.dot_slices().out_shape);
  }

  bool VerifyFftThunkEquality(const FftThunk& thunk_1,
                              const FftThunk& thunk_2) {
    return thunk_1.is_multi_thread_eigen() == thunk_2.is_multi_thread_eigen() &&
           absl::c_equal(thunk_1.fft_length(), thunk_2.fft_length()) &&
           VerifySliceShapeEquality(
               thunk_1.input_buffer(), thunk_1.input_shape(),
               thunk_2.input_buffer(), thunk_2.input_shape()) &&
           VerifySliceShapeEquality(
               thunk_1.output_buffer(), thunk_1.output_shape(),
               thunk_2.output_buffer(), thunk_2.output_shape()) &&
           thunk_1.fft_type() == thunk_2.fft_type();
  }

  bool VerifyInfeedThunkEquality(const InfeedThunk& thunk_1,
                                 const InfeedThunk& thunk_2) {
    InfeedThunk::InfeedResources infeed_resources_1 =
        thunk_1.infeed_resources();
    InfeedThunk::InfeedResources infeed_resources_2 =
        thunk_2.infeed_resources();

    if (!VerifyResourceEquality(infeed_resources_1.consume_token,
                                infeed_resources_2.consume_token) ||
        !VerifyResourceEquality(infeed_resources_1.produce_token,
                                infeed_resources_2.produce_token)) {
      return false;
    }

    return absl::c_equal(thunk_1.infeed_buffers(), thunk_2.infeed_buffers(),
                         [this](const InfeedThunk::InfeedBuffer& buffer_1,
                                const InfeedThunk::InfeedBuffer& buffer_2) {
                           return VerifySliceShapeEquality(
                               buffer_1.slice, buffer_1.shape, buffer_2.slice,
                               buffer_2.shape);
                         });
  }

  bool VerifyOutfeedThunkEquality(const OutfeedThunk& thunk_1,
                                  const OutfeedThunk& thunk_2) {
    OutfeedThunk::OutfeedResources outfeed_resources_1 =
        thunk_1.outfeed_resources();
    OutfeedThunk::OutfeedResources outfeed_resources_2 =
        thunk_2.outfeed_resources();

    if (!VerifyResourceEquality(outfeed_resources_1.consume_token,
                                outfeed_resources_2.consume_token) ||
        !VerifyResourceEquality(outfeed_resources_1.produce_token,
                                outfeed_resources_2.produce_token)) {
      return false;
    }

    return absl::c_equal(thunk_1.outfeed_buffers(), thunk_2.outfeed_buffers(),
                         [this](const OutfeedThunk::OutfeedBuffer& buffer_1,
                                const OutfeedThunk::OutfeedBuffer& buffer_2) {
                           return VerifySliceShapeEquality(
                               buffer_1.slice, buffer_1.shape, buffer_2.slice,
                               buffer_2.shape);
                         });
  }

  bool VerifyPartitionIdThunkEquality(const PartitionIdThunk& thunk_1,
                                      const PartitionIdThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.logical_id_buffer(),
                               thunk_2.logical_id_buffer());
  }

  bool VerifyReplicaIdThunkEquality(const ReplicaIdThunk& thunk_1,
                                    const ReplicaIdThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.logical_id_buffer(),
                               thunk_2.logical_id_buffer());
  }

  bool VerifyRngGetAndUpdateStateThunkEquality(
      const RngGetAndUpdateStateThunk& thunk_1,
      const RngGetAndUpdateStateThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.state_buffer(),
                               thunk_2.state_buffer()) &&
           thunk_1.delta() == thunk_2.delta();
  }

  bool VerifySortThunkEquality(const SortThunk& thunk_1,
                               const SortThunk& thunk_2) {
    return thunk_1.comparator_name() == thunk_2.comparator_name() &&
           thunk_1.dimension() == thunk_2.dimension() &&
           thunk_1.is_stable() == thunk_2.is_stable() &&
           thunk_1.has_less_than() == thunk_2.has_less_than() &&
           thunk_1.direction() == thunk_2.direction() &&
           absl::c_equal(thunk_1.inputs(), thunk_2.inputs(),
                         [this](const SortThunk::Input& input_1,
                                const SortThunk::Input& input_2) {
                           return VerifySliceShapeEquality(
                               input_1.slice, input_1.shape, input_2.slice,
                               input_2.shape);
                         });
  }

  bool VerifyTopKThunkEquality(const TopKThunk& thunk_1,
                               const TopKThunk& thunk_2) {
    return thunk_1.batch_size() == thunk_2.batch_size() &&
           thunk_1.k() == thunk_2.k() &&
           thunk_1.input_size() == thunk_2.input_size() &&
           VerifySliceEquality(thunk_1.values_buffer(),
                               thunk_2.values_buffer()) &&
           VerifySliceEquality(thunk_1.output_buffer(),
                               thunk_2.output_buffer()) &&
           VerifySliceEquality(thunk_1.indices_buffer(),
                               thunk_2.indices_buffer());
  }
  bool VerifyWhileThunkEquality(const WhileThunk& thunk_1,
                                const WhileThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.cond_buffer(), thunk_2.cond_buffer()) &&
           absl::c_equal(thunk_1.cond_executor().thunk_sequence(),
                         thunk_2.cond_executor().thunk_sequence(),
                         [this](const std::unique_ptr<Thunk>& thunk_1,
                                const std::unique_ptr<Thunk>& thunk_2) {
                           return VerifyThunkEquality(*thunk_1, *thunk_2);
                         }) &&
           absl::c_equal(thunk_1.body_executor().thunk_sequence(),
                         thunk_2.body_executor().thunk_sequence(),
                         [this](const std::unique_ptr<Thunk>& thunk_1,
                                const std::unique_ptr<Thunk>& thunk_2) {
                           return VerifyThunkEquality(*thunk_1, *thunk_2);
                         }) &&
           thunk_1.trip_count() == thunk_2.trip_count();
  }

  bool VerifyXnnDotThunkEquality(const XnnDotThunk& thunk_1,
                                 const XnnDotThunk& thunk_2) {
    const bool are_dot_dimensions_equal =
        absl::c_equal(thunk_1.dot_dimensions().lhs_batch_dimensions(),
                      thunk_2.dot_dimensions().lhs_batch_dimensions()) &&
        absl::c_equal(thunk_1.dot_dimensions().rhs_batch_dimensions(),
                      thunk_2.dot_dimensions().rhs_batch_dimensions()) &&
        absl::c_equal(thunk_1.dot_dimensions().lhs_contracting_dimensions(),
                      thunk_2.dot_dimensions().lhs_contracting_dimensions()) &&
        absl::c_equal(thunk_1.dot_dimensions().rhs_contracting_dimensions(),
                      thunk_2.dot_dimensions().rhs_contracting_dimensions());

    const bool are_options_equal =
        thunk_1.options().use_threadpool == thunk_2.options().use_threadpool;

    return are_options_equal && are_dot_dimensions_equal &&
           VerifySliceShapeEquality(thunk_1.dot_slices().lhs_buffer,
                                    thunk_1.dot_slices().lhs_shape,
                                    thunk_2.dot_slices().lhs_buffer,
                                    thunk_2.dot_slices().lhs_shape) &&
           VerifySliceShapeEquality(thunk_1.dot_slices().rhs_buffer,
                                    thunk_1.dot_slices().rhs_shape,
                                    thunk_2.dot_slices().rhs_buffer,
                                    thunk_2.dot_slices().rhs_shape) &&
           VerifySliceShapeEquality(
               thunk_1.dot_slices().out_buffer, thunk_1.dot_slices().out_shape,
               thunk_2.dot_slices().out_buffer, thunk_2.dot_slices().out_shape);
  }

  bool VerifyXnnConvolutionThunkEquality(const XnnConvolutionThunk& thunk_1,
                                         const XnnConvolutionThunk& thunk_2) {
    const bool are_dnums_equal =
        absl::c_equal(thunk_1.dnums().input_spatial_dimensions(),
                      thunk_2.dnums().input_spatial_dimensions()) &&
        absl::c_equal(thunk_1.dnums().kernel_spatial_dimensions(),
                      thunk_2.dnums().kernel_spatial_dimensions()) &&
        absl::c_equal(thunk_1.dnums().output_spatial_dimensions(),
                      thunk_2.dnums().output_spatial_dimensions()) &&
        thunk_1.dnums().input_batch_dimension() ==
            thunk_2.dnums().input_batch_dimension() &&
        thunk_1.dnums().input_feature_dimension() ==
            thunk_2.dnums().input_feature_dimension() &&
        thunk_1.dnums().kernel_input_feature_dimension() ==
            thunk_2.dnums().kernel_input_feature_dimension() &&
        thunk_1.dnums().kernel_output_feature_dimension() ==
            thunk_2.dnums().kernel_output_feature_dimension() &&
        thunk_1.dnums().output_batch_dimension() ==
            thunk_2.dnums().output_batch_dimension() &&
        thunk_1.dnums().output_feature_dimension() ==
            thunk_2.dnums().output_feature_dimension();

    const bool are_options_equal =
        thunk_1.options().use_threadpool == thunk_2.options().use_threadpool;

    const bool are_windows_equal = absl::c_equal(
        thunk_1.window().dimensions(), thunk_2.window().dimensions(),
        [](const WindowDimension& window_dimension_1,
           const WindowDimension& window_dimension_2) {
          return window_dimension_1.size() == window_dimension_2.size() &&
                 window_dimension_1.stride() == window_dimension_2.stride() &&
                 window_dimension_1.padding_low() ==
                     window_dimension_2.padding_low() &&
                 window_dimension_1.padding_high() ==
                     window_dimension_2.padding_high() &&
                 window_dimension_1.window_dilation() ==
                     window_dimension_2.window_dilation() &&
                 window_dimension_1.base_dilation() ==
                     window_dimension_2.base_dilation() &&
                 window_dimension_1.window_reversal() ==
                     window_dimension_2.window_reversal();
        });

    return are_dnums_equal && are_windows_equal && are_options_equal &&
           thunk_1.feature_group_count() == thunk_2.feature_group_count() &&
           VerifySliceShapeEquality(thunk_1.convolution_slices().input_buffer,
                                    thunk_1.convolution_slices().input_shape,
                                    thunk_2.convolution_slices().input_buffer,
                                    thunk_2.convolution_slices().input_shape);
    VerifySliceShapeEquality(thunk_1.convolution_slices().kernel_buffer,
                             thunk_1.convolution_slices().kernel_shape,
                             thunk_2.convolution_slices().kernel_buffer,
                             thunk_2.convolution_slices().kernel_shape);
    VerifySliceShapeEquality(thunk_1.convolution_slices().output_buffer,
                             thunk_1.convolution_slices().output_shape,
                             thunk_2.convolution_slices().output_buffer,
                             thunk_2.convolution_slices().output_shape);
  }

  bool VerifyKernelThunkEquality(const KernelThunkBase& thunk_1,
                                 const KernelThunkBase& thunk_2) {
    return thunk_1.kernel_name() == thunk_2.kernel_name() &&
           thunk_1.thread_dim() == thunk_2.thread_dim() &&
           thunk_1.min_alignment() == thunk_2.min_alignment() &&
           absl::c_equal(thunk_1.arguments_buffers(),
                         thunk_2.arguments_buffers(),
                         [this](const BufferAllocation::Slice& slice_1,
                                const BufferAllocation::Slice& slice_2) {
                           return VerifySliceEquality(slice_1, slice_2);
                         }) &&
           absl::c_equal(thunk_1.results_buffers(), thunk_2.results_buffers(),
                         [this](const BufferAllocation::Slice& slice_1,
                                const BufferAllocation::Slice& slice_2) {
                           return VerifySliceEquality(slice_1, slice_2);
                         });
  }

  bool VerifyConvolutionThunkEquality(const ConvolutionThunk& thunk_1,
                                      const ConvolutionThunk& thunk_2) {
    const bool are_dnums_equal =
        absl::c_equal(thunk_1.dnums().input_spatial_dimensions(),
                      thunk_2.dnums().input_spatial_dimensions()) &&
        absl::c_equal(thunk_1.dnums().kernel_spatial_dimensions(),
                      thunk_2.dnums().kernel_spatial_dimensions()) &&
        absl::c_equal(thunk_1.dnums().output_spatial_dimensions(),
                      thunk_2.dnums().output_spatial_dimensions()) &&
        thunk_1.dnums().input_batch_dimension() ==
            thunk_2.dnums().input_batch_dimension() &&
        thunk_1.dnums().input_feature_dimension() ==
            thunk_2.dnums().input_feature_dimension() &&
        thunk_1.dnums().kernel_input_feature_dimension() ==
            thunk_2.dnums().kernel_input_feature_dimension() &&
        thunk_1.dnums().kernel_output_feature_dimension() ==
            thunk_2.dnums().kernel_output_feature_dimension() &&
        thunk_1.dnums().output_batch_dimension() ==
            thunk_2.dnums().output_batch_dimension() &&
        thunk_1.dnums().output_feature_dimension() ==
            thunk_2.dnums().output_feature_dimension();

    const bool are_windows_equal = absl::c_equal(
        thunk_1.window().dimensions(), thunk_2.window().dimensions(),
        [](const WindowDimension& window_dimension_1,
           const WindowDimension& window_dimension_2) {
          return window_dimension_1.size() == window_dimension_2.size() &&
                 window_dimension_1.stride() == window_dimension_2.stride() &&
                 window_dimension_1.padding_low() ==
                     window_dimension_2.padding_low() &&
                 window_dimension_1.padding_high() ==
                     window_dimension_2.padding_high() &&
                 window_dimension_1.window_dilation() ==
                     window_dimension_2.window_dilation() &&
                 window_dimension_1.base_dilation() ==
                     window_dimension_2.base_dilation() &&
                 window_dimension_1.window_reversal() ==
                     window_dimension_2.window_reversal();
        });

    const bool are_options_equal =
        thunk_1.options().multi_threaded == thunk_2.options().multi_threaded;

    return are_dnums_equal && are_windows_equal && are_options_equal &&
           thunk_1.feature_group_count() == thunk_2.feature_group_count() &&
           VerifySliceShapeEquality(thunk_1.convolution_slices().input_buffer,
                                    thunk_1.convolution_slices().input_shape,
                                    thunk_2.convolution_slices().input_buffer,
                                    thunk_2.convolution_slices().input_shape);
    VerifySliceShapeEquality(thunk_1.convolution_slices().kernel_buffer,
                             thunk_1.convolution_slices().kernel_shape,
                             thunk_2.convolution_slices().kernel_buffer,
                             thunk_2.convolution_slices().kernel_shape);
    VerifySliceShapeEquality(thunk_1.convolution_slices().output_buffer,
                             thunk_1.convolution_slices().output_shape,
                             thunk_2.convolution_slices().output_buffer,
                             thunk_2.convolution_slices().output_shape);
  }

  bool VerifyReduceScatterThunkEquality(const ReduceScatterThunk& thunk_1,
                                        const ReduceScatterThunk& thunk_2) {
    return thunk_1.reduction_kind() == thunk_2.reduction_kind() &&
           VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyThunkEquality(const Thunk& thunk_1, const Thunk& thunk_2) {
    if (thunk_1.kind() != thunk_2.kind()) {
      return false;
    }

    if (!(thunk_1.info().op_name == thunk_2.info().op_name &&
          thunk_1.info().module_name == thunk_2.info().module_name &&
          thunk_1.info().module_id == thunk_2.info().module_id)) {
      return false;
    }

    switch (thunk_1.kind()) {
      case Thunk::Kind::kAllGather:
        return VerifyAllGatherThunkEquality(
            static_cast<const AllGatherThunk&>(thunk_1),
            static_cast<const AllGatherThunk&>(thunk_2));
      case Thunk::Kind::kAllReduce:
        return VerifyAllReduceThunkEquality(
            static_cast<const AllReduceThunk&>(thunk_1),
            static_cast<const AllReduceThunk&>(thunk_2));
      case Thunk::Kind::kAllToAll:
        return VerifyAllToAllThunkEquality(
            static_cast<const AllToAllThunk&>(thunk_1),
            static_cast<const AllToAllThunk&>(thunk_2));
      case Thunk::Kind::kCall:
        return VerifyCallThunkEquality(static_cast<const CallThunk&>(thunk_1),
                                       static_cast<const CallThunk&>(thunk_2));
      case Thunk::Kind::kCollectivePermute:
        return VerifyCollectivePermuteThunkEquality(
            static_cast<const CollectivePermuteThunk&>(thunk_1),
            static_cast<const CollectivePermuteThunk&>(thunk_2));
      case Thunk::Kind::kCopy:
        return VerifyCopyThunkEquality(static_cast<const CopyThunk&>(thunk_1),
                                       static_cast<const CopyThunk&>(thunk_2));
      case Thunk::Kind::kConditional:
        return VerifyConditionalThunkEquality(
            static_cast<const ConditionalThunk&>(thunk_1),
            static_cast<const ConditionalThunk&>(thunk_2));
      case Thunk::Kind::kCustomCall:
        return VerifyCustomCallThunkEquality(
            static_cast<const CustomCallThunk&>(thunk_1),
            static_cast<const CustomCallThunk&>(thunk_2));
      case Thunk::Kind::kDot:
        return VerifyDotThunkEquality(static_cast<const DotThunk&>(thunk_1),
                                      static_cast<const DotThunk&>(thunk_2));
      case Thunk::Kind::kFft:
        return VerifyFftThunkEquality(static_cast<const FftThunk&>(thunk_1),
                                      static_cast<const FftThunk&>(thunk_2));
      case Thunk::Kind::kInfeed:
        return VerifyInfeedThunkEquality(
            static_cast<const InfeedThunk&>(thunk_1),
            static_cast<const InfeedThunk&>(thunk_2));
      case Thunk::Kind::kOutfeed:
        return VerifyOutfeedThunkEquality(
            static_cast<const OutfeedThunk&>(thunk_1),
            static_cast<const OutfeedThunk&>(thunk_2));
      case Thunk::Kind::kPartitionId:
        return VerifyPartitionIdThunkEquality(
            static_cast<const PartitionIdThunk&>(thunk_1),
            static_cast<const PartitionIdThunk&>(thunk_2));
      case Thunk::Kind::kReplicaId:
        return VerifyReplicaIdThunkEquality(
            static_cast<const ReplicaIdThunk&>(thunk_1),
            static_cast<const ReplicaIdThunk&>(thunk_2));
      case Thunk::Kind::kRngGetAndUpdateState:
        return VerifyRngGetAndUpdateStateThunkEquality(
            static_cast<const RngGetAndUpdateStateThunk&>(thunk_1),
            static_cast<const RngGetAndUpdateStateThunk&>(thunk_2));
      case Thunk::Kind::kSort:
        return VerifySortThunkEquality(static_cast<const SortThunk&>(thunk_1),
                                       static_cast<const SortThunk&>(thunk_2));
      case Thunk::Kind::kTopK:
        return VerifyTopKThunkEquality(static_cast<const TopKThunk&>(thunk_1),
                                       static_cast<const TopKThunk&>(thunk_2));
      case Thunk::Kind::kWhile:
        return VerifyWhileThunkEquality(
            static_cast<const WhileThunk&>(thunk_1),
            static_cast<const WhileThunk&>(thunk_2));
      case Thunk::Kind::kXnnFusion: {
        auto* xnn_dot_1 = dynamic_cast<const XnnDotThunk*>(&thunk_1);
        auto* xnn_dot_2 = dynamic_cast<const XnnDotThunk*>(&thunk_2);
        auto* xnn_conv_1 = dynamic_cast<const XnnConvolutionThunk*>(&thunk_1);
        auto* xnn_conv_2 = dynamic_cast<const XnnConvolutionThunk*>(&thunk_2);
        if (xnn_dot_1 && xnn_dot_2) {
          return VerifyXnnDotThunkEquality(
              static_cast<const XnnDotThunk&>(thunk_1),
              static_cast<const XnnDotThunk&>(thunk_2));
        } else if (xnn_conv_1 && xnn_conv_2) {
          return VerifyXnnConvolutionThunkEquality(
              static_cast<const XnnConvolutionThunk&>(thunk_1),
              static_cast<const XnnConvolutionThunk&>(thunk_2));
        } else {
          CHECK(false) << "Unsupported XnnFusion thunk type";
          return false;
        }
      }
      case Thunk::Kind::kKernel:
        return VerifyKernelThunkEquality(
            static_cast<const KernelThunkBase&>(thunk_1),
            static_cast<const KernelThunkBase&>(thunk_2));
      case Thunk::Kind::kConvolution:
        return VerifyConvolutionThunkEquality(
            static_cast<const ConvolutionThunk&>(thunk_1),
            static_cast<const ConvolutionThunk&>(thunk_2));
      case Thunk::Kind::kReduceScatter:
        return VerifyReduceScatterThunkEquality(
            static_cast<const ReduceScatterThunk&>(thunk_1),
            static_cast<const ReduceScatterThunk&>(thunk_2));
      case Thunk::Kind::kUnknown:
        return false;
    }

    return true;
  }
  std::unique_ptr<SerDesBase<ThunkSequence>> thunk_sequence_serdes_;
  std::vector<BufferAllocation> buffer_allocations_;
  std::vector<Literal> literals_;
};

// List of all serdes implementations to test.
using Implementations = ::testing::Types<ThunkSequenceSerDesProtobuf>;

TYPED_TEST_SUITE(ThunkSequenceSerdesTest, Implementations, );

TYPED_TEST(ThunkSequenceSerdesTest, SerializeAndDeserialize) {
  TF_ASSERT_OK_AND_ASSIGN(ThunkSequence thunk_sequence,
                          this->CreateThunkSequenceFromAllThunkTypes());
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          this->Serialize(thunk_sequence));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ThunkSequence> deserialized,
                          this->Deserialize(serialized));
  EXPECT_TRUE(this->VerifyThunkSequenceEquality(thunk_sequence, *deserialized));
}

}  // namespace

}  // namespace xla::cpu
