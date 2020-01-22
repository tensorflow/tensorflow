/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <algorithm>
#include <ostream>
#include <set>
#include <unordered_set>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/human_readable_json.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::CEscape;
using absl::StrAppend;
using absl::StrCat;
using absl::StrJoin;

/* static */
StatusOr<std::unique_ptr<HloInstruction>> HloInstruction::CreateFromProto(
    const HloInstructionProto& proto,
    const absl::flat_hash_map<int64, HloInstruction*>& instruction_map,
    const absl::flat_hash_map<int64, HloComputation*>& computation_map,
    bool prohibit_empty_literal) {
  TF_RET_CHECK(!proto.opcode().empty());
  HloOpcode opcode;
  auto opcode_or = StringToHloOpcode(proto.opcode());
  absl::optional<ComparisonDirection> comparison_direction;
  if (opcode_or.ok()) {
    opcode = opcode_or.ConsumeValueOrDie();
  } else {
    // Unknown opcode. Try auto-upgrading deprecated "less-than",
    // "greater-than", etc opcodes, which are now rolled into the kCompare
    // opcode.
    if (proto.opcode() == "equal-to") {
      comparison_direction = ComparisonDirection::kEq;
    } else if (proto.opcode() == "not-equal-to") {
      comparison_direction = ComparisonDirection::kNe;
    } else if (proto.opcode() == "greater-than-or-equal-to") {
      comparison_direction = ComparisonDirection::kGe;
    } else if (proto.opcode() == "greater-than") {
      comparison_direction = ComparisonDirection::kGt;
    } else if (proto.opcode() == "less-than-or-equal-to") {
      comparison_direction = ComparisonDirection::kLe;
    } else if (proto.opcode() == "less-than") {
      comparison_direction = ComparisonDirection::kLt;
    }
    if (comparison_direction) {
      opcode = HloOpcode::kCompare;
    } else {
      return InvalidArgument("Unknown opcode: %s", proto.opcode());
    }
  }

  TF_RET_CHECK(proto.has_shape());

  std::unique_ptr<HloInstruction> instruction;
  const auto operands = [&instruction_map, &proto](int index) {
    return instruction_map.at(proto.operand_ids(index));
  };
  const auto all_operands = [&instruction_map, &proto]() {
    std::vector<HloInstruction*> result(proto.operand_ids_size());
    std::transform(proto.operand_ids().begin(), proto.operand_ids().end(),
                   result.begin(), [&instruction_map](int64 operand_id) {
                     return instruction_map.at(operand_id);
                   });
    return result;
  };
  const auto computations = [&computation_map, &proto](int index) {
    return computation_map.at(proto.called_computation_ids(index));
  };
  const auto all_computations = [&computation_map, &proto]() {
    std::vector<HloComputation*> result(proto.called_computation_ids_size());
    std::transform(proto.called_computation_ids().begin(),
                   proto.called_computation_ids().end(), result.begin(),
                   [&computation_map](int64 computation_id) {
                     return computation_map.at(computation_id);
                   });
    return result;
  };

  TF_RET_CHECK(
      absl::c_all_of(proto.operand_ids(),
                     [&](int64 id) { return instruction_map.contains(id); }))
      << proto.name() << " instruction contains invalid operand id(s)";

  TF_RET_CHECK(
      absl::c_all_of(proto.called_computation_ids(),
                     [&](int64 id) { return computation_map.contains(id); }))
      << proto.name() << " instruction references invalid computation id(s)";

  Shape shape(proto.shape());
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));

  absl::optional<int> arity = HloOpcodeArity(opcode);
  if (arity) {
    TF_RET_CHECK(proto.operand_ids_size() == *arity)
        << proto.opcode() << " instruction should have " << *arity
        << " operands but sees " << proto.operand_ids_size();
  }

  switch (opcode) {
    // Ops migrated to subclasses.
    case HloOpcode::kBatchNormTraining:
      instruction =
          CreateBatchNormTraining(shape, operands(0), operands(1), operands(2),
                                  proto.epsilon(), proto.feature_index());
      break;
    case HloOpcode::kBatchNormInference:
      instruction = CreateBatchNormInference(
          shape, operands(0), operands(1), operands(2), operands(3),
          operands(4), proto.epsilon(), proto.feature_index());
      break;
    case HloOpcode::kBatchNormGrad:
      instruction = CreateBatchNormGrad(shape, operands(0), operands(1),
                                        operands(2), operands(3), operands(4),
                                        proto.epsilon(), proto.feature_index());
      break;
    case HloOpcode::kFft: {
      std::vector<int64> fft_length(proto.fft_length().begin(),
                                    proto.fft_length().end());
      instruction = CreateFft(shape, operands(0), proto.fft_type(),
                              absl::Span<const int64>(fft_length));
      break;
    }
    case HloOpcode::kCompare: {
      // Auto-upgraded from deprecated opcode skips the following.
      if (!comparison_direction) {
        TF_ASSIGN_OR_RETURN(
            comparison_direction,
            StringToComparisonDirection(proto.comparison_direction()));
      }
      instruction =
          CreateCompare(shape, operands(0), operands(1), *comparison_direction);
      break;
    }
    case HloOpcode::kTriangularSolve: {
      instruction = CreateTriangularSolve(shape, operands(0), operands(1),
                                          proto.triangular_solve_options());
      break;
    }
    case HloOpcode::kCholesky: {
      instruction =
          CreateCholesky(shape, operands(0), proto.cholesky_options());
      break;
    }
    case HloOpcode::kSend:
      instruction = CreateSend(operands(0), operands(1), proto.channel_id(),
                               proto.is_host_transfer());
      break;
    case HloOpcode::kSendDone:
      instruction = CreateSendDone(operands(0), proto.is_host_transfer());
      break;
    case HloOpcode::kRecv:
      instruction = CreateRecv(shape.tuple_shapes(0), operands(0),
                               proto.channel_id(), proto.is_host_transfer());
      break;
    case HloOpcode::kRecvDone:
      instruction = CreateRecvDone(operands(0), proto.is_host_transfer());
      break;
    case HloOpcode::kReverse:
      instruction = CreateReverse(shape, operands(0),
                                  std::vector<int64>(proto.dimensions().begin(),
                                                     proto.dimensions().end()));
      break;
    case HloOpcode::kConcatenate:
      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "Concatenate instruction should have 1 dimension but sees "
          << proto.dimensions_size();
      instruction =
          CreateConcatenate(shape, all_operands(), proto.dimensions(0));
      break;
    case HloOpcode::kConditional: {
      TF_RET_CHECK(proto.called_computation_ids_size() > 0)
          << "conditional should have at least 1 called computation";
      if (operands(0)->shape().element_type() == PRED) {
        TF_RET_CHECK(proto.called_computation_ids_size() == 2)
            << "conditional should have exactly 2 called computations but got "
            << proto.called_computation_ids_size();
      }
      TF_RET_CHECK(proto.operand_ids_size() ==
                   proto.called_computation_ids_size() + 1)
          << "conditional should have one branch_index operand plus one "
             "operand per called computation but got "
          << proto.operand_ids_size() << " operands for "
          << proto.called_computation_ids_size() << " branch computations";
      auto cond_operands = all_operands();
      instruction =
          CreateConditional(shape, cond_operands[0], all_computations(),
                            absl::MakeSpan(cond_operands).subspan(1));
      break;
    }
    case HloOpcode::kReduce:
      TF_RET_CHECK(proto.operand_ids_size() % 2 == 0)
          << "Reduce instruction should have an even number of operands but "
             "sees "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Reduce instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      {
        const auto reduce_operands = all_operands();
        auto inputs = absl::MakeSpan(reduce_operands)
                          .subspan(0, reduce_operands.size() / 2);
        auto init_values =
            absl::MakeSpan(reduce_operands)
                .subspan(reduce_operands.size() / 2, reduce_operands.size());
        instruction =
            CreateReduce(shape, inputs, init_values,
                         std::vector<int64>(proto.dimensions().begin(),
                                            proto.dimensions().end()),
                         computations(0));
      }
      break;
    case HloOpcode::kSort: {
      TF_RET_CHECK(proto.operand_ids_size() >= 1)
          << "Sort instruction should have at least 1 operand but has "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.dimensions().size() == 1)
          << "Sort instruction should have 1 dimension";
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Sort instruction should one called computation but sees "
          << proto.called_computation_ids_size();
      auto sort_operands = all_operands();
      instruction = CreateSort(shape, proto.dimensions(0), all_operands(),
                               computations(0), proto.is_stable());
      break;
    }
    case HloOpcode::kTranspose:
      instruction =
          CreateTranspose(shape, operands(0),
                          std::vector<int64>(proto.dimensions().begin(),
                                             proto.dimensions().end()));
      break;
    case HloOpcode::kBroadcast:
      instruction =
          CreateBroadcast(shape, operands(0),
                          std::vector<int64>(proto.dimensions().begin(),
                                             proto.dimensions().end()));
      break;
    case HloOpcode::kMap:
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Map instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      instruction = CreateMap(shape, all_operands(), computations(0));
      break;
    case HloOpcode::kSlice: {
      std::vector<int64> slice_starts, slice_limits, slice_strides;
      for (const HloInstructionProto::SliceDimensions& slice_dimensions :
           proto.slice_dimensions()) {
        slice_starts.push_back(slice_dimensions.start());
        slice_limits.push_back(slice_dimensions.limit());
        slice_strides.push_back(slice_dimensions.stride());
      }
      instruction = CreateSlice(shape, operands(0), slice_starts, slice_limits,
                                slice_strides);
      break;
    }
    case HloOpcode::kConstant: {
      // TODO(b/110214922): Revert this to CHECK(proto.has_literal()).
      if (proto.has_literal()) {
        TF_ASSIGN_OR_RETURN(
            auto literal,
            Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
        instruction = CreateConstant(std::move(literal));
        // Literal's shape may have no/different tiling info.
        TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
            instruction->shape(), shape));
        *instruction->mutable_shape() = shape;
      } else {
        instruction = absl::make_unique<HloConstantInstruction>(shape);
      }
      break;
    }
    case HloOpcode::kTrace: {
      TF_RET_CHECK(proto.has_literal());
      TF_ASSIGN_OR_RETURN(
          auto literal,
          Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
      instruction = CreateTrace(literal.GetR1U8AsString(), operands(0));
      break;
    }
    case HloOpcode::kFusion: {
      // In the proto, fused computations are held exclusively within the
      // HloInstructionProto and do not appear as an HloComputationProto within
      // the HloModuleProto.
      TF_RET_CHECK(!proto.fusion_kind().empty());
      TF_ASSIGN_OR_RETURN(FusionKind fusion_kind,
                          StringToFusionKind(proto.fusion_kind()));

      // Find the fused computation and set its fusion instruction.
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Expect 1 called computation for fusion instruction but sees "
          << proto.called_computation_ids_size();
      const int64 fusion_id = proto.called_computation_ids(0);
      auto* fused_computation =
          tensorflow::gtl::FindPtrOrNull(computation_map, fusion_id);
      TF_RET_CHECK(fused_computation != nullptr)
          << "No fusion computation with id " << fusion_id;
      instruction =
          CreateFusion(shape, fusion_kind, all_operands(), fused_computation);
      break;
    }
    case HloOpcode::kRng:
      instruction = CreateRng(shape, proto.distribution(), all_operands());
      break;
    case HloOpcode::kRngGetAndUpdateState:
      instruction = CreateRngGetAndUpdateState(shape, proto.delta());
      break;
    case HloOpcode::kParameter:
      instruction =
          CreateParameter(proto.parameter_number(), shape, proto.name());
      if (!proto.parameter_replication().replicated_at_leaf_buffers().empty()) {
        instruction->set_parameter_replicated_at_leaf_buffers(
            proto.parameter_replication().replicated_at_leaf_buffers());
      }
      break;
    case HloOpcode::kGetTupleElement:
      instruction =
          CreateGetTupleElement(shape, operands(0), proto.tuple_index());
      break;
    case HloOpcode::kReducePrecision:
      instruction = CreateReducePrecision(
          shape, operands(0), proto.exponent_bits(), proto.mantissa_bits());
      break;
    case HloOpcode::kInfeed: {
      TF_RET_CHECK(shape.IsTuple() &&
                   (ShapeUtil::TupleElementCount(shape) == 2))
          << "Infeed should have a tuple shape with 2 operands, but has: "
          << shape;
      const Shape& data_shape = ShapeUtil::GetTupleElementShape(shape, 0);
      instruction =
          CreateInfeed(data_shape, operands(0), proto.infeed_config());
    } break;
    case HloOpcode::kOutfeed: {
      Shape outfeed_shape(proto.outfeed_shape());
      TF_RETURN_IF_ERROR(
          ShapeUtil::ValidateShapeWithOptionalLayout(outfeed_shape));
      instruction = CreateOutfeed(outfeed_shape, operands(0), operands(1),
                                  proto.outfeed_config());
      break;
    }
    case HloOpcode::kAllReduce: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "AllReduce should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      TF_RET_CHECK(proto.channel_id() <= 0 || proto.all_reduce_id() <= 0)
          << "AllReduce cannot have both channel_id() and all_reduce_id()";
      absl::optional<int64> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      if (proto.all_reduce_id() > 0) {
        channel_id = proto.all_reduce_id();
      }
      instruction = CreateAllReduce(
          shape, all_operands(), computations(0),
          /*replica_groups=*/
          std::vector<ReplicaGroup>(proto.replica_groups().begin(),
                                    proto.replica_groups().end()),
          /*constrain_layout=*/proto.constrain_layout(),
          /*channel_id=*/channel_id);
      break;
    }
    case HloOpcode::kAllToAll: {
      absl::optional<int64> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      instruction = CreateAllToAll(
          shape, all_operands(),
          /*replica_groups=*/
          std::vector<ReplicaGroup>(proto.replica_groups().begin(),
                                    proto.replica_groups().end()),
          /*channel_id=*/channel_id);
      break;
    }
    case HloOpcode::kCollectivePermute: {
      std::vector<std::pair<int64, int64>> source_target_pairs(
          proto.source_target_pairs_size());
      absl::optional<int64> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      for (int i = 0; i < source_target_pairs.size(); i++) {
        source_target_pairs[i].first = proto.source_target_pairs(i).source();
        source_target_pairs[i].second = proto.source_target_pairs(i).target();
      }
      instruction = CreateCollectivePermute(shape, operands(0),
                                            source_target_pairs, channel_id);
      break;
    }
    case HloOpcode::kReplicaId: {
      instruction = CreateReplicaId();
      break;
    }
    case HloOpcode::kPartitionId: {
      instruction = CreatePartitionId();
      break;
    }
    case HloOpcode::kConvolution: {
      TF_RET_CHECK(proto.has_window());
      TF_RET_CHECK(proto.has_convolution_dimension_numbers());
      PrecisionConfig precision_config = proto.precision_config();
      precision_config.mutable_operand_precision()->Resize(
          proto.operand_ids_size(), PrecisionConfig::DEFAULT);
      instruction = CreateConvolve(
          shape, operands(0), operands(1),
          std::max<int64>(proto.feature_group_count(), 1),
          std::max<int64>(proto.batch_group_count(), 1), proto.window(),
          proto.convolution_dimension_numbers(), precision_config);
      break;
    }
    case HloOpcode::kReduceWindow:
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "ReduceWindow should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      instruction = CreateReduceWindow(shape, operands(0), operands(1),
                                       proto.window(), computations(0));
      break;
    case HloOpcode::kSelectAndScatter:
      TF_RET_CHECK(proto.called_computation_ids_size() == 2)
          << "SelectAndScatter should have 2 called computations but sees "
          << proto.called_computation_ids_size();
      instruction = CreateSelectAndScatter(shape, operands(0), computations(0),
                                           proto.window(), operands(1),
                                           operands(2), computations(1));
      break;
    case HloOpcode::kCustomCall: {
      if (proto.constrain_layout()) {
        // A proto RepeatedPtrField cannot be converted to a Span (it is a
        // vector of pointers essentially) so create a vector of shapes to pass
        // in.
        std::vector<Shape> operand_shapes;
        for (const ShapeProto& shape_proto :
             proto.operand_shapes_with_layout()) {
          operand_shapes.emplace_back(shape_proto);
        }
        instruction =
            CreateCustomCall(shape, all_operands(), proto.custom_call_target(),
                             operand_shapes, proto.backend_config());
      } else {
        instruction =
            CreateCustomCall(shape, all_operands(), proto.custom_call_target(),
                             proto.backend_config());
      }
      auto custom_call_instr =
          Cast<HloCustomCallInstruction>(instruction.get());
      if (proto.has_window()) {
        custom_call_instr->set_window(proto.window());
      }
      if (proto.has_convolution_dimension_numbers()) {
        custom_call_instr->set_convolution_dimension_numbers(
            proto.convolution_dimension_numbers());
      }
      custom_call_instr->set_feature_group_count(
          std::max(static_cast<int64>(proto.feature_group_count()), int64{1}));
      custom_call_instr->set_batch_group_count(
          std::max(static_cast<int64>(proto.batch_group_count()), int64{1}));
      custom_call_instr->set_custom_call_has_side_effect(
          proto.custom_call_has_side_effect());
      break;
    }
    case HloOpcode::kPad:
      TF_RET_CHECK(proto.has_padding_config());
      instruction =
          CreatePad(shape, operands(0), operands(1), proto.padding_config());
      break;
    case HloOpcode::kDynamicSlice: {
      std::vector<int64> slice_sizes(proto.dynamic_slice_sizes_size());
      absl::c_copy(proto.dynamic_slice_sizes(), slice_sizes.begin());
      TF_RET_CHECK(proto.operand_ids_size() >= 1)
          << "DynamicSlice instruction should have at least 1 operands but "
             "sees "
          << proto.operand_ids_size();
      // TODO(b/118437727): Old form, make the check unconditional.
      if (proto.operand_ids_size() != 2 || operands(1)->shape().rank() != 1) {
        auto expected_operands = 1 + operands(0)->shape().rank();
        TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
            << "DynamicSlice instruction should have " << expected_operands
            << " operands, but has " << proto.operand_ids_size();
      }
      const auto& operand_vector = all_operands();
      instruction = CreateDynamicSlice(
          shape, operands(0), absl::MakeSpan(operand_vector).subspan(1),
          slice_sizes);
      break;
    }
    case HloOpcode::kDynamicUpdateSlice: {
      TF_RET_CHECK(proto.operand_ids_size() >= 2)
          << "DynamicUpdateSlice instruction should have at least 2 operands "
             "but sees "
          << proto.operand_ids_size();
      // TODO(b/118437727): Old form, make the check unconditional.
      if (proto.operand_ids_size() != 3 || operands(2)->shape().rank() != 1) {
        auto expected_operands = 2 + operands(0)->shape().rank();
        TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
            << "DynamicUpdateSlice instruction should have "
            << expected_operands << " operands, but has "
            << proto.operand_ids_size();
      }
      const auto& operand_vector = all_operands();
      instruction =
          CreateDynamicUpdateSlice(shape, operands(0), operands(1),
                                   absl::MakeSpan(operand_vector).subspan(2));

      break;
    }
    case HloOpcode::kGather: {
      TF_RET_CHECK(proto.has_gather_dimension_numbers())
          << "Gather instruction should have GatherDimensionNumbers set.";
      auto gather_dimension_numbers = absl::make_unique<GatherDimensionNumbers>(
          proto.gather_dimension_numbers());
      std::vector<int64> gather_slice_sizes;
      for (int64 bound : proto.gather_slice_sizes()) {
        gather_slice_sizes.push_back(bound);
      }
      instruction = CreateGather(shape, operands(0), operands(1),
                                 *gather_dimension_numbers, gather_slice_sizes,
                                 proto.indices_are_sorted());
      break;
    }
    case HloOpcode::kScatter: {
      TF_RET_CHECK(proto.has_scatter_dimension_numbers())
          << "Scatter instruction should have ScatterDimensionNumbers set.";
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Scatter instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      auto scatter_dimension_numbers =
          absl::make_unique<ScatterDimensionNumbers>(
              proto.scatter_dimension_numbers());
      instruction =
          CreateScatter(shape, operands(0), operands(1), operands(2),
                        computations(0), *scatter_dimension_numbers,
                        proto.indices_are_sorted(), proto.unique_indices());
      break;
    }
    case HloOpcode::kIota:
      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "Iota instruction should have 1 dimension but sees "
          << proto.dimensions_size();
      instruction = CreateIota(shape, proto.dimensions(0));
      break;
    case HloOpcode::kDot: {
      TF_RET_CHECK(proto.has_dot_dimension_numbers())
          << "Dot instruction should have dot_dimension_numbers.";
      PrecisionConfig precision_config = proto.precision_config();
      precision_config.mutable_operand_precision()->Resize(
          proto.operand_ids_size(), PrecisionConfig::DEFAULT);
      instruction = absl::make_unique<HloDotInstruction>(
          shape, operands(0), operands(1), proto.dot_dimension_numbers(),
          precision_config);
      break;
    }
    case HloOpcode::kDomain: {
      std::shared_ptr<const HloSharding> entry_hlo_sharding;
      std::shared_ptr<const HloSharding> exit_hlo_sharding;
      if (proto.has_domain_entry_sharding()) {
        TF_ASSIGN_OR_RETURN(
            HloSharding sharding,
            HloSharding::FromProto(proto.domain_entry_sharding()));
        entry_hlo_sharding = std::make_shared<const HloSharding>(sharding);
      }
      if (proto.has_domain_exit_sharding()) {
        TF_ASSIGN_OR_RETURN(
            HloSharding sharding,
            HloSharding::FromProto(proto.domain_exit_sharding()));
        exit_hlo_sharding = std::make_shared<const HloSharding>(sharding);
      }
      instruction = absl::make_unique<HloDomainInstruction>(
          shape, operands(0),
          absl::make_unique<ShardingMetadata>(entry_hlo_sharding),
          absl::make_unique<ShardingMetadata>(exit_hlo_sharding));
      break;
    }
    case HloOpcode::kGetDimensionSize:
      TF_RET_CHECK(proto.dimensions_size() == 1);
      instruction =
          CreateGetDimensionSize(shape, operands(0), proto.dimensions(0));
      break;
    case HloOpcode::kSetDimensionSize:
      TF_RET_CHECK(proto.dimensions_size() == 1);
      instruction = CreateSetDimensionSize(shape, operands(0), operands(1),
                                           proto.dimensions(0));
      break;
    case HloOpcode::kReshape: {
      int64 inferred_dimension = -1;
      if (!proto.dimensions().empty()) {
        inferred_dimension = proto.dimensions()[0];
      }
      TF_RET_CHECK(shape.IsArray() && operands(0)->shape().IsArray() &&
                   ShapeUtil::ElementsIn(shape) ==
                       ShapeUtil::ElementsIn(operands(0)->shape()))
          << "shape: " << ShapeUtil::HumanString(shape)
          << " operand: " << ShapeUtil::HumanString(operands(0)->shape());
      instruction = CreateReshape(shape, operands(0), inferred_dimension);
      break;
    }
    default: {
      instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
      for (const int64 operand_id : proto.operand_ids()) {
        instruction->AppendOperand(instruction_map.at(operand_id));
      }
      if (instruction->opcode() != HloOpcode::kFusion) {
        if (instruction->opcode() == HloOpcode::kCall) {
          TF_RET_CHECK(proto.called_computation_ids_size() == 1)
              << "Call should have 1 called computation but has "
              << proto.called_computation_ids_size();
        }
        for (const int64 computation_id : proto.called_computation_ids()) {
          instruction->called_computations_.push_back(
              computation_map.at(computation_id));
        }
      }
      TF_RET_CHECK(!proto.has_precision_config())
          << instruction->opcode() << proto.DebugString();
      TF_RET_CHECK(!proto.has_dot_dimension_numbers()) << instruction->opcode();
      break;
    }
  }

  for (const int64 predecessor_id : proto.control_predecessor_ids()) {
    TF_RET_CHECK(ContainsKey(instruction_map, predecessor_id))
        << "No instruction with id " << predecessor_id;
    TF_RETURN_IF_ERROR(instruction_map.at(predecessor_id)
                           ->AddControlDependencyTo(instruction.get()));
  }

  TF_RET_CHECK(!proto.name().empty());
  instruction->SetAndSanitizeName(proto.name());
  instruction->metadata_ = proto.metadata();
  instruction->backend_config_ = proto.backend_config();
  instruction->outer_dimension_partitions_.assign(
      proto.outer_dimension_partitions().begin(),
      proto.outer_dimension_partitions().end());

  TF_RET_CHECK(proto.id() >= 0)
      << "Instruction with negative id: " << proto.id();
  TF_RET_CHECK(proto.id() <= INT_MAX)
      << "Instruction with id > INT_MAX: " << proto.id();
  instruction->unique_id_ = proto.id();

  if (proto.has_sharding()) {
    TF_ASSIGN_OR_RETURN(const auto& sharding,
                        HloSharding::FromProto(proto.sharding()));
    instruction->set_sharding(sharding);
  }

  if (proto.has_frontend_attributes()) {
    instruction->set_frontend_attributes(proto.frontend_attributes());
  }

  return std::move(instruction);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateParameter(
    int64 parameter_number, const Shape& shape, const string& name) {
  return absl::make_unique<HloParameterInstruction>(parameter_number, shape,
                                                    name);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTrace(
    const string& tag, HloInstruction* operand) {
  return absl::make_unique<HloTraceInstruction>(tag, operand);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    Literal literal) {
  return absl::make_unique<HloConstantInstruction>(std::move(literal));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateIota(
    const Shape& shape, int64 iota_dimension) {
  return absl::make_unique<HloIotaInstruction>(shape, iota_dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetTupleElement(const Shape& shape,
                                      HloInstruction* operand, int64 index) {
  return absl::make_unique<HloGetTupleElementInstruction>(shape, operand,
                                                          index);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRng(
    const Shape& shape, RandomDistribution distribution,
    absl::Span<HloInstruction* const> parameters) {
  return absl::make_unique<HloRngInstruction>(shape, distribution, parameters);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateRngGetAndUpdateState(const Shape& shape, int64 delta) {
  return absl::make_unique<HloRngGetAndUpdateStateInstruction>(shape, delta);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateNary(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
  if (opcode == HloOpcode::kCopy) {
    // It is impossible to copy an opaque shape, we don't know how big it is.
    CHECK(!shape.IsOpaque());
  }
  auto instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateUnary(
    const Shape& shape, HloOpcode opcode, HloInstruction* operand) {
  // Only certain opcodes are supported with CreateUnary: opcodes of unary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kClz:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
      break;
    default:
      LOG(FATAL) << "Invalid unary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {operand});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBinary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs) {
  // Only certain opcodes are supported with CreateBinary: opcodes of binary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kComplex:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      break;
    default:
      LOG(FATAL) << "Invalid binary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTernary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs, HloInstruction* ehs) {
  // Only certain opcodes are supported with CreateTernary: opcodes of ternary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect:
      break;
    default:
      LOG(FATAL) << "Invalid ternary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs, ehs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateVariadic(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
  CHECK_EQ(HloOpcode::kTuple, opcode);
  return CreateNary(shape, opcode, operands);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateMap(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* map_computation) {
  return absl::make_unique<HloMapInstruction>(shape, operands, map_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvolve(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64 feature_group_count, int64 batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config) {
  return absl::make_unique<HloConvolutionInstruction>(
      shape, lhs, rhs, feature_group_count, batch_group_count, window,
      dimension_numbers, precision_config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFft(
    const Shape& shape, HloInstruction* operand, FftType fft_type,
    absl::Span<const int64> fft_length) {
  return absl::make_unique<HloFftInstruction>(shape, operand, fft_type,
                                              fft_length);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCompare(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    ComparisonDirection direction) {
  return absl::make_unique<HloCompareInstruction>(shape, lhs, rhs, direction);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateTriangularSolve(const Shape& shape, HloInstruction* a,
                                      HloInstruction* b,
                                      const TriangularSolveOptions& options) {
  return absl::make_unique<HloTriangularSolveInstruction>(shape, a, b, options);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCholesky(
    const Shape& shape, HloInstruction* a, const CholeskyOptions& options) {
  return absl::make_unique<HloCholeskyInstruction>(shape, a, options);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDot(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config) {
  return absl::make_unique<HloDotInstruction>(
      shape, lhs, rhs, dimension_numbers, precision_config);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateReducePrecision(const Shape& shape,
                                      HloInstruction* operand,
                                      const int exponent_bits,
                                      const int mantissa_bits) {
  return absl::make_unique<HloReducePrecisionInstruction>(
      shape, operand, exponent_bits, mantissa_bits);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAllReduce(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    const std::vector<ReplicaGroup>& replica_groups, bool constrain_layout,
    const absl::optional<int64>& channel_id) {
  return absl::make_unique<HloAllReduceInstruction>(
      shape, operands, reduce_computation, replica_groups, constrain_layout,
      channel_id);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAllToAll(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<ReplicaGroup>& replica_groups,
    const absl::optional<int64>& channel_id) {
  return absl::make_unique<HloAllToAllInstruction>(shape, operands,
                                                   replica_groups, channel_id);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCollectivePermute(
    const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64, int64>>& source_target_pairs,
    const absl::optional<int64>& channel_id) {
  return absl::make_unique<HloCollectivePermuteInstruction>(
      shape, operand, source_target_pairs, channel_id);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReplicaId() {
  return absl::WrapUnique(
      new HloInstruction(HloOpcode::kReplicaId, ShapeUtil::MakeShape(U32, {})));
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreatePartitionId() {
  return absl::WrapUnique(new HloInstruction(HloOpcode::kPartitionId,
                                             ShapeUtil::MakeShape(U32, {})));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateInfeed(
    const Shape& infeed_shape, HloInstruction* token_operand,
    const string& config) {
  return absl::make_unique<HloInfeedInstruction>(infeed_shape, token_operand,
                                                 config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateOutfeed(
    const Shape& outfeed_shape, HloInstruction* operand,
    HloInstruction* token_operand, absl::string_view outfeed_config) {
  return absl::make_unique<HloOutfeedInstruction>(
      outfeed_shape, operand, token_operand, outfeed_config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSend(
    HloInstruction* operand, HloInstruction* token, int64 channel_id,
    bool is_host_transfer) {
  return absl::make_unique<HloSendInstruction>(operand, token, channel_id,
                                               is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSendDone(
    HloInstruction* operand, bool is_host_transfer) {
  auto send_operand = DynCast<HloSendInstruction>(operand);
  CHECK(send_operand != nullptr)
      << "SendDone must take the context operand from Send";
  return absl::make_unique<HloSendDoneInstruction>(send_operand,
                                                   is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecv(
    const Shape& shape, HloInstruction* token, int64 channel_id,
    bool is_host_transfer) {
  return absl::make_unique<HloRecvInstruction>(shape, token, channel_id,
                                               is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecvDone(
    HloInstruction* operand, bool is_host_transfer) {
  auto recv_operand = DynCast<HloRecvInstruction>(operand);
  CHECK(recv_operand != nullptr)
      << "RecvDone must take the context operand from Recv";
  return absl::make_unique<HloRecvDoneInstruction>(recv_operand,
                                                   is_host_transfer);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReverse(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64> dimensions) {
  return absl::make_unique<HloReverseInstruction>(shape, operand, dimensions);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateAfterAll(
    absl::Span<HloInstruction* const> operands) {
  CHECK(!operands.empty());
  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateToken() {
  return absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateAddDependency(HloInstruction* data_operand,
                                    HloInstruction* token_operand) {
  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAddDependency, data_operand->shape()));
  instruction->AppendOperand(data_operand);
  instruction->AppendOperand(token_operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateWhile(
    const Shape& shape, HloComputation* condition, HloComputation* body,
    HloInstruction* init) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kWhile, shape));
  instruction->AppendOperand(init);
  // Body comes before condition computation in the vector.
  instruction->called_computations_.push_back(body);
  instruction->called_computations_.push_back(condition);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* pred,
    HloInstruction* true_computation_arg, HloComputation* true_computation,
    HloInstruction* false_computation_arg, HloComputation* false_computation) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(pred);
  instruction->AppendOperand(true_computation_arg);
  instruction->AppendOperand(false_computation_arg);
  // In called_computations_, the index of true_computation must be 0 and that
  // of false computation must be 1, as defined by kTrueComputationIndex and
  // kFalseComputationIndex.
  instruction->called_computations_.push_back(true_computation);
  instruction->called_computations_.push_back(false_computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* branch_index,
    absl::Span<HloComputation* const> branch_computations,
    absl::Span<HloInstruction* const> branch_computation_args) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(branch_index);
  CHECK_EQ(branch_computations.size(), branch_computation_args.size());
  for (int i = 0; i < branch_computations.size(); ++i) {
    instruction->called_computations_.push_back(branch_computations[i]);
    instruction->AppendOperand(branch_computation_args[i]);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSlice(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64> start_indices,
    absl::Span<const int64> limit_indices, absl::Span<const int64> strides) {
  return absl::make_unique<HloSliceInstruction>(shape, operand, start_indices,
                                                limit_indices, strides);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicSlice(
    const Shape& shape, HloInstruction* operand,
    absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64> slice_sizes) {
  return absl::make_unique<HloDynamicSliceInstruction>(
      shape, operand, start_indices, slice_sizes);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateDynamicUpdateSlice(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices) {
  return absl::make_unique<HloDynamicUpdateSliceInstruction>(
      shape, operand, update, start_indices);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConcatenate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64 dimension) {
  return absl::make_unique<HloConcatenateInstruction>(shape, operands,
                                                      dimension);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvert(
    const Shape& shape, HloInstruction* operand) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBitcastConvert(const Shape& shape,
                                     HloInstruction* operand) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kBitcastConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBitcast(
    const Shape& shape, HloInstruction* operand) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kBitcast, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  auto instruction = absl::WrapUnique(new HloReduceInstruction(
      shape, {operand, init_value}, dimensions_to_reduce, reduce_computation));
  return std::move(instruction);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  std::vector<HloInstruction*> all_args;
  all_args.reserve(operands.size() * 2);
  all_args.insert(all_args.end(), operands.begin(), operands.end());
  all_args.insert(all_args.end(), init_values.begin(), init_values.end());
  return absl::make_unique<HloReduceInstruction>(
      shape, all_args, dimensions_to_reduce, reduce_computation);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation) {
  return absl::make_unique<HloReduceWindowInstruction>(
      shape, operand, init_value, window, reduce_computation);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormTraining(const Shape& shape,
                                        HloInstruction* operand,
                                        HloInstruction* scale,
                                        HloInstruction* offset, float epsilon,
                                        int64 feature_index) {
  return absl::make_unique<HloBatchNormTrainingInstruction>(
      shape, operand, scale, offset, epsilon, feature_index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormInference(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
    float epsilon, int64 feature_index) {
  return absl::make_unique<HloBatchNormInferenceInstruction>(
      shape, operand, scale, offset, mean, variance, epsilon, feature_index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBatchNormGrad(const Shape& shape, HloInstruction* operand,
                                    HloInstruction* scale, HloInstruction* mean,
                                    HloInstruction* variance,
                                    HloInstruction* grad_output, float epsilon,
                                    int64 feature_index) {
  return absl::make_unique<HloBatchNormGradInstruction>(
      shape, operand, scale, mean, variance, grad_output, epsilon,
      feature_index);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateSelectAndScatter(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter) {
  return absl::make_unique<HloSelectAndScatterInstruction>(
      shape, operand, select, window, source, init_value, scatter);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBroadcast(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64> broadcast_dimensions) {
  return absl::make_unique<HloBroadcastInstruction>(shape, operand,
                                                    broadcast_dimensions);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetDimensionSize(const Shape& shape,
                                       HloInstruction* operand,
                                       int64 dimension) {
  return absl::make_unique<HloGetDimensionSizeInstruction>(shape, operand,
                                                           dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateSetDimensionSize(const Shape& shape,
                                       HloInstruction* operand,
                                       HloInstruction* val, int64 dimension) {
  return absl::make_unique<HloSetDimensionSizeInstruction>(shape, operand, val,
                                                           dimension);
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateBroadcastSequence(
    const Shape& output_shape, HloInstruction* operand,
    const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
        adder) {
  CHECK(ShapeUtil::IsScalar(operand->shape()) ||
        operand->shape().rank() == output_shape.rank());
  Shape broadcast_shape = ShapeUtil::ChangeElementType(
      output_shape, operand->shape().element_type());
  // Do explicit broadcast for scalar.
  if (ShapeUtil::IsScalar(operand->shape())) {
    auto broadcast =
        HloInstruction::CreateBroadcast(broadcast_shape, operand, {});
    broadcast->set_metadata(operand->metadata());
    if (operand->has_sharding()) {
      broadcast->set_sharding(operand->sharding());
    }
    broadcast->set_frontend_attributes(operand->frontend_attributes());
    return broadcast;
  }
  // Do explicit broadcast for degenerate broadcast.
  std::vector<int64> broadcast_dimensions;
  std::vector<int64> reshaped_dimensions;
  for (int i = 0; i < operand->shape().rank(); i++) {
    if (operand->shape().dimensions(i) == output_shape.dimensions(i)) {
      broadcast_dimensions.push_back(i);
      reshaped_dimensions.push_back(operand->shape().dimensions(i));
    } else {
      CHECK_EQ(operand->shape().dimensions(i), 1)
          << "An explicit broadcast sequence requires the broadcasted "
             "dimensions to be trivial; operand: "
          << operand->ToString() << "; output_shape: " << output_shape;
    }
  }
  // Eliminate the size one dimensions.
  HloInstruction* reshaped_operand = adder(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(operand->shape().element_type(),
                           reshaped_dimensions),
      operand));
  reshaped_operand->set_metadata(operand->metadata());
  if (operand->has_sharding()) {
    reshaped_operand->set_sharding(operand->sharding());
  }
  reshaped_operand->set_frontend_attributes(operand->frontend_attributes());
  // Broadcast 'reshape' up to the larger size.
  auto broadcast = HloInstruction::CreateBroadcast(
      broadcast_shape, reshaped_operand, broadcast_dimensions);
  broadcast->set_metadata(operand->metadata());
  if (operand->has_sharding()) {
    broadcast->set_sharding(operand->sharding());
  }
  broadcast->set_frontend_attributes(operand->frontend_attributes());
  return broadcast;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreatePad(
    const Shape& shape, HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config) {
  return absl::make_unique<HloPadInstruction>(shape, operand, padding_value,
                                              padding_config);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReshape(
    const Shape& shape, HloInstruction* operand, int64 inferred_dimension) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape),
           ShapeUtil::ElementsIn(operand->shape()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << " operand: " << ShapeUtil::HumanString(operand->shape());

  return absl::make_unique<HloReshapeInstruction>(shape, operand,
                                                  inferred_dimension);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTranspose(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64> dimensions) {
  return absl::make_unique<HloTransposeInstruction>(shape, operand, dimensions);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSort(
    const Shape& shape, int64 dimension,
    absl::Span<HloInstruction* const> operands, HloComputation* compare,
    bool is_stable) {
  return absl::make_unique<HloSortInstruction>(shape, dimension, operands,
                                               compare, is_stable);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root) {
  return absl::make_unique<HloFusionInstruction>(shape, fusion_kind,
                                                 fused_root);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind,
    absl::Span<HloInstruction* const> operands,
    HloComputation* fusion_computation) {
  return absl::make_unique<HloFusionInstruction>(shape, fusion_kind, operands,
                                                 fusion_computation);
}

void HloInstruction::set_single_sharding(const HloSharding& sharding) {
  CHECK(!sharding.IsTuple()) << sharding;
  if (shape().IsTuple()) {
    set_sharding(HloSharding::Tuple(sharding.GetAsShapeTree(shape())));
  } else {
    set_sharding(sharding);
  }
}

void HloInstruction::SetupDerivedInstruction(
    HloInstruction* derived_instruction) const {
  if (sharding_ != nullptr && ShapeUtil::CompatibleIgnoringElementType(
                                  shape_, derived_instruction->shape())) {
    // Only copy sharding if the shape of the two instruction is compatible
    // because copying it between differently shaped instructions can produce
    // invalid shardings.
    derived_instruction->set_sharding(*sharding_);
  } else {
    derived_instruction->clear_sharding();
  }
  derived_instruction->set_metadata(metadata_);
  derived_instruction->set_frontend_attributes(frontend_attributes_);
}

bool HloInstruction::HasSideEffectNoRecurse() const {
  switch (opcode_) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kTrace:
      return true;
    case HloOpcode::kAllReduce:
      return channel_id().has_value() ||
             Cast<HloAllReduceInstruction>(this)->constrain_layout();
    case HloOpcode::kCustomCall:
      return Cast<HloCustomCallInstruction>(this)
          ->custom_call_has_side_effect();
    default:
      return false;
  }
}

bool HloInstruction::HasSideEffect() const {
  if (HasSideEffectNoRecurse()) {
    return true;
  }
  // Check if any of the called computations has a side effect.
  for (const auto& computation : called_computations()) {
    if (computation->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* computation) {
  std::unique_ptr<HloInstruction> instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kCall, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->called_computations_.push_back(computation);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, string opaque) {
  return absl::make_unique<HloCustomCallInstruction>(
      shape, operands, custom_call_target, std::move(opaque));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target,
    absl::Span<const Shape> operand_shapes_with_layout, string opaque) {
  return absl::make_unique<HloCustomCallInstruction>(
      shape, operands, custom_call_target, std::move(opaque),
      operand_shapes_with_layout);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTuple(
    absl::Span<HloInstruction* const> elements) {
  std::vector<Shape> element_shapes;
  for (auto element : elements) {
    element_shapes.push_back(element->shape());
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);
  return CreateVariadic(tuple_shape, HloOpcode::kTuple, elements);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateGather(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64> slice_sizes, bool indices_are_sorted) {
  return absl::make_unique<HloGatherInstruction>(
      shape, operand, start_indices, gather_dim_numbers, slice_sizes,
      indices_are_sorted);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateScatter(
    const Shape& shape, HloInstruction* operand,
    HloInstruction* scatter_indices, HloInstruction* updates,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices) {
  return absl::make_unique<HloScatterInstruction>(
      shape, operand, scatter_indices, updates, update_computation,
      scatter_dim_numbers, indices_are_sorted, unique_indices);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDomain(
    const Shape& shape, HloInstruction* operand,
    std::unique_ptr<DomainMetadata> operand_side_metadata,
    std::unique_ptr<DomainMetadata> user_side_metadata) {
  return absl::make_unique<HloDomainInstruction>(
      shape, operand, std::move(operand_side_metadata),
      std::move(user_side_metadata));
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  VLOG(3) << "CloneWithNewOperands:\n  " << ToString();
  VLOG(3) << "  new operands:";
  for (const HloInstruction* new_operand : new_operands) {
    VLOG(3) << "    %" << new_operand->name();
  }

  std::unique_ptr<HloInstruction> clone;
  // Explicitly call the factory for the instruction type. This is more robust
  // in the face of code changes than copying fields explicitly. This also
  // properly sets the user fields of the operands.
  switch (opcode_) {
    // Ops migrated to subclasses.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kFft:
    case HloOpcode::kCompare:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kMap:
    case HloOpcode::kSlice:
    case HloOpcode::kConstant:
    case HloOpcode::kTrace:
    case HloOpcode::kFusion:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kConvolution:
    case HloOpcode::kCustomCall:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kPad:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kSort:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kIota:
    case HloOpcode::kDot:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      clone = CloneWithNewOperandsImpl(shape, new_operands, context);
      break;
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateUnary(shape, opcode_, new_operands[0]);
      break;
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateBinary(shape, opcode_, new_operands[0], new_operands[1]);
      break;
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect:
      CHECK_EQ(new_operands.size(), 3);
      clone = CreateTernary(shape, opcode_, new_operands[0], new_operands[1],
                            new_operands[2]);
      break;
    // Other supported ops.
    case HloOpcode::kCall:
      clone = CreateCall(shape, new_operands, to_apply());
      break;
    case HloOpcode::kConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kBitcastConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateBitcastConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kDynamicUpdateSlice:
      clone = CreateDynamicUpdateSlice(shape, new_operands[0], new_operands[1],
                                       new_operands.subspan(2));
      break;
    case HloOpcode::kTuple:
      clone = CreateTuple(new_operands);
      *clone->mutable_shape() = shape;
      break;
    case HloOpcode::kWhile:
      CHECK_EQ(new_operands.size(), 1);
      clone =
          CreateWhile(shape, while_condition(), while_body(), new_operands[0]);
      break;
    case HloOpcode::kConditional:
      CHECK_EQ(new_operands.size(), branch_count() + 1);
      clone = CreateConditional(shape, new_operands[0],
                                absl::MakeSpan(branch_computations()),
                                new_operands.subspan(1));
      break;
    case HloOpcode::kAfterAll:
      if (new_operands.empty()) {
        clone = CreateToken();
      } else {
        clone = CreateAfterAll(new_operands);
      }
      break;
    case HloOpcode::kAddDependency:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateAddDependency(new_operands[0], new_operands[1]);
      break;
    case HloOpcode::kReplicaId:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreateReplicaId();
      *clone->mutable_shape() = shape;
      break;
    case HloOpcode::kPartitionId:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreatePartitionId();
      *clone->mutable_shape() = shape;
      break;
  }
  // SetupDerivedInstruction will setup the precision_config_ field.
  SetupDerivedInstruction(clone.get());
  clone->set_parent(parent_);
  clone->set_outer_dimension_partitions(outer_dimension_partitions_);
  clone->set_raw_backend_config_string(backend_config_);
  if (context != nullptr) {
    context->MapInstruction(this, clone.get());
    clone->ReplaceCalledComputations([&](HloComputation* callee) {
      return callee->parent() != context->module()
                 ? context->module()->DeepCloneComputation(callee, context)
                 : callee;
    });
  }
  return clone;
}

HloInstruction::~HloInstruction() {
  // Detach from operands. An instruction may be repeated as an operand. To
  // avoid calling RemoveUser twice on the same operand, check before remove.
  for (int64 operand_num = 0; operand_num < operand_count(); ++operand_num) {
    HloInstruction* operand = operands_[operand_num];
    if (operand == nullptr) {
      continue;
    }
    if (operand->user_map_.find(this) != operand->user_map_.end()) {
      operand->RemoveUser(this);
    }
    operands_[operand_num] = nullptr;
  }

  // Update users. Set `nullptr` to the corresponding operand slot for users.
  for (auto& user : this->users()) {
    for (int i = 0; i < user->operand_count(); ++i) {
      if (user->operands_[i] == this) {
        user->operands_[i] = nullptr;
      }
    }
  }
}

std::unique_ptr<HloInstruction> HloInstruction::Clone(
    const string& suffix, HloCloneContext* context) const {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewOperands(shape_, operands_, context);
  if (suffix.empty()) {
    clone->name_ = name();
  } else {
    // If an instruction is cloned multiple times avoid names like
    // foo.suffix.suffix.suffix. Instead of repeating the suffix add a numeric
    // suffix. Specifically, the clone of foo.suffix is named foo.suffix2, the
    // clone of foo.suffix2 is named foo.suffix3 and so on.
    const string dot_suffix = "." + suffix;
    size_t index = name().rfind(dot_suffix);
    if (index == string::npos) {
      // Existing name does not include ".suffix".
      clone->name_ = name() + dot_suffix;
    } else {
      // Existing name includes ".suffix". Determine if substring after
      // ".suffix" is numeric and should be replaced with an incremented number.
      string after_suffix = name().substr(index + dot_suffix.size());
      if (after_suffix.empty()) {
        // Existing name ends in ".suffix". New name should end in ".suffix2".
        clone->name_ = name() + "2";
      } else {
        // If names ends with .suffix[0-9]+ then replace with a suffix with the
        // numeric value incremented.
        int64 numeric_suffix;
        if (absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
          clone->name_ =
              StrCat(name().substr(0, index), dot_suffix, numeric_suffix + 1);
        } else {
          // Substring after ".suffix" is non-numeric.
          clone->name_ = name() + dot_suffix;
        }
      }
    }
  }
  return clone;
}

std::pair<const HloInstruction*, ShapeIndex>
HloInstruction::LatestNonGteAncestorAndIndex() const {
  const HloInstruction* hlo = this;
  ShapeIndex index;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    index.push_back(hlo->tuple_index());
    hlo = hlo->operand(0);
  }

  // We built up index in the reverse order from what we want.
  std::reverse(index.begin(), index.end());

  return {hlo, index};
}

const HloInstruction* HloInstruction::LatestNonGteAncestor() const {
  const HloInstruction* hlo = this;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    hlo = hlo->operand(0);
  }
  return hlo;
}

const HloInstruction* HloInstruction::operand(int64 i) const {
  return operands_.at(i);
}

HloInstruction* HloInstruction::mutable_operand(int64 i) {
  CHECK(operands_[i] != nullptr);
  return operands_.at(i);
}

int64 HloInstruction::operand_index(const HloInstruction* target) const {
  for (int64 i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      return i;
    }
  }
  LOG(FATAL) << "target was not an operand: " << target->ToString();
}

HloInstruction::InstructionVector HloInstruction::unique_operands() const {
  InstructionVector unique;
  absl::flat_hash_set<const HloInstruction*> seen;
  for (HloInstruction* operand : operands()) {
    if (seen.insert(operand).second) {
      unique.push_back(operand);
    }
  }
  return unique;
}

Status HloInstruction::AddControlDependencyTo(HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  if (!absl::c_linear_search(control_successors_, instruction)) {
    control_successors_.push_back(instruction);
    TF_RET_CHECK(
        !absl::c_linear_search(instruction->control_predecessors_, this));
    instruction->control_predecessors_.push_back(this);
  }
  return Status::OK();
}

Status HloInstruction::RemoveControlDependencyTo(HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  TF_RETURN_IF_ERROR(EraseElementFromVector(&control_successors_, instruction));
  TF_RETURN_IF_ERROR(
      EraseElementFromVector(&instruction->control_predecessors_, this));
  return Status::OK();
}

Status HloInstruction::DropAllControlDeps() {
  for (auto* ctrl_succ : control_successors_) {
    TF_RETURN_IF_ERROR(
        EraseElementFromVector(&ctrl_succ->control_predecessors_, this));
  }
  for (auto* ctrl_pred : control_predecessors_) {
    TF_RETURN_IF_ERROR(
        EraseElementFromVector(&ctrl_pred->control_successors_, this));
  }
  control_successors_.clear();
  control_predecessors_.clear();
  return Status::OK();
}

Status HloInstruction::CopyAllControlDepsFrom(const HloInstruction* inst) {
  for (auto* ctrl_pred : inst->control_predecessors()) {
    TF_RETURN_IF_ERROR(ctrl_pred->AddControlDependencyTo(this));
  }

  for (auto* ctrl_succ : inst->control_successors()) {
    TF_RETURN_IF_ERROR(this->AddControlDependencyTo(ctrl_succ));
  }

  return Status::OK();
}

void HloInstruction::AppendOperand(HloInstruction* operand) {
  operands_.push_back(operand);
  operand->AddUser(this);
}

void HloInstruction::RemoveOperandsAtAscendingIndices(
    absl::Span<const int> ascending_indices) {
  if (ascending_indices.empty()) {
    return;
  }
  int next_index = 0;
  int removed_count = 0;
  for (int to_remove : ascending_indices) {
    while (next_index < to_remove) {
      operands_[next_index - removed_count] = operands_[next_index];
      ++next_index;
    }
    CHECK_LT(to_remove, operands_.size());
    ++removed_count;
    ++next_index;
  }
  while (next_index < operands_.size()) {
    operands_[next_index - removed_count] = operands_[next_index];
    ++next_index;
  }
  CHECK_EQ(removed_count, ascending_indices.size());
  operands_.resize(operands_.size() - removed_count);
}

void HloInstruction::AddUser(HloInstruction* user) {
  if (!ContainsKey(user_map_, user)) {
    user_map_.emplace(user, users_.size());
    users_.push_back(user);
  }
}

int64 HloInstruction::UserId(HloInstruction* user) {
  auto result = user_map_.find(user);
  CHECK(result != user_map_.end());
  return result->second;
}

bool HloInstruction::HasConstantOperand() const {
  for (const HloInstruction* operand : operands_) {
    if (operand->IsConstant()) {
      return true;
    }
  }
  return false;
}

bool HloInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  // Perform opcode specific checks.
  switch (opcode()) {
    // The result of these instructions only depend upon their opcode and
    // operands.
    case HloOpcode::kAbs:
    case HloOpcode::kAtan2:
    case HloOpcode::kAdd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kAnd:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kTuple:
    case HloOpcode::kTupleSelect:
      return true;

    // This opcode has complex or special behavior so just return false.
    case HloOpcode::kAfterAll:
    case HloOpcode::kAddDependency:
      return false;

    // Remaining instructions with special values.
    case HloOpcode::kCall:
      return eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kConditional:
      for (int j = 0; j < branch_count(); ++j) {
        if (!eq_computations(branch_computation(j),
                             other.branch_computation(j))) {
          return false;
        }
      }
      return true;
    case HloOpcode::kWhile:
      return (eq_computations(while_body(), other.while_body()) &&
              eq_computations(while_condition(), other.while_condition()));

    // Ops migrated to subclasses should never come to this line.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kFft:
    case HloOpcode::kCompare:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kMap:
    case HloOpcode::kSlice:
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kTrace:
    case HloOpcode::kFusion:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kConvolution:
    case HloOpcode::kCustomCall:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kPad:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kDot:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      LOG(FATAL) << "Base class impl called for opcode with subclass: "
                 << opcode();
  }
  return false;
}

static uint64 HashOperand(const HloInstruction* hlo) {
  return ShapeUtil::Hash(hlo->shape());
}

uint64 HloInstruction::Hash(
    const std::function<uint64(const HloInstruction*)>& hash_operand) const {
  using tensorflow::Hash64Combine;

  uint64 hash_value = Hash64Combine(0, static_cast<uint64>(opcode()));
  hash_value = Hash64Combine(hash_value, ShapeUtil::Hash(shape()));

  if (!IsCrossModuleAllReduce()) {
    if (!operands().empty()) {
      for (size_t i = 0; i < operands().size(); ++i) {
        hash_value = Hash64Combine(hash_value, hash_operand(operand(i)));
      }
    }
  }

  hash_value = Hash64Combine(hash_value, InnerHash());
  return hash_value;
}

uint64 HloInstruction::Hash() const {
  // Use HashOperand as an argument to prevent non-termination.
  return Hash(HashOperand);
}

uint64 HloInstruction::InnerHash() const { return 13; }

void HloInstruction::RemoveUser(HloInstruction* user) {
  auto map_it = user_map_.find(user);
  CHECK(map_it != user_map_.end());

  const int64 index = map_it->second;
  CHECK_EQ(users_[index], user);

  // Move the last user into the position of the removed user.
  users_[index] = users_.back();
  user_map_[users_.back()] = index;

  // Remove the user from the map and drop the last slot from the vector what
  // have been moved to the position of the original user.
  user_map_.erase(map_it);
  users_.pop_back();
}

Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                      HloInstruction* new_producer) {
  TF_RET_CHECK(
      ShapeUtil::CompatibleIgnoringFpPrecision(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  return ReplaceUseWithDifferentShape(user, new_producer);
}

Status HloInstruction::ReplaceUseWithDifferentShape(
    HloInstruction* user, HloInstruction* new_producer) {
  VLOG(3) << "Replacing uses of " << name() << " in " << user->name()
          << " with " << new_producer->name();

  RemoveUser(user);

  TF_RET_CHECK(absl::c_count(user->operands_, this) >= 0);
  std::replace(user->operands_.begin(), user->operands_.end(), this,
               new_producer);
  new_producer->AddUser(user);
  // Custom fusions may not be able to handle deduplicated operands.
  if (user->opcode() == HloOpcode::kFusion) {
    TF_RETURN_IF_ERROR(
        Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
  }
  return Status::OK();
}

Status HloInstruction::ReplaceOperandWith(int64 operand_num,
                                          HloInstruction* new_operand) {
  auto old_operand = operand(operand_num);
  TF_RET_CHECK(ShapeUtil::CompatibleIgnoringFpPrecision(old_operand->shape(),
                                                        new_operand->shape()))
      << old_operand->shape() << " is not compatible with "
      << new_operand->shape();
  return ReplaceOperandWithDifferentShape(operand_num, new_operand);
}

Status HloInstruction::ReplaceOperandWithDifferentShape(
    int64 operand_num, HloInstruction* new_operand) {
  TF_RET_CHECK(operand_num >= 0);
  TF_RET_CHECK(operand_num < operand_count());
  HloInstruction* old_operand = mutable_operand(operand_num);
  if (old_operand == new_operand) {
    return Status::OK();
  }

  operands_[operand_num] = new_operand;

  VLOG(3) << "Replacing operand " << operand_num << " of " << name() << " with "
          << new_operand->name() << ", was " << old_operand->name();

  if (!absl::c_linear_search(operands_, old_operand)) {
    old_operand->RemoveUser(this);
  }
  new_operand->AddUser(this);
  return Status::OK();
}

Status HloInstruction::ReplaceAllUsesWith(HloInstruction* new_producer) {
  TF_RET_CHECK(
      ShapeUtil::CompatibleIgnoringFpPrecision(shape(), new_producer->shape()))
      << shape() << " is not compatible with " << new_producer->shape();
  return ReplaceAllUsesWithDifferentShape(new_producer);
}

Status HloInstruction::ReplaceAllUsesWithDifferentShape(
    HloInstruction* new_producer) {
  bool new_producer_is_user = false;
  for (HloInstruction* user : users()) {
    if (user == new_producer) {
      // It's possible that new_producer is a user of this instruction as might
      // be the case when replacing an instruction with a kCopy of itself. In
      // this case, don't do the replacement to avoid creating a cycle in the
      // graph. new_producer remains the only user of this instruction.
      new_producer_is_user = true;
    } else {
      std::replace(user->operands_.begin(), user->operands_.end(), this,
                   new_producer);
      new_producer->AddUser(user);
      if (user->opcode() == HloOpcode::kFusion) {
        TF_RETURN_IF_ERROR(
            Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
      }
    }
  }
  users_.clear();
  user_map_.clear();
  if (new_producer_is_user) {
    AddUser(new_producer);
  }
  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer,
                                  /*accept_different_shape=*/true);
  }

  return Status::OK();
}

HloComputation* HloInstruction::to_apply() const {
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
    case HloOpcode::kAllReduce:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
      CHECK_EQ(called_computations_.size(), 1);
      return called_computations_[0];
    default:
      LOG(FATAL) << "Invalid opcode for to_apply(): "
                 << HloOpcodeString(opcode());
  }
}

void HloInstruction::set_to_apply(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
    case HloOpcode::kAllReduce:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
      CHECK_EQ(called_computations_.size(), 1);
      called_computations_[0] = computation;
      break;
    default:
      LOG(FATAL) << "Invalid opcode for to_apply(): "
                 << HloOpcodeString(opcode());
  }
}

HloComputation* HloInstruction::while_condition() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations_[kConditionComputationIndex];
}

HloComputation* HloInstruction::while_body() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations_[kBodyComputationIndex];
}

void HloInstruction::set_while_condition(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  called_computations_[kConditionComputationIndex] = computation;
}

void HloInstruction::set_while_body(HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  called_computations_[kBodyComputationIndex] = computation;
}

HloInstruction* HloInstruction::while_init() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return operands_[0];
}

HloComputation* HloInstruction::true_computation() const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_EQ(PRED, operand(0)->shape().element_type());
  return called_computations_[kTrueComputationIndex];
}

HloComputation* HloInstruction::false_computation() const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_EQ(PRED, operand(0)->shape().element_type());
  return called_computations_[kFalseComputationIndex];
}

const std::vector<HloComputation*>& HloInstruction::branch_computations()
    const {
  CHECK(HloOpcode::kConditional == opcode_);
  return called_computations_;
}

int HloInstruction::branch_count() const {
  CHECK(HloOpcode::kConditional == opcode_);
  return called_computations_.size();
}

HloComputation* HloInstruction::branch_computation(int b) const {
  CHECK(HloOpcode::kConditional == opcode_);
  CHECK_GE(b, 0);
  CHECK_LT(b, called_computations_.size());
  return called_computations_[b];
}

void HloInstruction::set_branch_computation(int b,
                                            HloComputation* computation) {
  // Don't allow changing the computation for fused instructions so we don't
  // have to recompute called_instructions for the entire fusion instruction.
  CHECK(!IsFused());
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  called_computations_[b] = computation;
}

string HloInstruction::SignatureString() const {
  string operands =
      StrJoin(operands_, ", ", [](string* out, HloInstruction* operand) {
        StrAppend(out, ShapeUtil::HumanString(operand->shape()));
      });
  return StrCat("(", operands, ") -> ", ShapeUtil::HumanString(shape()));
}

string PrintName(const string& name, bool print_ids) {
  if (print_ids) {
    return name;
  } else {
    auto dot_position = name.find_first_of(".");
    return name.substr(0, dot_position);
  }
}

namespace {

using DFSStack = absl::InlinedVector<std::pair<int, HloInstruction*>, 16>;

string PrintNameInternal(const string& name, const HloPrintOptions& options) {
  return StrCat(options.print_percent() ? "%" : "",
                PrintName(name, options.print_ids()));
}

void PrintCycle(const HloInstruction* child, DFSStack* dfs_stack) {
  // This set contains HloInstructions from the top of `DFSStack` that might
  // belong to the cycle, i.e. if  DFSStack :=[back,...,child,...,top], then
  // `subgraph` := {child,...,top}.
  absl::flat_hash_set<const HloInstruction*> subgraph;
  while (!dfs_stack->empty() && dfs_stack->back().second != child) {
    subgraph.insert(dfs_stack->back().second);
    dfs_stack->pop_back();
  }
  // Start dfs at `child` and find a cycle with all nodes in `subgraph`.
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 16> dfs;
  dfs.push_back(child);
  while (!dfs.empty()) {
    bool found_next_instr = false;
    for (const auto& user : dfs.back()->users()) {
      if (user == child) {
        dfs.push_back(child);
        LOG(INFO) << "\n\nDirected cycle:\n  "
                  << absl::StrJoin(
                         dfs, "\n  ",
                         [](std::string* out, const HloInstruction* instr) {
                           out->append(instr->name());
                         });
        return;
      }
      if (!subgraph.contains(user) || visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      dfs.push_back(user);
      found_next_instr = true;
    }
    if (!found_next_instr) {
      dfs.pop_back();
    }
  }
}

}  // namespace

string HloInstruction::ToString(const HloPrintOptions& options) const {
  CanonicalNameMap new_map;
  return ToStringWithCanonicalNameMap(options, &new_map);
}

bool HloInstruction::IsElementwiseImpl(
    const absl::optional<int64>& operand_idx) const {
  switch (opcode_) {
    // Unary elementwise operations.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTanh:
      CHECK_EQ(1, operand_count());
      return true;

    // Binary elementwise operations, the same as in IsElementwiseBinary().
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      CHECK_EQ(2, operand_count());
      return true;

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
    case HloOpcode::kClamp:
      return true;

    case HloOpcode::kDynamicUpdateSlice:
      return operand_idx.has_value() && operand_idx.value() == 0;

    default:
      return false;
  }
}

bool HloInstruction::IsCrossModuleAllReduce() const {
  return opcode() == HloOpcode::kAllReduce && channel_id();
}

bool HloInstruction::IsCrossReplicaAllReduce() const {
  return opcode() == HloOpcode::kAllReduce && !channel_id();
}

string HloInstruction::ToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  string result = "";

  // Logic to print the instruction name (e.g. "%foo = ").
  if (options.canonicalize_instruction_names()) {
    if (options.is_in_nested_computation()) {
      // If we are canonicalizing instruction names and this is a top-level
      // HloInstruction::ToString() call, don't print an instruction name.
      StrAppend(&result,
                PrintNameInternal(canonical_name_map->LookupOrInsert(name()),
                                  options),
                " = ");
    }
  } else {
    StrAppend(&result, PrintNameInternal(name(), options), " = ");
  }

  // Print shape.
  if (options.include_layout_in_shapes()) {
    StrAppend(&result, ShapeUtil::HumanStringWithLayout(shape()));
  } else {
    StrAppend(&result, ShapeUtil::HumanString(shape()));
  }

  // Print opcode, operand(s).
  StrAppend(&result, " ", HloOpcodeString(opcode()), "(",
            OperandsToStringWithCanonicalNameMap(options, canonical_name_map),
            ")");

  // Print additional attributes. If an instruction contains a subcomputation,
  // the subcomputation is also printed here.
  for (const string& extra : ExtraAttributesToString(options)) {
    StrAppend(&result, ", ", extra);
  }

  if (options.print_metadata() &&
      (!metadata_.op_type().empty() || !metadata_.op_name().empty() ||
       !metadata_.source_file().empty())) {
    StrAppend(&result, ", metadata={", xla::OpMetadataToString(metadata_), "}");
  }
  if (options.print_backend_config() && !backend_config_.empty()) {
    StrAppend(&result, ", backend_config=\"", CEscape(backend_config_), "\"");
  }
  return result;
}

string HloInstruction::OperandsToString(const HloPrintOptions& options) const {
  CanonicalNameMap new_map;
  return OperandsToStringWithCanonicalNameMap(options, &new_map);
}

string HloInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  string operands;
  absl::Span<HloInstruction* const> slice(operands_);
  const int64 kMaxOperandsToShowIfCompact = 4;
  if (options.compact_operands() &&
      slice.size() > kMaxOperandsToShowIfCompact) {
    slice.remove_suffix(slice.size() - kMaxOperandsToShowIfCompact);
  }
  operands = StrJoin(slice, ", ", [&](string* out, HloInstruction* operand) {
    // If operand is already been deleted, put `null` to the string output.
    if (operand == nullptr) {
      StrAppend(out, "null ");
      return;
    }
    std::vector<string> str;
    if (options.print_operand_shape()) {
      if (options.include_layout_in_shapes()) {
        str.push_back(ShapeUtil::HumanStringWithLayout(operand->shape()));
      } else {
        str.push_back(ShapeUtil::HumanString(operand->shape()));
      }
    }

    // In a top-level HloInstruction::ToString() call, the operand name is not
    // part of the canonical string.
    if (options.canonicalize_instruction_names() &&
        options.is_in_nested_computation()) {
      str.push_back(PrintNameInternal(
          canonical_name_map->LookupOrInsert(operand->name()), options));
    } else if (options.print_operand_names()) {
      str.push_back(PrintNameInternal(operand->name(), options));
    }
    StrAppend(out, StrJoin(str, " "));
  });
  const int64 remaining = operands_.size() - slice.size();
  if (slice.size() != operands_.size()) {
    StrAppend(&operands, ", ...(+", remaining, ")");
  }
  return operands;
}

std::vector<string> HloInstruction::ExtraAttributesToString(
    const HloPrintOptions& options) const {
  std::vector<string> extra = ExtraAttributesToStringImpl(options);

  if (options.print_subcomputation_mode() ==
      HloPrintOptions::PrintSubcomputationMode::kNameOnly) {
    if (opcode() == HloOpcode::kWhile) {
      extra.push_back(StrCat(
          "condition=", PrintNameInternal(while_condition()->name(), options)));
      extra.push_back(
          StrCat("body=", PrintNameInternal(while_body()->name(), options)));
    } else if (opcode() == HloOpcode::kSelectAndScatter) {
      extra.push_back(
          StrCat("select=", PrintNameInternal(select()->name(), options)));
      extra.push_back(
          StrCat("scatter=", PrintNameInternal(scatter()->name(), options)));
    } else if (opcode() == HloOpcode::kConditional) {
      if (operand(0)->shape().element_type() == PRED) {
        extra.push_back(
            StrCat("true_computation=",
                   PrintNameInternal(true_computation()->name(), options)));
        extra.push_back(
            StrCat("false_computation=",
                   PrintNameInternal(false_computation()->name(), options)));
      } else {
        extra.push_back(StrCat(
            "branch_computations={",
            StrJoin(branch_computations(), ", ",
                    [&](string* out, const HloComputation* computation) {
                      StrAppend(
                          out, PrintNameInternal(computation->name(), options));
                    }),
            "}"));
      }
    } else if (opcode() == HloOpcode::kCall || opcode() == HloOpcode::kMap ||
               opcode() == HloOpcode::kReduceWindow ||
               opcode() == HloOpcode::kReduce ||
               opcode() == HloOpcode::kAllReduce ||
               opcode() == HloOpcode::kScatter ||
               opcode() == HloOpcode::kSort) {
      extra.push_back(
          StrCat("to_apply=", PrintNameInternal(to_apply()->name(), options)));
    } else if (!called_computations().empty()) {
      extra.push_back(StrCat(
          "calls=",
          StrJoin(called_computations(), ", ",
                  [&](string* out, const HloComputation* computation) {
                    StrAppend(out,
                              PrintNameInternal(computation->name(), options));
                  })));
    }
  } else if (options.print_subcomputation_mode() ==
             HloPrintOptions::PrintSubcomputationMode::kFullBodies) {
    HloPrintOptions new_options = options;
    new_options.set_is_in_nested_computation(true);
    switch (opcode()) {
      case HloOpcode::kWhile:
        extra.push_back(
            StrCat("condition=\n", while_condition()->ToString(new_options)));
        extra.push_back(StrCat("body=\n", while_body()->ToString(new_options)));
        break;
      case HloOpcode::kSelectAndScatter:
        extra.push_back(StrCat("select=\n", select()->ToString(new_options)));
        extra.push_back(StrCat("scatter=\n", scatter()->ToString(new_options)));
        break;
      case HloOpcode::kConditional:
        if (operand(0)->shape().element_type() == PRED) {
          extra.push_back(StrCat("true_computation=\n",
                                 true_computation()->ToString(new_options)));
          extra.push_back(StrCat("false_computation=\n",
                                 false_computation()->ToString(new_options)));
        } else {
          extra.push_back(StrCat(
              "branch_computations={\n",
              StrJoin(branch_computations(), ",\n",
                      [&](string* out, const HloComputation* computation) {
                        StrAppend(out, computation->ToString(new_options));
                      }),
              "\n}"));
        }
        break;
      case HloOpcode::kCall:
      case HloOpcode::kMap:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kReduce:
      case HloOpcode::kAllReduce:
      case HloOpcode::kScatter:
      case HloOpcode::kSort:
        extra.push_back(
            StrCat("to_apply=\n", to_apply()->ToString(new_options)));
        break;
      default:
        if (!called_computations().empty()) {
          extra.push_back(StrCat(
              "calls=\n",
              StrJoin(called_computations(), ", ",
                      [&](string* out, const HloComputation* computation) {
                        StrAppend(out, computation->ToString(new_options));
                      })));
        }
        break;
    }
  }

  if (has_sharding()) {
    extra.push_back(StrCat("sharding=", sharding().ToString()));
  }
  if (!frontend_attributes_.map().empty()) {
    extra.push_back(StrCat("frontend_attributes=",
                           FrontendAttributesToString(frontend_attributes_)));
  }
  if (!outer_dimension_partitions_.empty()) {
    extra.push_back(absl::StrFormat("outer_dimension_partitions={%s}",
                                    StrJoin(outer_dimension_partitions_, ",")));
  }

  if (options.print_control_dependencies() && !control_predecessors_.empty()) {
    extra.push_back(StrCat("control-predecessors={",
                           StrJoin(control_predecessors_, ", ",
                                   [&](string* out, HloInstruction* pre) {
                                     StrAppend(out, PrintNameInternal(
                                                        pre->name(), options));
                                   }),
                           "}"));
  }

  return extra;
}

string HloInstruction::ToShortString() const {
  return StrCat("%", name(), " = ", HloOpcodeString(opcode()), "(",
                StrJoin(operands_, ", ",
                        [](string* out, HloInstruction* operand) {
                          StrAppend(out, "%", operand->name());
                        }),
                ")");
}

HloInstructionProto HloInstruction::ToProto() const {
  HloInstructionProto proto;
  CHECK(unique_id_ != -1)
      << "This instruction does not have a valid id. Please make sure the "
         "instruction is inside a module before dumping it.";
  proto.set_id(unique_id_);
  proto.set_name(name_);
  proto.set_opcode(HloOpcodeString(opcode_));
  *proto.mutable_shape() = shape_.ToProto();
  for (const HloInstruction* operand : operands_) {
    proto.add_operand_ids(operand->unique_id());
  }
  for (const HloInstruction* control : control_predecessors_) {
    proto.add_control_predecessor_ids(control->unique_id());
  }

  *proto.mutable_metadata() = metadata_;
  proto.set_backend_config(backend_config_);
  if (opcode() != HloOpcode::kFusion) {
    for (const HloComputation* computation : called_computations_) {
      proto.add_called_computation_ids(computation->unique_id());
    }
  }

  if (has_sharding()) {
    *proto.mutable_sharding() = sharding().ToProto();
  }
  if (!outer_dimension_partitions_.empty()) {
    for (const auto& idx : outer_dimension_partitions_) {
      proto.mutable_outer_dimension_partitions()->Add(idx);
    }
  }

  *proto.mutable_frontend_attributes() = frontend_attributes_;

  return proto;
}

string HloInstruction::ToCategory() const {
  if (opcode() == HloOpcode::kTranspose || opcode() == HloOpcode::kCopy ||
      opcode() == HloOpcode::kReshape) {
    return "data formatting";
  }

  if (IsElementwise()) {
    return "non-fusion elementwise";
  }

  return HloOpcodeString(opcode());
}

HloInstruction* HloInstruction::tracing() const { return trace_instruction_; }

void HloInstruction::set_tracing(HloInstruction* trace_instruction) {
  trace_instruction_ = trace_instruction;
}

bool HloInstruction::IsFused() const { return parent_->IsFusionComputation(); }

bool HloInstruction::IsInputFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kInput;
}

bool HloInstruction::IsLoopFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kLoop;
}

bool HloInstruction::IsOutputFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kOutput;
}

bool HloInstruction::IsCustomFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kCustom;
}

bool HloInstruction::IsFusible() const {
  // Instructions which are traced should not be fused.
  if (tracing()) {
    return false;
  }
  // Some kinds of instructions don't make sense to fuse.
  switch (opcode_) {
    case HloOpcode::kDomain:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kCall:
      return false;
    // Fusions are always fusible.
    case HloOpcode::kFusion:
    // Side effecting reduce and reduce window would be invalid HLO.
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return true;
    // Side effecting instructions cannot be fused.
    default:
      return !HasSideEffect();
  }
}

HloInstruction::HloInstruction(HloOpcode opcode, const Shape& shape)
    : unique_id_(-1),
      opcode_(opcode),
      shape_(shape),
      name_(HloOpcodeString(opcode)) {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
}

template <typename HloInstructionPtr>
Status HloInstruction::Visit(DfsHloVisitorBase<HloInstructionPtr>* visitor) {
  switch (opcode_) {
    case HloOpcode::kAbs:
      return visitor->HandleAbs(this);
    case HloOpcode::kAtan2:
      return visitor->HandleAtan2(this);
    case HloOpcode::kRoundNearestAfz:
      return visitor->HandleRound(this);
    case HloOpcode::kBatchNormTraining:
      return visitor->HandleBatchNormTraining(this);
    case HloOpcode::kBatchNormInference:
      return visitor->HandleBatchNormInference(this);
    case HloOpcode::kBatchNormGrad:
      return visitor->HandleBatchNormGrad(this);
    case HloOpcode::kSign:
      return visitor->HandleSign(this);
    case HloOpcode::kConstant:
      return visitor->HandleConstant(this);
    case HloOpcode::kGetTupleElement:
      return visitor->HandleGetTupleElement(this);
    case HloOpcode::kParameter:
      return visitor->HandleParameter(this);
    case HloOpcode::kCompare:
      return visitor->HandleCompare(this);
    case HloOpcode::kComplex:
      return visitor->HandleComplex(this);
    case HloOpcode::kAdd:
      return visitor->HandleAdd(this);
    case HloOpcode::kDivide:
      return visitor->HandleDivide(this);
    case HloOpcode::kSubtract:
      return visitor->HandleSubtract(this);
    case HloOpcode::kMaximum:
      return visitor->HandleMaximum(this);
    case HloOpcode::kMinimum:
      return visitor->HandleMinimum(this);
    case HloOpcode::kAnd:
      return visitor->HandleAnd(this);
    case HloOpcode::kOr:
      return visitor->HandleOr(this);
    case HloOpcode::kXor:
      return visitor->HandleXor(this);
    case HloOpcode::kShiftLeft:
      return visitor->HandleShiftLeft(this);
    case HloOpcode::kShiftRightArithmetic:
      return visitor->HandleShiftRightArithmetic(this);
    case HloOpcode::kShiftRightLogical:
      return visitor->HandleShiftRightLogical(this);
    case HloOpcode::kConcatenate:
      return visitor->HandleConcatenate(this);
    case HloOpcode::kConvert:
      return visitor->HandleConvert(this);
    case HloOpcode::kBitcastConvert:
      return visitor->HandleBitcastConvert(this);
    case HloOpcode::kCopy:
      return visitor->HandleCopy(this);
    case HloOpcode::kMultiply:
      return visitor->HandleMultiply(this);
    case HloOpcode::kDot:
      return visitor->HandleDot(this);
    case HloOpcode::kPower:
      return visitor->HandlePower(this);
    case HloOpcode::kRemainder:
      return visitor->HandleRemainder(this);
    case HloOpcode::kSelect:
      return visitor->HandleSelect(this);
    case HloOpcode::kTupleSelect:
      return visitor->HandleTupleSelect(this);
    case HloOpcode::kConvolution:
      return visitor->HandleConvolution(this);
    case HloOpcode::kFft:
      return visitor->HandleFft(this);
    case HloOpcode::kAllReduce:
      return visitor->HandleAllReduce(this);
    case HloOpcode::kAllToAll:
      return visitor->HandleAllToAll(this);
    case HloOpcode::kCollectivePermute:
      return visitor->HandleCollectivePermute(this);
    case HloOpcode::kReplicaId:
      return visitor->HandleReplicaId(this);
    case HloOpcode::kPartitionId:
      return visitor->HandlePartitionId(this);
    case HloOpcode::kTuple:
      return visitor->HandleTuple(this);
    case HloOpcode::kMap:
      return visitor->HandleMap(this);
    case HloOpcode::kClamp:
      return visitor->HandleClamp(this);
    case HloOpcode::kReduce:
      return visitor->HandleReduce(this);
    case HloOpcode::kReduceWindow:
      return visitor->HandleReduceWindow(this);
    case HloOpcode::kSelectAndScatter:
      return visitor->HandleSelectAndScatter(this);
    case HloOpcode::kNegate:
      return visitor->HandleNegate(this);
    case HloOpcode::kExp:
      return visitor->HandleExp(this);
    case HloOpcode::kExpm1:
      return visitor->HandleExpm1(this);
    case HloOpcode::kFloor:
      return visitor->HandleFloor(this);
    case HloOpcode::kCeil:
      return visitor->HandleCeil(this);
    case HloOpcode::kClz:
      return visitor->HandleClz(this);
    case HloOpcode::kLog:
      return visitor->HandleLog(this);
    case HloOpcode::kLog1p:
      return visitor->HandleLog1p(this);
    case HloOpcode::kTanh:
      return visitor->HandleTanh(this);
    case HloOpcode::kCos:
      return visitor->HandleCos(this);
    case HloOpcode::kSin:
      return visitor->HandleSin(this);
    case HloOpcode::kSqrt:
      return visitor->HandleSqrt(this);
    case HloOpcode::kRsqrt:
      return visitor->HandleRsqrt(this);
    case HloOpcode::kReal:
      return visitor->HandleReal(this);
    case HloOpcode::kImag:
      return visitor->HandleImag(this);
    case HloOpcode::kIsFinite:
      return visitor->HandleIsFinite(this);
    case HloOpcode::kNot:
      return visitor->HandleNot(this);
    case HloOpcode::kPopulationCount:
      return visitor->HandlePopulationCount(this);
    case HloOpcode::kBitcast:
      return visitor->HandleBitcast(this);
    case HloOpcode::kBroadcast:
      return visitor->HandleBroadcast(this);
    case HloOpcode::kPad:
      return visitor->HandlePad(this);
    case HloOpcode::kReshape:
      return visitor->HandleReshape(this);
    case HloOpcode::kTranspose:
      return visitor->HandleTranspose(this);
    case HloOpcode::kReverse:
      return visitor->HandleReverse(this);
    case HloOpcode::kReducePrecision:
      return visitor->HandleReducePrecision(this);
    case HloOpcode::kSlice:
      return visitor->HandleSlice(this);
    case HloOpcode::kDynamicSlice:
      return visitor->HandleDynamicSlice(this);
    case HloOpcode::kDynamicUpdateSlice:
      return visitor->HandleDynamicUpdateSlice(this);
    case HloOpcode::kSort:
      return visitor->HandleSort(this);
    case HloOpcode::kInfeed:
      return visitor->HandleInfeed(this);
    case HloOpcode::kOutfeed:
      return visitor->HandleOutfeed(this);
    case HloOpcode::kRng:
      return visitor->HandleRng(this);
    case HloOpcode::kRngGetAndUpdateState:
      return visitor->HandleRngGetAndUpdateState(this);
    case HloOpcode::kWhile:
      return visitor->HandleWhile(this);
    case HloOpcode::kFusion:
      return visitor->HandleFusion(this);
    case HloOpcode::kCall:
      return visitor->HandleCall(this);
    case HloOpcode::kConditional:
      return visitor->HandleConditional(this);
    case HloOpcode::kCustomCall:
      return visitor->HandleCustomCall(this);
    case HloOpcode::kCopyStart:
      return visitor->HandleCopyStart(this);
    case HloOpcode::kCopyDone:
      return visitor->HandleCopyDone(this);
    case HloOpcode::kRecv:
      return visitor->HandleRecv(this);
    case HloOpcode::kRecvDone:
      return visitor->HandleRecvDone(this);
    case HloOpcode::kSend:
      return visitor->HandleSend(this);
    case HloOpcode::kSendDone:
      return visitor->HandleSendDone(this);
    case HloOpcode::kGather:
      return visitor->HandleGather(this);
    case HloOpcode::kScatter:
      return visitor->HandleScatter(this);
    case HloOpcode::kDomain:
      return visitor->HandleDomain(this);
    case HloOpcode::kAfterAll:
      return visitor->HandleAfterAll(this);
    case HloOpcode::kAddDependency:
      return visitor->HandleAddDependency(this);
    case HloOpcode::kIota:
      return visitor->HandleIota(this);
    case HloOpcode::kGetDimensionSize:
      return visitor->HandleGetDimensionSize(this);
    case HloOpcode::kSetDimensionSize:
      return visitor->HandleSetDimensionSize(this);
    case HloOpcode::kTriangularSolve:
      return visitor->HandleTriangularSolve(this);
    case HloOpcode::kCholesky:
      return visitor->HandleCholesky(this);

    // These opcodes are not handled here.
    case HloOpcode::kTrace:
      return Status::OK();
  }
  return InternalError(
      "Unhandled HloOpcode for DfsHloVisitor: %s. This should not happen - "
      "please file a bug for XLA.",
      HloOpcodeString(opcode_));
}

// Explicit instantiations.
template Status HloInstruction::Visit(DfsHloVisitor* visitor);
template Status HloInstruction::Visit(ConstDfsHloVisitor* visitor);

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

using InternalCompareFunction =
    std::function<bool(std::pair<int, const HloInstruction*>,
                       std::pair<int, const HloInstruction*>)>;
template <typename Visitor>
static Status PostOrderDFS(HloInstruction* root, Visitor* visitor,
                           const InternalCompareFunction* operand_order,
                           bool ignore_control_predecessors) {
  // Calculating the instruction count within a module can be expensive on large
  // models so only do it if the visit state is empty. This will help when the
  // same visitor is reused across many computations of a single module.
  if (visitor->VisitStateCapacity() == 0) {
    visitor->ReserveVisitStates(root->GetModule()->instruction_count());
  }

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0) << current_id << ": " << current_node
                            << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    if (visit_state == Visitor::kVisiting) {
      dfs_stack.pop_back();

      TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
      VLOG(2) << "Visiting HLO %" << current_node->name();
      TF_RETURN_IF_ERROR(current_node->Visit(visitor));
      visitor->SetVisitState(current_id, Visitor::kVisited);
      TF_RETURN_IF_ERROR(visitor->Postprocess(current_node));
      continue;
    }

    visitor->SetVisitState(current_id, Visitor::kVisiting);

    const size_t old_dfs_stack_size = dfs_stack.size();
    for (HloInstruction* child : current_node->operands()) {
      if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        PrintCycle(child, &dfs_stack);
        return FailedPrecondition(
            "A cycle is detected while visiting instruction %s",
            current_node->ToString());
      }
    }

    if (!ignore_control_predecessors) {
      for (HloInstruction* child : current_node->control_predecessors()) {
        if (!TF_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
          PrintCycle(child, &dfs_stack);
          return FailedPrecondition(
              "A cycle is detected while visiting instruction %s",
              current_node->ToString());
        }
      }
    }

    if (operand_order != nullptr) {
      std::sort(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end(),
                *operand_order);
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return Status::OK();
}

template <typename HloInstructionPtr>
Status HloInstruction::Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                              bool call_finish_visit,
                              bool ignore_control_predecessors) {
  VLOG(3) << "HloInstruction::Accept(%" << name() << ")";
  TF_RETURN_IF_ERROR(
      PostOrderDFS(this, visitor, nullptr, ignore_control_predecessors));
  if (call_finish_visit) {
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
  }
  return Status::OK();
}

// Explicit instantiations.
template Status HloInstruction::Accept(DfsHloVisitor*, bool, bool);
template Status HloInstruction::Accept(ConstDfsHloVisitor*, bool, bool);

Status HloInstruction::AcceptWithOperandOrder(
    DfsHloVisitor* visitor, const CompareFunction& operand_order,
    bool call_finish_visit) {
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder(%" << name() << ")";
  InternalCompareFunction func = [&operand_order](
                                     std::pair<int, const HloInstruction*> a,
                                     std::pair<int, const HloInstruction*> b) {
    // Call the client's comparison function on the actual HloInstruction*
    // objects (ignoring the internal ids we also have in our stack entries)
    return operand_order(a.second, b.second);
  };
  TF_RETURN_IF_ERROR(PostOrderDFS(this, visitor, &func,
                                  /*ignore_control_predecessors=*/false));
  if (call_finish_visit) {
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder BEFORE FINISH VISIT";
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder AFTER FINISH VISIT";
  }
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder EXIT";
  return Status::OK();
}

const Shape& HloInstruction::shape() const { return shape_; }

absl::InlinedVector<int64, 4> HloInstruction::OperandIndices(
    const HloInstruction* operand) const {
  absl::InlinedVector<int64, 4> result;
  for (int64 i = 0; i < operand_count(); ++i) {
    if (this->operand(i) == operand) {
      result.push_back(i);
    }
  }
  return result;
}

bool HloInstruction::IsElementwiseBinary() const {
  return IsElementwise() && operand_count() == 2;
}

bool HloInstruction::IsElementwise() const {
  return IsElementwiseImpl(absl::nullopt);
}

bool HloInstruction::IsElementwiseOnOperand(int64 operand_idx) const {
  return IsElementwiseImpl(operand_idx);
}

// A helper class for memoized, recursive computation of HloOpcode::kFusion
// in HloInstruction::OperandElementUse below.
class HloInstruction::FusionReusesParamElements {
 public:
  using UseKind = HloInstruction::UseKind;

  // We could rather iterate backwards through fused_instructions_ here, as it
  // is in reverse postorder, and compute whether each fused instruction reuses
  // the value of this parameter, which would save stack space but not allow us
  // to finish early if we find a reuse.
  static UseKind Compute(int64 i, const HloInstruction& hlo) {
    absl::flat_hash_map<const HloInstruction*, UseKind> memoization_cache;
    return ComputeInternal(i, hlo, &memoization_cache);
  }

 private:
  static UseKind ComputeInternal(
      int64 i, const HloInstruction& hlo,
      absl::flat_hash_map<const HloInstruction*, UseKind>* cache) {
    if (auto hlo_param = DynCast<HloParameterInstruction>(&hlo)) {
      if (hlo_param->parameter_number() == i) {
        return UseKind::kUse;
      }
    }

    auto p = cache->emplace(&hlo, UseKind::kNoUse);
    auto value_it = p.first;
    const bool key_is_new = p.second;

    if (key_is_new) {
      for (int64 j = 0; j < hlo.operands_.size(); ++j) {
        UseKind old_val = value_it->second;

        // The next operation invalidates iterators.
        UseKind new_val =
            Fold(old_val,
                 FoldUseMandatory(hlo.OperandElementUse(j),
                                  ComputeInternal(i, *hlo.operand(j), cache)));

        // Re-acquire the iterator. We could work harder to do this only if
        // absolutely necessary, but this code is not hot enough to warrant
        // that.
        value_it = cache->find(&hlo);
        value_it->second = new_val;
      }
    }
    return value_it->second;
  }

  // Combines two UseKinds.
  //
  // This is the min operation on the lattice
  //
  //   kReuse < kUse < kNoUse.
  //
  // Two kUses uses which have different permutations count as kReuse.
  static UseKind Fold(UseKind a, UseKind b) {
    // Without loss of generality, let `b` be the operation with the larger use
    // kind.
    if (b.kind < a.kind) {
      std::swap(a, b);
    }
    // If the kinds are different, return the smaller one, namely `a`.
    if (a.kind != b.kind) {
      return a;
    }
    // If the kinds are both kUse, check that they're the same permutation.
    if (a.kind == UseKind::kUse && b.kind == UseKind::kUse &&
        a.permutation_instr != b.permutation_instr) {
      return UseKind::kReuse;
    }
    return a;  // They're the same.
  }

  // Combines two UseKinds differently than Fold().
  //
  // This is the min operation on the lattice
  //
  //   kNoUse < kReuse < kUse.
  //
  // If `a` and `b` are both kUse and one has a non-null permutation
  // instruction, returns kUse with that permutation.  OTOH if both have
  // different, non-null permutation instructions, returns kReuse.
  //
  // You can think of this sort of as a conjunction, whereas Fold is sort of a
  // disjunction.  FoldUseMandatory() says "no use" if either input isn't used,
  // whereas Fold() would say "use".
  static UseKind FoldUseMandatory(UseKind a, UseKind b) {
    if (a.kind == UseKind::kNoUse || b.kind == UseKind::kNoUse) {
      return UseKind::kNoUse;
    }
    if (a.kind == UseKind::kReuse || b.kind == UseKind::kReuse) {
      return UseKind::kReuse;
    }
    if (a.permutation_instr == b.permutation_instr) {
      return a;  // They're the same.
    }
    if (b.permutation_instr == nullptr) {
      return a;
    }
    if (a.permutation_instr == nullptr) {
      return b;
    }
    return UseKind::kReuse;
  }
};

HloInstruction::UseKind HloInstruction::OperandElementUse(
    int64 operand_num) const {
  switch (opcode_) {
    case HloOpcode::kBitcast:
      // A bitcast that only adds or removes degenerate (i.e. size 1) dimensions
      // doesn't permute its elements, so it counts as a plain, non-permuting
      // use.
      return ShapeUtil::DropDegenerateDimensions(shape()) ==
                     ShapeUtil::DropDegenerateDimensions(operand(0)->shape())
                 ? UseKind::kUse
                 : UseKind::Permuting(this);
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return UseKind::Permuting(this);
    case HloOpcode::kPad:
      // Pad reuses the padding value but not the padded array elements.
      return operand_num > 0 ? UseKind::kReuse : UseKind::Permuting(this);
    case HloOpcode::kReduce:
      // Reduce reuses the init values but not the operand array elements.
      return operand_num >= Cast<HloReduceInstruction>(this)->input_count()
                 ? UseKind::kReuse
                 : UseKind::Permuting(this);
    case HloOpcode::kFusion:
      // Uses the memoizing, recursive computation defined above.
      return FusionReusesParamElements::Compute(operand_num,
                                                *fused_expression_root());
    case HloOpcode::kDot:
      // Matrix-vector dots do not reuse the matrix operand.
      if (shape().dimensions_size() <= 1) {
        if ((operand_num == 0 && operand(1)->shape().rank() <= 1) ||
            (operand_num == 1 && operand(0)->shape().rank() <= 1)) {
          return UseKind::kUse;
        }
      }
      return UseKind::kReuse;
    case HloOpcode::kDynamicUpdateSlice:
      // Dynamic-update-slice reuses only start_indices.
      if (operand_num == 0 || operand_num == 1) {
        return UseKind::kUse;
      }
      return UseKind::kReuse;
    case HloOpcode::kGather:
      // Gather reads its indices in a linear fashion, and it permutes the
      // vector it's gathering from.
      return operand_num == 0 ? UseKind::kUse : UseKind::Permuting(this);
    default:
      return IsElementwise() ? UseKind::kUse : UseKind::kReuse;
  }
}

std::tuple<bool, std::vector<int64>, std::vector<int64>>
HloInstruction::ReshapeMerelyInsertsOrDeletes1SizedDimensions() const {
  if (HloOpcode::kReshape != opcode_) {
    return std::make_tuple(false, std::vector<int64>(), std::vector<int64>());
  }
  return ShapeUtil::InsertedOrDeleted1SizedDimensions(operand(0)->shape_,
                                                      shape_);
}

string ToString(HloInstruction::FusionKind kind) {
  switch (kind) {
    case HloInstruction::FusionKind::kLoop:
      return "kLoop";
    case HloInstruction::FusionKind::kInput:
      return "kInput";
    case HloInstruction::FusionKind::kOutput:
      return "kOutput";
    case HloInstruction::FusionKind::kCustom:
      return "kCustom";
  }
}

StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    const string& kind_name) {
  if (kind_name == "kLoop") {
    return HloInstruction::FusionKind::kLoop;
  }
  if (kind_name == "kInput") {
    return HloInstruction::FusionKind::kInput;
  }
  if (kind_name == "kOutput") {
    return HloInstruction::FusionKind::kOutput;
  }
  if (kind_name == "kCustom") {
    return HloInstruction::FusionKind::kCustom;
  }
  return InvalidArgument("Unknown fusion kind: %s", kind_name);
}

string FrontendAttributesToString(
    const FrontendAttributes& frontend_attributes) {
  std::vector<std::pair<string, string>> sorted_attributes(
      frontend_attributes.map().begin(), frontend_attributes.map().end());
  absl::c_sort(sorted_attributes);
  return absl::StrFormat(
      "{%s}", absl::StrJoin(sorted_attributes, ",", absl::PairFormatter("=")));
}

string PaddingConfigToString(const PaddingConfig& padding) {
  bool has_interior_padding =
      absl::c_any_of(padding.dimensions(),
                     [](const PaddingConfig::PaddingConfigDimension& dim) {
                       return dim.interior_padding() != 0;
                     });
  return StrJoin(
      padding.dimensions(), "x",
      [&](string* out, const PaddingConfig::PaddingConfigDimension& dim) {
        StrAppend(
            out, dim.edge_padding_low(), "_", dim.edge_padding_high(),
            has_interior_padding ? StrCat("_", dim.interior_padding()) : "");
      });
}

string OpMetadataToString(const OpMetadata& metadata) {
  std::vector<string> result;
  if (!metadata.op_type().empty()) {
    result.push_back(StrCat("op_type=\"", CEscape(metadata.op_type()), "\""));
  }
  if (!metadata.op_name().empty()) {
    result.push_back(StrCat("op_name=\"", CEscape(metadata.op_name()), "\""));
  }
  if (!metadata.source_file().empty()) {
    result.push_back(
        StrCat("source_file=\"", CEscape(metadata.source_file()), "\""));
  }
  if (metadata.source_line() != 0) {
    result.push_back(StrCat("source_line=", metadata.source_line()));
  }
  return StrJoin(result, " ");
}

string RandomDistributionToString(const RandomDistribution& distribution) {
  return absl::AsciiStrToLower(RandomDistribution_Name(distribution));
}

string PrecisionToString(const PrecisionConfig::Precision& precision) {
  return absl::AsciiStrToLower(PrecisionConfig::Precision_Name(precision));
}

string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dnums) {
  // lhs_dims[i] is the symbol of the logical dimension i for the lhs
  // operand. E.g. if batch has dimension number 2, then lhs_dims[2] == "b".
  std::vector<string> lhs_dims(2 + dnums.input_spatial_dimensions().size());
  lhs_dims[dnums.input_batch_dimension()] = 'b';
  lhs_dims[dnums.input_feature_dimension()] = 'f';
  for (int64 i = 0; i < dnums.input_spatial_dimensions().size(); ++i) {
    lhs_dims[dnums.input_spatial_dimensions(i)] = StrCat(i);
  }

  std::vector<string> rhs_dims(2 + dnums.kernel_spatial_dimensions().size());
  rhs_dims[dnums.kernel_input_feature_dimension()] = "i";
  rhs_dims[dnums.kernel_output_feature_dimension()] = "o";
  for (int64 i = 0; i < dnums.kernel_spatial_dimensions().size(); ++i) {
    rhs_dims[dnums.kernel_spatial_dimensions(i)] = StrCat(i);
  }

  std::vector<string> output_dims(2 + dnums.output_spatial_dimensions().size());
  output_dims[dnums.output_batch_dimension()] = 'b';
  output_dims[dnums.output_feature_dimension()] = 'f';
  for (int64 i = 0; i < dnums.output_spatial_dimensions().size(); ++i) {
    output_dims[dnums.output_spatial_dimensions(i)] = StrCat(i);
  }

  return StrCat(StrJoin(lhs_dims, ""), "_", StrJoin(rhs_dims, ""), "->",
                StrJoin(output_dims, ""));
}

string ReplicaGroupsToString(const std::vector<ReplicaGroup>& replica_groups) {
  std::vector<string> replica_group_str;
  replica_group_str.reserve(replica_groups.size());
  for (const ReplicaGroup& group : replica_groups) {
    replica_group_str.push_back(
        StrCat("{", StrJoin(group.replica_ids(), ","), "}"));
  }
  return StrCat("{", StrJoin(replica_group_str, ","), "}");
}

StatusOr<RandomDistribution> StringToRandomDistribution(const string& name) {
  static std::unordered_map<string, RandomDistribution>* map = [] {
    static auto* map = new std::unordered_map<string, RandomDistribution>;
    for (int i = 0; i < RandomDistribution_ARRAYSIZE; i++) {
      if (RandomDistribution_IsValid(i)) {
        auto value = static_cast<RandomDistribution>(i);
        (*map)[RandomDistributionToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown distribution");
  }
  return found->second;
}

StatusOr<PrecisionConfig::Precision> StringToPrecision(const string& name) {
  static std::unordered_map<string, PrecisionConfig::Precision>* map = [] {
    static auto* map =
        new std::unordered_map<string, PrecisionConfig::Precision>;
    for (int i = 0; i < PrecisionConfig::Precision_ARRAYSIZE; i++) {
      if (PrecisionConfig::Precision_IsValid(i)) {
        auto value = static_cast<PrecisionConfig::Precision>(i);
        (*map)[PrecisionToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return InvalidArgument("Unknown distribution");
  }
  return found->second;
}

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind) {
  return os << ToString(kind);
}

bool HloPtrComparator::operator()(const HloInstruction* const& lhs,
                                  const HloInstruction* const& rhs) const {
  if (rhs == nullptr) {
    // Nothing compares less than nullptr.
    return false;
  }
  if (lhs == nullptr) {
    return true;
  }
  auto lhs_module = lhs->GetModule();
  auto rhs_module = rhs->GetModule();
  CHECK((lhs_module == nullptr && rhs_module == nullptr) ||
        (lhs_module != nullptr && rhs_module != nullptr));
  if (lhs_module != nullptr &&
      lhs_module->unique_id() != rhs_module->unique_id()) {
    return lhs_module->unique_id() < rhs_module->unique_id();
  }
  return lhs->unique_id() < rhs->unique_id();
}

bool HloInstruction::CouldBeBitcast() const {
  switch (opcode_) {
    case HloOpcode::kTranspose:
      return true;
    case HloOpcode::kReshape:
      return std::get<0>(ReshapeMerelyInsertsOrDeletes1SizedDimensions());
    default:
      return false;
  }
}

Status HloInstruction::GetBackendConfigInternal(
    tensorflow::protobuf::Message* proto) const {
  proto->Clear();

  // Empty string does not parse as valid JSON, but it's a valid backend config,
  // corresponding to the empty proto.
  if (backend_config_.empty()) {
    return Status::OK();
  }
  return tensorflow::HumanReadableJsonToProto(backend_config_, proto);
}

Status HloInstruction::set_backend_config(
    const tensorflow::protobuf::Message& proto) {
  TF_ASSIGN_OR_RETURN(backend_config_, BackendConfigToRawString(proto));
  return Status::OK();
}

/* static */ StatusOr<string> HloInstruction::BackendConfigToRawString(
    const tensorflow::protobuf::Message& proto) {
  string ret;
  // Pass ignore_accuracy_loss = true because estimated_cycles field can be
  // INT64_MAX. If ignore_accuracy_loss = false and estimated_cycles =
  // INT64_MAX, JsonFormat will return an error status, although there is no
  // accuracy loss for int64.
  TF_RETURN_IF_ERROR(tensorflow::ProtoToHumanReadableJson(
      proto, &ret, /*ignore_accuracy_loss=*/true));
  return ret;
}

const PrecisionConfig& HloInstruction::precision_config() const {
  if (auto* convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->precision_config();
  }
  if (auto* dot = DynCast<HloDotInstruction>(this)) {
    return dot->precision_config();
  }
  LOG(FATAL) << "Unimplemented method.";
}

PrecisionConfig* HloInstruction::mutable_precision_config() {
  if (auto* convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->mutable_precision_config();
  }
  if (auto* dot = DynCast<HloDotInstruction>(this)) {
    return dot->mutable_precision_config();
  }
  LOG(FATAL) << "Unimplemented method.";
}

HloModule* HloInstruction::GetModule() const {
  if (parent_) {
    return parent_->parent();
  }
  return nullptr;
}

void HloInstruction::UniquifyName(NameUniquer* name_uniquer) {
  string parent_str = parent() == nullptr ? "noparent" : parent()->name();
  name_ = name_uniquer->GetUniqueName(name_);
}

void HloInstruction::set_outer_dimension_partitions(
    const std::vector<int64>& outer_dimension_partitions) {
  outer_dimension_partitions_ = outer_dimension_partitions;
}

// TODO(b/80131774): Remove these temporary methods after transition.
int64 HloInstruction::feature_index() const {
  return Cast<HloBatchNormInstruction>(this)->feature_index();
}

float HloInstruction::epsilon() const {
  return Cast<HloBatchNormInstruction>(this)->epsilon();
}

FftType HloInstruction::fft_type() const {
  return Cast<HloFftInstruction>(this)->fft_type();
}

const std::vector<int64>& HloInstruction::fft_length() const {
  return Cast<HloFftInstruction>(this)->fft_length();
}

int64 HloInstruction::concatenate_dimension() const {
  return Cast<HloConcatenateInstruction>(this)->concatenate_dimension();
}

int64 HloInstruction::dimension() const {
  if (auto set_size = DynCast<HloSetDimensionSizeInstruction>(this)) {
    return set_size->dimension();
  }
  return Cast<HloGetDimensionSizeInstruction>(this)->dimension();
}

int64 HloInstruction::inferred_dimension() const {
  return Cast<HloReshapeInstruction>(this)->inferred_dimension();
}

bool HloInstruction::IsRank2Transpose() const {
  auto transpose = DynCast<HloTransposeInstruction>(this);
  return transpose != nullptr && transpose->IsRank2Transpose();
}

int64 HloInstruction::slice_starts(int64 dimension) const {
  return Cast<HloSliceInstruction>(this)->slice_starts(dimension);
}

const std::vector<int64>& HloInstruction::slice_starts() const {
  return Cast<HloSliceInstruction>(this)->slice_starts();
}

int64 HloInstruction::slice_limits(int64 dimension) const {
  return Cast<HloSliceInstruction>(this)->slice_limits(dimension);
}

const std::vector<int64>& HloInstruction::slice_limits() const {
  return Cast<HloSliceInstruction>(this)->slice_limits();
}

int64 HloInstruction::slice_strides(int64 dimension) const {
  return Cast<HloSliceInstruction>(this)->slice_strides(dimension);
}

const std::vector<int64>& HloInstruction::slice_strides() const {
  return Cast<HloSliceInstruction>(this)->slice_strides();
}

const Literal& HloInstruction::literal() const {
  return Cast<HloConstantInstruction>(this)->literal();
}

bool HloInstruction::IsConstant() const {
  return DynCast<HloConstantInstruction>(this) != nullptr;
}

void HloInstruction::RelayoutConstant(const Layout& new_layout,
                                      const ShapeIndex& shape_index) {
  Cast<HloConstantInstruction>(this)->RelayoutConstant(new_layout, shape_index);
}

string HloInstruction::TracingTag() const {
  return Cast<HloTraceInstruction>(this)->TracingTag();
}

HloInstruction* HloInstruction::AddFusionOperand(HloInstruction* new_operand) {
  return Cast<HloFusionInstruction>(this)->AddFusionOperand(new_operand);
}

// Delegates to HloFusionInstruction::MergeFusionInstruction.
void HloInstruction::MergeFusionInstruction(
    HloInstruction* instruction_to_merge) {
  return Cast<HloFusionInstruction>(this)->MergeFusionInstruction(
      Cast<HloFusionInstruction>(instruction_to_merge));
}

// Delegates to HloFusionInstruction::MergeFusionInstructionIntoMultiOutput.
void HloInstruction::MergeFusionInstructionIntoMultiOutput(
    HloInstruction* instruction_to_merge) {
  return Cast<HloFusionInstruction>(this)
      ->MergeFusionInstructionIntoMultiOutput(
          Cast<HloFusionInstruction>(instruction_to_merge));
}

HloInstruction* HloInstruction::FuseInstruction(
    HloInstruction* instruction_to_fuse) {
  return Cast<HloFusionInstruction>(this)->FuseInstruction(instruction_to_fuse);
}

HloInstruction* HloInstruction::FuseInstructionIntoMultiOutput(
    HloInstruction* instruction_to_fuse) {
  return Cast<HloFusionInstruction>(this)->FuseInstructionIntoMultiOutput(
      instruction_to_fuse);
}

HloComputation* HloInstruction::fused_instructions_computation() const {
  return Cast<HloFusionInstruction>(this)->fused_instructions_computation();
}

HloInstruction* HloInstruction::fused_expression_root() const {
  return Cast<HloFusionInstruction>(this)->fused_expression_root();
}

const tensorflow::gtl::iterator_range<UnwrappingIterator<
    std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
HloInstruction::fused_instructions() const {
  return Cast<HloFusionInstruction>(this)->fused_instructions();
}

const tensorflow::gtl::iterator_range<
    UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
HloInstruction::fused_instructions() {
  return Cast<HloFusionInstruction>(this)->fused_instructions();
}

int64 HloInstruction::fused_instruction_count() const {
  return Cast<HloFusionInstruction>(this)->fused_instruction_count();
}

HloInstruction* HloInstruction::fused_parameter(int64 parameter_number) const {
  return Cast<HloFusionInstruction>(this)->fused_parameter(parameter_number);
}

const std::vector<HloInstruction*>& HloInstruction::fused_parameters() const {
  return Cast<HloFusionInstruction>(this)->fused_parameters();
}

const bool HloInstruction::IsMultiOutputFusion() const {
  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(this);
  return fusion != nullptr && fusion->IsMultiOutputFusion();
}

HloInstruction::FusionKind HloInstruction::fusion_kind() const {
  return Cast<HloFusionInstruction>(this)->fusion_kind();
}

void HloInstruction::set_fusion_kind(FusionKind kind) {
  return Cast<HloFusionInstruction>(this)->set_fusion_kind(kind);
}

RandomDistribution HloInstruction::random_distribution() const {
  return Cast<HloRngInstruction>(this)->random_distribution();
}

int64 HloInstruction::parameter_number() const {
  return Cast<HloParameterInstruction>(this)->parameter_number();
}

void HloInstruction::set_parameter_replicated_at_leaf_buffers(
    absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
  return Cast<HloParameterInstruction>(this)
      ->set_parameter_replicated_at_leaf_buffers(
          parameter_replicated_at_leaf_buffers);
}

void HloInstruction::set_parameter_replicated_at_leaf_buffers(
    const std::vector<bool>& parameter_replicated_at_leaf_buffers) {
  return Cast<HloParameterInstruction>(this)
      ->set_parameter_replicated_at_leaf_buffers(
          parameter_replicated_at_leaf_buffers);
}

const absl::optional<std::vector<bool>>&
HloInstruction::parameter_replicated_at_leaf_buffers() const {
  return Cast<HloParameterInstruction>(this)
      ->parameter_replicated_at_leaf_buffers();
}

int64 HloInstruction::tuple_index() const {
  return Cast<HloGetTupleElementInstruction>(this)->tuple_index();
}

void HloInstruction::set_tuple_index(int64 new_tuple_index) {
  return Cast<HloGetTupleElementInstruction>(this)->set_tuple_index(
      new_tuple_index);
}

int32 HloInstruction::exponent_bits() const {
  return Cast<HloReducePrecisionInstruction>(this)->exponent_bits();
}

int32 HloInstruction::mantissa_bits() const {
  return Cast<HloReducePrecisionInstruction>(this)->mantissa_bits();
}

string HloInstruction::infeed_config() const {
  return Cast<HloInfeedInstruction>(this)->infeed_config();
}

void HloInstruction::set_infeed_config(const string& config) {
  return Cast<HloInfeedInstruction>(this)->set_infeed_config(config);
}

const Shape& HloInstruction::outfeed_shape() const {
  return Cast<HloOutfeedInstruction>(this)->outfeed_shape();
}

const string& HloInstruction::outfeed_config() const {
  return Cast<HloOutfeedInstruction>(this)->outfeed_config();
}

const std::vector<ReplicaGroup>& HloInstruction::replica_groups() const {
  return Cast<HloCollectiveInstruction>(this)->replica_groups();
}

const std::vector<std::pair<int64, int64>>&
HloInstruction::source_target_pairs() const {
  return Cast<HloCollectivePermuteInstruction>(this)->source_target_pairs();
}

absl::optional<int64> HloInstruction::channel_id() const {
  return Cast<HloChannelInstruction>(this)->channel_id();
}

void HloInstruction::set_channel_id(const absl::optional<int64>& channel_id) {
  return Cast<HloChannelInstruction>(this)->set_channel_id(channel_id);
}

const ConvolutionDimensionNumbers&
HloInstruction::convolution_dimension_numbers() const {
  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->convolution_dimension_numbers();
  }
  if (auto custom_call = DynCast<HloCustomCallInstruction>(this)) {
    return custom_call->convolution_dimension_numbers();
  }
  LOG(FATAL) << "Unimplemented method.";
}

void HloInstruction::set_convolution_dimension_numbers(
    const ConvolutionDimensionNumbers& dnums) {
  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    convolution->set_convolution_dimension_numbers(dnums);
  } else if (auto custom_call = DynCast<HloCustomCallInstruction>(this)) {
    custom_call->set_convolution_dimension_numbers(dnums);
  } else {
    LOG(FATAL) << "Unimplemented method.";
  }
}

int64 HloInstruction::feature_group_count() const {
  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->feature_group_count();
  }
  return Cast<HloCustomCallInstruction>(this)->feature_group_count();
}

void HloInstruction::set_feature_group_count(int64 feature_group_count) {
  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->set_feature_group_count(feature_group_count);
  }
  Cast<HloCustomCallInstruction>(this)->set_feature_group_count(
      feature_group_count);
}

int64 HloInstruction::batch_group_count() const {
  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->batch_group_count();
  }
  return Cast<HloCustomCallInstruction>(this)->batch_group_count();
}

void HloInstruction::set_batch_group_count(int64 batch_group_count) {
  if (auto convolution = DynCast<HloConvolutionInstruction>(this)) {
    return convolution->set_batch_group_count(batch_group_count);
  }
  Cast<HloCustomCallInstruction>(this)->set_batch_group_count(
      batch_group_count);
}

HloComputation* HloInstruction::select() const {
  return Cast<HloSelectAndScatterInstruction>(this)->select();
}

HloComputation* HloInstruction::scatter() const {
  return Cast<HloSelectAndScatterInstruction>(this)->scatter();
}

void HloInstruction::set_select(HloComputation* computation) {
  return Cast<HloSelectAndScatterInstruction>(this)->set_select(computation);
}

void HloInstruction::set_scatter(HloComputation* computation) {
  return Cast<HloSelectAndScatterInstruction>(this)->set_scatter(computation);
}

const string& HloInstruction::custom_call_target() const {
  return Cast<HloCustomCallInstruction>(this)->custom_call_target();
}

const PaddingConfig& HloInstruction::padding_config() const {
  return Cast<HloPadInstruction>(this)->padding_config();
}

int64 HloInstruction::slice_sizes(int64 dimension) const {
  return Cast<HloDynamicSliceInstruction>(this)->slice_sizes(dimension);
}

const std::vector<int64>& HloInstruction::dynamic_slice_sizes() const {
  return Cast<HloDynamicSliceInstruction>(this)->dynamic_slice_sizes();
}

const GatherDimensionNumbers& HloInstruction::gather_dimension_numbers() const {
  return Cast<HloGatherInstruction>(this)->gather_dimension_numbers();
}

absl::Span<const int64> HloInstruction::gather_slice_sizes() const {
  return Cast<HloGatherInstruction>(this)->gather_slice_sizes();
}

const ScatterDimensionNumbers& HloInstruction::scatter_dimension_numbers()
    const {
  return Cast<HloScatterInstruction>(this)->scatter_dimension_numbers();
}

const DotDimensionNumbers& HloInstruction::dot_dimension_numbers() const {
  return Cast<HloDotInstruction>(this)->dot_dimension_numbers();
}

const DomainMetadata& HloInstruction::operand_side_metadata() const {
  return Cast<HloDomainInstruction>(this)->operand_side_metadata();
}

const DomainMetadata& HloInstruction::user_side_metadata() const {
  return Cast<HloDomainInstruction>(this)->user_side_metadata();
}

ComparisonDirection HloInstruction::comparison_direction() const {
  return Cast<HloCompareInstruction>(this)->direction();
}

const TriangularSolveOptions& HloInstruction::triangular_solve_options() const {
  return Cast<HloTriangularSolveInstruction>(this)->triangular_solve_options();
}

const CholeskyOptions& HloInstruction::cholesky_options() const {
  return Cast<HloCholeskyInstruction>(this)->cholesky_options();
}

}  // namespace xla
