/* Copyright 2017 The OpenXLA Authors.

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

// Class to create computations for testing rematerialization methods.

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_TEST_UTILS_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_TEST_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class RematerializationTestBase : public HloHardwareIndependentTestBase {
 protected:
  // Creates and returns a computation which can benefit from
  // rematerialization. The computation looks like:
  //
  //   F32[1] %param = {...}
  //   F32[] %reshape = reshape(F32[], param)
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %negate = negate(%bcast)
  //   F32[2048] %concat_1 = concat({%negate, %negate})
  //   F32[1] %slice_1 = slice(%concat_1, {0:1})
  //   F32[1025] %concat_2 = concat({%bcast, %slice_1})
  //   F32[1] %slice_2 = slice(%concat_2, {0:1});
  //
  // The instruction %bcast can be rematerialized before its use at %concat_2
  // to reduce peak memory usage. This avoids %bcast and %concat_1 being
  // simultaneously live. Peak memory use is about 16KB before rematerialization
  // (during execution of %concat_1) and about 12KB after rematerializing %bcast
  // for its use in %concat_2.
  std::unique_ptr<HloComputation> MakeRematerializableComputation(
      const std::string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, param));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, reshape, {}));
    auto negate = builder.AddInstruction(
        HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, bcast));
    auto concat_1 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {negate, negate},
        /*dimension=*/0));
    auto slice_1 = builder.AddInstruction(HloInstruction::CreateSlice(
        vec1_shape_, concat_1, /*start_indices=*/{0},
        /*limit_indices=*/{1},
        /*strides=*/{1}));
    auto concat_2 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, slice_1},
        /*dimension=*/0));
    // Add a final slice to make the parameter shape match the output shape
    // which is necessary to use this computation in a while.
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat_2,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1},
                                                       /*strides=*/{1}));
    return builder.Build();
  }

  // Creates and returns a computation which includes a while and can benefit
  // from rematerialization. The computation looks like:
  //
  //   F32[] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1] %slice_1 = slice(%bcast, {0:1})
  //   F32[1] %while = while(%slice_1, while_body, while_cond)
  //   F32[1025] %concat = concat({%bcast, %while})
  //   F32[1] %slice_2 = slice(%concat, {0:1});
  //
  // The instruction %bcast can be rematerialized before its use at %concat to
  // reduce peak memory usage. This avoids %bcast being live during execution of
  // the while. Peak memory use is maximum of 8K and 4K plus the memory use of
  // the while subcomputations.
  std::unique_ptr<HloComputation> MakeRematerializableWhileComputation(
      HloComputation* while_cond, HloComputation* while_body,
      const std::string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, param));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, reshape, {}));
    auto slice_1 = builder.AddInstruction(
        HloInstruction::CreateSlice(vec1_shape_, bcast, /*start_indices=*/{0},
                                    /*limit_indices=*/{1},
                                    /*strides=*/{1}));
    auto while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
        vec1_shape_, while_cond, while_body, slice_1));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, while_inst},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1},
                                                       /*strides=*/{1}));
    return builder.Build();
  }

  // Create and return a trivial computation appropriate for use as a while
  // condition.
  std::unique_ptr<HloComputation> MakeConditionComputation() {
    auto builder = HloComputation::Builder(TestName() + ".cond");
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
    return builder.Build();
  }

  // Return the byte size of the top-level buffer of the given shape.
  static int64_t ByteSizeOf(const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

  // Used to hold instructions and computation generated by
  // MakeInstrRemattedMultipleTimesComputation that is used to test
  // rematerialization of an instruction multiple times.
  struct InstrRemattedMultipleTimesGraph {
    HloInstruction* param;
    HloInstruction* bcast;
    HloInstruction* add_1;
    HloInstruction* add_2;
    HloInstruction* add_3;
    HloInstruction* add_4;
    HloComputation* entry_computation;
  };

  InstrRemattedMultipleTimesGraph MakeInstrRemattedMultipleTimesComputation(
      HloModule* module) {
    // Test that a single instruction is rematerialized several times. Module:
    //
    // Entry computation:
    //   F32[] %param = {...}
    //   F32[1024] %bcast = broadcast(%param)
    //   F32[1024] %add_1 = add(%bcast, bcast)
    //   F32[1024] %call_1 = call(Subcomputation, {%add_1})
    //   F32[1024] %add_2 = add(%bcast, call_1)
    //   F32[1024] %call_2 = call(SubComputation, {%add_2})
    //   F32[1024] %add_3 = add(%bcast, call_2)
    //   F32[1024] %call_3 = call(Subcomputation, {%add_3})
    //   F32[1024] %add_4 = add(%bcast, call_3)
    //
    // Subcomputation:
    //   F32[1024] %param = {...}
    //   F32[2048] %concat = concat({%param, %param})
    //   F32[1024] %slice = slice(%concat)
    //
    // The value %bcast is live across each call of Subcomputation (which
    // requires 8KB) though the value is not used in the calls. Rematerializing
    // %bcast across these calls reduces peak memory use from ~20KB down to
    // ~16KB.

    HloComputation* subcomputation = nullptr;
    {
      auto builder = HloComputation::Builder(TestName() + ".subcomputation");
      auto* param = builder.AddInstruction(
          HloInstruction::CreateParameter(0, vec1024_shape_, "param"));
      auto* concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeShape(xla::F32, {2048}), {param, param},
          /*dimension=*/0));
      builder.AddInstruction(HloInstruction::CreateSlice(
          vec1024_shape_, concat, /*start_indices=*/{0},
          /*limit_indices=*/{1024}, /*strides=*/{1}));
      subcomputation = module->AddEmbeddedComputation(builder.Build());
    }

    auto builder = HloComputation::Builder(TestName());
    auto* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape_, "param"));
    auto* bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
    auto* add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
        vec1024_shape_, HloOpcode::kAdd, bcast, bcast));
    auto* call_1 = builder.AddInstruction(
        HloInstruction::CreateCall(vec1024_shape_, {add_1}, subcomputation));
    auto* add_2 = builder.AddInstruction(HloInstruction::CreateBinary(
        vec1024_shape_, HloOpcode::kAdd, bcast, call_1));
    auto* call_2 = builder.AddInstruction(
        HloInstruction::CreateCall(vec1024_shape_, {add_2}, subcomputation));
    auto* add_3 = builder.AddInstruction(HloInstruction::CreateBinary(
        vec1024_shape_, HloOpcode::kAdd, bcast, call_2));
    auto* call_3 = builder.AddInstruction(
        HloInstruction::CreateCall(vec1024_shape_, {add_3}, subcomputation));
    auto* add_4 = builder.AddInstruction(HloInstruction::CreateBinary(
        vec1024_shape_, HloOpcode::kAdd, bcast, call_3));
    return {param,
            bcast,
            add_1,
            add_2,
            add_3,
            add_4,
            module->AddEntryComputation(builder.Build())};
  }

 protected:
  // Various shapes used in the canned computations.
  const Shape scalar_shape_ = ShapeUtil::MakeShape(xla::F32, {});
  const Shape vec1_shape_ = ShapeUtil::MakeShape(xla::F32, {1});
  const Shape vec1024_shape_ = ShapeUtil::MakeShape(xla::F32, {1024});
  AliasInfo alias_info_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_TEST_UTILS_H_
