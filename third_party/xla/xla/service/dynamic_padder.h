/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_DYNAMIC_PADDER_H_
#define XLA_SERVICE_DYNAMIC_PADDER_H_

#include <functional>

#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// With bounded shapes, only part of the shape contains effective data and the
// rest contains padded data, whose value can be anything depending on the
// source of the data. When a bounded shape is directly consumed by an
// instruction that collapses dimensions (reduce for example), the padding data
// would affect result of the instruction.
//
// DynamicPadder uses DynamicDimensionInference to detect bounded shapes in a
// hlo module, it then inserts certain instructions to reset the padding into an
// identity value so that in doesn't affect the result of subsequent
// instruction. For example, it'd reset the padding to 0 before a bounded shape
// is consumed by a reduce-sum.
//
// Dynamic_padder removes dynamic shapes from the entry computation, and inserts
// custom calls (with dynamic shapes), which are lowered by specialized
// emitters: PadToStatic and SliceToDynamic.
//
// Note that it is not currently possible to send the output of PadToStatic
// across thread boundaries, and such shapes will be sent across the boundary in
// dynamic form. The DynamicPadder should be run separately for each thread that
// requires static shapes, and the dynamic shapes will be padded within the
// thread's computation.

struct DynamicPadderOptions {
  // Determines the form of dynamism supported by an HLO op.
  OpSupportsDynamismHandler op_supports_dynamism_handler = nullptr;

  // Instruct how to inference output dynamic dimensions of custom calls.
  DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler =
      nullptr;

  // If `slice_dynamic_output` is true, insert 'slice_to_dynamic' ops to all
  // outputs that are inferred to be dynamic.
  bool slice_dynamic_output = true;

  // Assertion generator for shape checks, only used if shape check mode is
  // "runtime".
  DynamicDimensionInference::AssertionGenerator assertion_generator;

  // If set to true, pessimisticly assumes runtime shape checks may fail and
  // returns a compile-time error.
  DynamicDimensionInference::ShapeCheckMode shape_check_mode =
      DynamicDimensionInference::ShapeCheckMode::kIgnore;
};

class DynamicPadder : public HloModulePass {
 public:
  explicit DynamicPadder(DynamicPadderOptions options = DynamicPadderOptions())
      : options_(options) {}

  absl::string_view name() const override { return "dynamic_padder"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  DynamicPadderOptions options_;
};

}  // namespace xla

#endif  // XLA_SERVICE_DYNAMIC_PADDER_H_
