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

// Example HLO graph which demonstrates Graphviz dumper for HLO
// computations. When run, pushes the example DOT graph to the Graphviz service
// and prints the URL. Useful for seeing effect of changes to the graph
// generation code.

#include <stdio.h>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// Adds a computation to the given HLO module which adds a scalar constant to
// its parameter and returns the result.
HloComputation* AddScalarConstantComputation(int64 addend, HloModule* module) {
  auto builder =
      HloComputation::Builder(tensorflow::strings::StrCat("add_", addend));
  auto x_value = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "x_value"));
  auto half = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.5)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      half->shape(), HloOpcode::kAdd, x_value, half));
  return module->AddEmbeddedComputation(builder.Build());
}

// Adds a computation to the given HLO module which sums its two parameters and
// returns the result.
HloComputation* ScalarSumComputation(HloModule* module) {
  auto builder = HloComputation::Builder("add");
  auto lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "lhs"));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "rhs"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
  return module->AddEmbeddedComputation(builder.Build());
}

// Adds a computation to the given HLO module which forwards its argument to a
// kCall instruction which then calls the given computation.
HloComputation* CallForwardingComputation(HloComputation* computation,
                                          HloModule* module) {
  auto builder = HloComputation::Builder("call_forward");
  auto arg = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "arg"));
  builder.AddInstruction(
      HloInstruction::CreateCall(arg->shape(), {arg}, computation));
  return module->AddEmbeddedComputation(builder.Build());
}

// Create a large, arbitrary computation with many different kinds of
// instructions. Sets the computation as the entry to an HLO module and returns
// the module.
std::unique_ptr<HloModule> MakeBigGraph() {
  HloModuleConfig config;
  auto module = MakeUnique<HloModule>("BigGraph", config);

  auto builder = HloComputation::Builder("TestBigGraphvizGraph");

  // Shapes used in the computation.
  auto mshape = ShapeUtil::MakeShape(F32, {3, 5});
  auto vshape = ShapeUtil::MakeShape(F32, {3});
  auto sshape = ShapeUtil::MakeShape(F32, {3});

  // Create a set of parameter instructions.
  auto param_v0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vshape, "foo"));
  auto param_v1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, vshape, "bar"));
  auto param_v2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, vshape, "baz"));
  auto param_s =
      builder.AddInstruction(HloInstruction::CreateParameter(3, sshape, "qux"));
  auto param_m =
      builder.AddInstruction(HloInstruction::CreateParameter(4, mshape, "zzz"));

  // Add an arbitrary expression of different instructions.
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(vshape, HloOpcode::kCopy, param_v0));
  auto clamp = builder.AddInstruction(HloInstruction::CreateTernary(
      vshape, HloOpcode::kClamp, copy, param_v1, param_v2));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(
      HloInstruction::CreateDot(vshape, clamp, param_v0, dot_dnums));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({dot, param_s, clamp}));
  auto scalar = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(sshape, tuple, 2));
  auto add_one = AddScalarConstantComputation(1.0, module.get());
  auto rng = builder.AddInstruction(
      HloInstruction::CreateRng(vshape, RNG_UNIFORM, {param_m, param_m}));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto add_computation = ScalarSumComputation(module.get());
  builder.AddInstruction(
      HloInstruction::CreateReduce(vshape, rng, one, {1}, add_computation));
  auto map1 = builder.AddInstruction(
      HloInstruction::CreateMap(sshape, {scalar}, add_one));
  auto map2 = builder.AddInstruction(
      HloInstruction::CreateMap(sshape, {map1}, add_one));
  auto map3 = builder.AddInstruction(
      HloInstruction::CreateMap(sshape, {map2}, add_one));

  // Create a fusion instruction containing the chain of map instructions.
  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      sshape, HloInstruction::FusionKind::kLoop, map3));
  fusion->FuseInstruction(map2);
  fusion->FuseInstruction(map1);

  // Add a random trace instruction.
  builder.AddInstruction(HloInstruction::CreateTrace("trace", dot));

  // Add a call instruction will calls the call-forwarding computation to call
  // another computation.
  auto call_computation = CallForwardingComputation(add_one, module.get());
  builder.AddInstruction(
      HloInstruction::CreateCall(fusion->shape(), {fusion}, call_computation));

  module->AddEntryComputation(builder.Build());
  return module;
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  auto module = xla::MakeBigGraph();

  printf("Graph URL: %s\n", xla::hlo_graph_dumper::DumpGraph(
                                *module->entry_computation(),
                                "Example computation", xla::DebugOptions())
                                .c_str());
  return 0;
}
