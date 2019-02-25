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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/entry_visitor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include "absl/memory/memory.h"

#include <poplar/Device.hpp>
#include <poprand/RandomGen.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace xla {
namespace poplarplugin {
namespace {

class DeferredAllocationsVisitorTest : public HloTestBase {};

std::unique_ptr<CompilerResources> GetMockResources(HloModule* module) {
  auto resources = absl::make_unique<CompilerResources>(
      poplar::Device::createCPUDevice(), 0, poprand::NOT_REPEATABLE,
      poplar::OptionFlags(), poplar::OptionFlags(), false, module);
  poplin::addCodelets(resources->main_graph);
  popnn::addCodelets(resources->main_graph);
  popops::addCodelets(resources->main_graph);
  poprand::addCodelets(resources->main_graph);
  return std::move(resources);
}

HloPassPipeline GetMockPipeline(CompilerResources& resources) {
  HloPassPipeline pipeline("mock_pipeline");
  pipeline.AddPass<InplaceFinder>(resources.annotations);
  pipeline.AddPass<ShardingPass>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
  pipeline.AddPass<AllocationFinder>(resources.annotations);
  pipeline.AddPass<HloPassFix<ForwardAllocation>>(resources.annotations);
  pipeline.AddPass<HloMemoryScheduler>(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      DefaultMemoryScheduler);
  return pipeline;
}

TEST_F(DeferredAllocationsVisitorTest, TestDeferredAllocation) {
  const string& hlo_string = R"(

HloModule module
%_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  %arg_0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  %arg_1 = f32[2]{0} parameter(1)
  %broadcast.6.clone = f32[1,4,4,2]{3,2,1,0} broadcast(f32[2]{0} %arg_1), dimensions={3}
  ROOT %add.7.clone = f32[1,4,4,2]{3,2,1,0} add(f32[1,4,4,2]{3,2,1,0} %arg_0, f32[1,4,4,2]{3,2,1,0} %broadcast.6.clone)
}

ENTRY %cluster (arg0.1: (f32[1,4,4,2], f32[2], f32[1,1,2,2])) -> f32[1,4,4,2] {
  %arg = (f32[1,4,4,2], f32[2], f32[1,1,2,2]) parameter(0)
  %gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[2], f32[1,1,2,2]) %arg), index=0
  %gte2 = f32[1,1,2,2] get-tuple-element((f32[1,4,4,2], f32[2], f32[1,1,2,2]) %arg), index=2
  %convolution.5 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %gte0, f32[1,1,2,2]{3,2,1,0} %gte2), window={size=1x1}, dim_labels=b01f_01io->b01f
  %gte1 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2], f32[1,1,2,2]) %arg), index=1
  ROOT %fusion = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} %convolution.5, f32[2]{0} %gte1), kind=kCustom, calls=%_pop_op_conv_biasadd
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get());
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  ASSERT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EntryVisitor visitor(*resources.get(), false);
  auto entry_computation = module->entry_computation();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.at(entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1 = root->operand(1);
  auto arg = gte1->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.sequence, false)
          .ValueOrDie();
  poplar::Tensor gte1_tensor = FindInstructionOutputs(tensor_map, gte1)[0];
  poplar::Tensor arg_tensor = FindInstructionOutputs(tensor_map, arg)[1];
  CHECK_EQ(root_tensor, gte1_tensor);
  CHECK_EQ(gte1_tensor, arg_tensor);
}

TEST_F(DeferredAllocationsVisitorTest, TestDeferredAllocationNestedTuple) {
  const string& hlo_string = R"(

HloModule module
%_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  %arg_0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  %arg_1 = f32[2]{0} parameter(1)
  %broadcast.6.clone = f32[1,4,4,2]{3,2,1,0} broadcast(f32[2]{0} %arg_1), dimensions={3}
  ROOT %add.7.clone = f32[1,4,4,2]{3,2,1,0} add(f32[1,4,4,2]{3,2,1,0} %arg_0, f32[1,4,4,2]{3,2,1,0} %broadcast.6.clone)
}

ENTRY %cluster (arg0.1: ((f32[1,4,4,2], f32[2], f32[1,1,2,2]))) -> f32[1,4,4,2] {
  %arg = ((f32[1,4,4,2], f32[2], f32[1,1,2,2])) parameter(0)
  %gte = (f32[1,4,4,2], f32[2], f32[1,1,2,2]) get-tuple-element(((f32[1,4,4,2], f32[2], f32[1,1,2,2])) %arg), index=0
  %gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[2], f32[1,1,2,2]) %gte), index=0
  %gte2 = f32[1,1,2,2] get-tuple-element((f32[1,4,4,2], f32[2], f32[1,1,2,2]) %gte), index=2
  %convolution.5 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %gte0, f32[1,1,2,2]{3,2,1,0} %gte2), window={size=1x1}, dim_labels=b01f_01io->b01f
  %gte1 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2], f32[1,1,2,2]) %gte), index=1
  ROOT %fusion = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} %convolution.5, f32[2]{0} %gte1), kind=kCustom, calls=%_pop_op_conv_biasadd
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get());
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  ASSERT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EntryVisitor visitor(*resources.get(), false);
  auto entry_computation = module->entry_computation();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.at(entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1 = root->operand(1);
  auto gte = gte1->operand(0);
  auto arg = gte->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.sequence, false)
          .ValueOrDie();
  poplar::Tensor gte1_tensor = FindInstructionOutputs(tensor_map, gte1)[0];
  poplar::Tensor gte_tensor = FindInstructionOutputs(tensor_map, gte)[1];
  poplar::Tensor arg_tensor = FindInstructionOutputs(tensor_map, arg)[1];
  CHECK_EQ(root_tensor, gte1_tensor);
  CHECK_EQ(gte1_tensor, gte_tensor);
  CHECK_EQ(gte_tensor, arg_tensor);
}

TEST_F(DeferredAllocationsVisitorTest,
       TestDeferredAllocationNestedTupleInfeed) {
  const string& hlo_string = R"(

HloModule module
%_pop_op_conv_biasadd (arg_0: f32[1,4,4,2], arg_1: f32[2]) -> f32[1,4,4,2] {
  %arg_0 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  %arg_1 = f32[2]{0} parameter(1)
  %broadcast.6.clone = f32[1,4,4,2]{3,2,1,0} broadcast(f32[2]{0} %arg_1), dimensions={3}
  ROOT %add.7.clone = f32[1,4,4,2]{3,2,1,0} add(f32[1,4,4,2]{3,2,1,0} %arg_0, f32[1,4,4,2]{3,2,1,0} %broadcast.6.clone)
}

ENTRY %cluster (arg: f32[1,1,2,2]) -> f32[1,4,4,2] {
  %arg = f32[1,1,2,2] parameter(0)
  %after-all = token[] after-all()
  %infeed = ((f32[1,4,4,2], f32[2]), token[]) infeed(token[] %after-all), infeed_config="7"
  %gte = (f32[1,4,4,2], f32[2]) get-tuple-element(((f32[1,4,4,2], f32[2]), token[]) %infeed), index=0
  %gte0 = f32[1,4,4,2] get-tuple-element((f32[1,4,4,2], f32[2]) %gte), index=0
  %convolution.5 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %gte0, f32[1,1,2,2]{3,2,1,0} %arg), window={size=1x1}, dim_labels=b01f_01io->b01f
  %gte1 = f32[2] get-tuple-element((f32[1,4,4,2], f32[2]) %gte), index=1
  ROOT %fusion = f32[1,4,4,2]{3,2,1,0} fusion(f32[1,4,4,2]{3,2,1,0} %convolution.5, f32[2]{0} %gte1), kind=kCustom, calls=%_pop_op_conv_biasadd
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto resources = GetMockResources(module.get());
  HloPassPipeline pipeline = GetMockPipeline(*resources.get());
  ASSERT_TRUE(pipeline.Run(module.get()).ValueOrDie());
  EntryVisitor visitor(*resources.get(), false);
  auto entry_computation = module->entry_computation();
  TF_EXPECT_OK(entry_computation->Accept(&visitor));

  // Verify that gte1 has a tensor and all the deferred allocations have that
  // tensor too.
  auto tensor_map = resources->tensor_maps.at(entry_computation->name());
  auto root = entry_computation->root_instruction();
  auto gte1 = root->operand(1);
  auto gte = gte1->operand(0);
  auto infeed = gte->operand(0);
  poplar::Tensor root_tensor =
      FindInstructionInput(tensor_map, *resources.get(), root, 1,
                           visitor.sequence, false)
          .ValueOrDie();
  poplar::Tensor gte1_tensor = FindInstructionOutputs(tensor_map, gte1)[0];
  poplar::Tensor gte_tensor = FindInstructionOutputs(tensor_map, gte)[1];
  poplar::Tensor infeed_tensor = FindInstructionOutputs(tensor_map, infeed)[1];
  CHECK_EQ(root_tensor, gte1_tensor);
  CHECK_EQ(gte1_tensor, gte_tensor);
  CHECK_EQ(gte_tensor, infeed_tensor);
}

TEST_F(DeferredAllocationsVisitorTest, TestDeferredAllocationInsideLoops) {
  const string& hlo_string = R"(
HloModule module

%while_Sum-reduction.13 (x.14: f32[], y.15: f32[]) -> f32[] {
  %x.14 = f32[] parameter(0)
  %y.15 = f32[] parameter(1)
  ROOT %add.16 = f32[] add(f32[] %x.14, f32[] %y.15)
}

%_functionalize_body_1__.17 (arg_tuple.18: (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2])) -> (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) {
  %arg_tuple.18 = (s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) parameter(0)
  %get-tuple-element.21 = f32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=2
  %get-tuple-element.19 = s32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=0
  %constant.26 = s32[] constant(1)
  %add.27 = s32[] add(s32[] %get-tuple-element.19, s32[] %constant.26)
  %get-tuple-element.20 = s32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=1
  %constant.30 = s32[] constant(1)
  %add.31 = s32[] add(s32[] %get-tuple-element.20, s32[] %constant.30)
  %get-tuple-element.24 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=5
  %get-tuple-element.25 = f32[1,1,2,2]{3,2,1,0} get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=6
  %convolution.48 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %get-tuple-element.24, f32[1,1,2,2]{3,2,1,0} %get-tuple-element.25), window={size=1x1}, dim_labels=b01f_01io->b01f
  %get-tuple-element.23 = f32[2]{0} get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=4
  %constant.33 = f32[] constant(0.1)
  %broadcast.34 = f32[2]{0} broadcast(f32[] %constant.33), dimensions={}
  %constant.32 = f32[2]{0} constant({16, 16})
  %multiply.35 = f32[2]{0} multiply(f32[2]{0} %broadcast.34, f32[2]{0} %constant.32)
  %subtract.36 = f32[2]{0} subtract(f32[2]{0} %get-tuple-element.23, f32[2]{0} %multiply.35)
  %broadcast.49 = f32[1,4,4,2]{3,2,1,0} broadcast(f32[2]{0} %subtract.36), dimensions={3}
  %add.50 = f32[1,4,4,2]{3,2,1,0} add(f32[1,4,4,2]{3,2,1,0} %convolution.48, f32[1,4,4,2]{3,2,1,0} %broadcast.49)
  %convert.51 = f32[1,4,4,2]{3,2,1,0} convert(f32[1,4,4,2]{3,2,1,0} %add.50)
  %constant.52 = f32[] constant(0)
  %convert.53 = f32[] convert(f32[] %constant.52)
  %reduce.54 = f32[] reduce(f32[1,4,4,2]{3,2,1,0} %convert.51, f32[] %convert.53), dimensions={0,1,2,3}, to_apply=%while_Sum-reduction.13
  %convert.55 = f32[] convert(f32[] %reduce.54)
  %get-tuple-element.22 = s32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.18), index=3
  %tuple.56 = (f32[2]{0}) tuple(f32[2]{0} %subtract.36)
  %get-tuple-element.57 = f32[2]{0} get-tuple-element((f32[2]{0}) %tuple.56), index=0
  %constant.40 = f32[] constant(0.1)
  %broadcast.41 = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] %constant.40), dimensions={}
  %constant.28 = f32[] constant(1)
  %broadcast.29 = f32[1,4,4,2]{3,2,1,0} broadcast(f32[] %constant.28), dimensions={}
  %reverse.38 = f32[1,1,2,2]{3,2,1,0} reverse(f32[1,1,2,2]{3,2,1,0} %get-tuple-element.25), dimensions={0,1}
  %convolution.39 = f32[1,4,4,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %broadcast.29, f32[1,1,2,2]{3,2,1,0} %reverse.38), window={size=1x1}, dim_labels=b01f_01oi->b01f
  %multiply.42 = f32[1,4,4,2]{3,2,1,0} multiply(f32[1,4,4,2]{3,2,1,0} %broadcast.41, f32[1,4,4,2]{3,2,1,0} %convolution.39)
  %subtract.43 = f32[1,4,4,2]{3,2,1,0} subtract(f32[1,4,4,2]{3,2,1,0} %get-tuple-element.24, f32[1,4,4,2]{3,2,1,0} %multiply.42)
  %tuple.58 = (f32[1,4,4,2]{3,2,1,0}) tuple(f32[1,4,4,2]{3,2,1,0} %subtract.43)
  %get-tuple-element.59 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((f32[1,4,4,2]{3,2,1,0}) %tuple.58), index=0
  %constant.44 = f32[] constant(0.1)
  %broadcast.45 = f32[1,1,2,2]{3,2,1,0} broadcast(f32[] %constant.44), dimensions={}
  %convolution.37 = f32[1,1,2,2]{3,2,1,0} convolution(f32[1,4,4,2]{3,2,1,0} %get-tuple-element.24, f32[1,4,4,2]{3,2,1,0} %broadcast.29), window={size=4x4}, dim_labels=f01b_i01o->01bf
  %multiply.46 = f32[1,1,2,2]{3,2,1,0} multiply(f32[1,1,2,2]{3,2,1,0} %broadcast.45, f32[1,1,2,2]{3,2,1,0} %convolution.37)
  %subtract.47 = f32[1,1,2,2]{3,2,1,0} subtract(f32[1,1,2,2]{3,2,1,0} %get-tuple-element.25, f32[1,1,2,2]{3,2,1,0} %multiply.46)
  %tuple.60 = (f32[1,1,2,2]{3,2,1,0}) tuple(f32[1,1,2,2]{3,2,1,0} %subtract.47)
  %get-tuple-element.61 = f32[1,1,2,2]{3,2,1,0} get-tuple-element((f32[1,1,2,2]{3,2,1,0}) %tuple.60), index=0
  ROOT %tuple.62 = (s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) tuple(s32[] %add.27, s32[] %add.31, f32[] %convert.55, s32[] %get-tuple-element.22, f32[2]{0} %get-tuple-element.57, f32[1,4,4,2]{3,2,1,0} %get-tuple-element.59, f32[1,1,2,2]{3,2,1,0} %get-tuple-element.61)
}

%_functionalize_cond_1__.63 (arg_tuple.64: (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2])) -> (pred[]) {
  %arg_tuple.64 = (s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) parameter(0)
  %get-tuple-element.67 = f32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=2
  %get-tuple-element.69 = f32[2]{0} get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=4
  %get-tuple-element.70 = f32[1,4,4,2]{3,2,1,0} get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=5
  %get-tuple-element.71 = f32[1,1,2,2]{3,2,1,0} get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=6
  %get-tuple-element.65 = s32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=0
  %get-tuple-element.68 = s32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=3
  %less-than.74 = pred[] less-than(s32[] %get-tuple-element.65, s32[] %get-tuple-element.68)
  %get-tuple-element.66 = s32[] get-tuple-element((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %arg_tuple.64), index=1
  %constant.72 = s32[] constant(2)
  %less-than.73 = pred[] less-than(s32[] %get-tuple-element.66, s32[] %constant.72)
  %and.75 = pred[] and(pred[] %less-than.74, pred[] %less-than.73)
  ROOT %tuple.76 = (pred[]) tuple(pred[] %and.75)
}

%cond_wrapper.77 (inputs.78: (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2])) -> pred[] {
  %inputs.78 = (s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) parameter(0)
  %call.79 = (pred[]) call((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %inputs.78), to_apply=%_functionalize_cond_1__.63
  ROOT %get-tuple-element.80 = pred[] get-tuple-element((pred[]) %call.79), index=0
}

ENTRY %cluster_4790582643659166751_f15n_0__.98 (arg0.1: f32[1,4,4,2], arg1.2: f32[2], arg2.3: f32[1,1,2,2]) -> (s32[], s32[], f32[], s32[], f32[2], f32[1,4,4,2], f32[1,1,2,2]) {
  %constant.4 = s32[] constant(0)
  %constant.5 = s32[] constant(0)
  %constant.6 = f32[] constant(0)
  %constant.7 = s32[] constant(10)
  %constant.8 = s32[] constant(0)
  %constant.9 = s32[] constant(0)
  %constant.10 = f32[] constant(0)
  %constant.11 = s32[] constant(10)
  %arg1.2 = f32[2]{0} parameter(1)
  %arg0.1 = f32[1,4,4,2]{3,2,1,0} parameter(0)
  %arg2.3 = f32[1,1,2,2]{3,2,1,0} parameter(2)
  %tuple.12 = (s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) tuple(s32[] %constant.8, s32[] %constant.9, f32[] %constant.10, s32[] %constant.11, f32[2]{0} %arg1.2, f32[1,4,4,2]{3,2,1,0} %arg0.1, f32[1,1,2,2]{3,2,1,0} %arg2.3)
  ROOT %while.81 = (s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) while((s32[], s32[], f32[], s32[], f32[2]{0}, f32[1,4,4,2]{3,2,1,0}, f32[1,1,2,2]{3,2,1,0}) %tuple.12), condition=%cond_wrapper.77, body=%_functionalize_body_1__.17
}
)";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).ConsumeValueOrDie();
  auto layout_d4 = LayoutUtil::MakeLayout({3, 2, 1, 0});
  Array4D<float> input_arr({
      // clang-format off
    {  // i0=0
        {  // i1=0
            { 1,  2},  // i2=0
            { 3,  4},  // i2=1
            { 5,  6},  // i2=2
            { 7,  8},  // i2=3
        },
        {  // i1=1
            { 9, 10},  // i2=0
            {11, 12},  // i2=1
            {13, 14},  // i2=2
            {15, 16},  // i2=3
        },
        {  // i1=2
            {17, 18},  // i2=0
            {19, 20},  // i2=1
            {21, 22},  // i2=2
            {23, 24},  // i2=3
        },
        {  // i1=3
            {25, 26},  // i2=0
            {27, 28},  // i2=1
            {29, 30},  // i2=2
            {31, 32},  // i2=3
        },
    },
      // clang-format on
  });
  auto input_literal =
      LiteralUtil::CreateR4FromArray4DWithLayout<float>(input_arr, layout_d4);
  auto biases = LiteralUtil::CreateR1<float>({-100, 100});

  Array4D<float> weights_arr({
      // clang-format off
    {  // i0=0
        {  // i1=0
            {1, 2},  // i2=0
            {3, 4},  // i2=1
        },
    },
      // clang-format on
  });
  auto weights_literal =
      LiteralUtil::CreateR4FromArray4DWithLayout<float>(weights_arr, layout_d4);
  auto result = ExecuteAndTransfer(std::move(module),
                                   {&input_literal, &biases, &weights_literal});
  auto result_tuple = result.DecomposeTuple();
  // Expect correct value for the biases.
  EXPECT_EQ(result_tuple[4], LiteralUtil::CreateR1<float>({-103.2, 96.8}));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
