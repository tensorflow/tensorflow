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

#include "tensorflow/compiler/plugin/poplar/driver/entry_visitor.h"

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/forward_allocation.h"
#include "tensorflow/compiler/plugin/poplar/driver/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
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
  pipeline.AddPass<Scheduler>();
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

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
