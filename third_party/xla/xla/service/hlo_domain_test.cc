/* Copyright 2018 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using HloDomainTest = HloHardwareIndependentTestBase;

// Tests that text dumps of domain instructions can be parsed back, in the
// specific case of null shardings.
TEST_F(HloDomainTest, DumpParseNullSharding) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {});
  auto sharding_md_0 = std::make_unique<ShardingMetadata>(nullptr);
  auto sharding_md_1 = std::make_unique<ShardingMetadata>(nullptr);
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p"));
  HloInstruction* domain = builder.AddInstruction(HloInstruction::CreateDomain(
      shape, param, std::move(sharding_md_0), std::move(sharding_md_1)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, domain, domain));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  auto hlo_string = module->ToString();
  ASSERT_TRUE(ParseAndReturnVerifiedModule(hlo_string).status().ok());
}

}  // namespace
}  // namespace xla
