/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/spmd/partition_assignment.h"

#include <memory>

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace {

using PartitionAssignmentTest = HloTestBase;

TEST_F(PartitionAssignmentTest, NoopAlg) {
  absl::string_view hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[16,16]{1,0} parameter(0)
  ROOT %copy = f32[16,16]{1,0} copy(%param0)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // The following redundantly sets the flag to the default value. We keep it
  // for the future tests to have the same style.
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_partitioning_algorithm(
      DebugOptions::PARTITIONING_ALGORITHM_NOOP);
  PartitionAssignment partition_assignment(/*num_partitions=*/16);
  EXPECT_EQ(partition_assignment.algorithm(), nullptr);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, partition_assignment.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_NE(partition_assignment.algorithm(), nullptr);
  EXPECT_EQ(partition_assignment.algorithm()->kind(),
            PartitioningAlgorithm::AlgorithmKind::kNoop);
}

}  // namespace
}  // namespace xla
