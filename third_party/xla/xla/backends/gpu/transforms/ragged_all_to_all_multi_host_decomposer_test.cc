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

#include "xla/backends/gpu/transforms/ragged_all_to_all_multi_host_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/hlo_cse.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using RaggedAllToAllDecomposerTest = HloHardwareIndependentTestBase;

TEST_F(RaggedAllToAllDecomposerTest,
       SimpleRaggedAllToAllCrossReplicaIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module, replica_count=16

ENTRY main {
  input = bf16[128] parameter(0)
  output = bf16[256] parameter(1)
  input_offsets = s64[16] parameter(2)
  send_sizes = s64[16] parameter(3)
  output_offsets = s64[16] parameter(4)
  recv_sizes = s64[16] parameter(5)
  ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), 
    replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));

  EXPECT_TRUE(changed);
  EXPECT_OK(VerifyHloModule(module.get(), true, true));
  EXPECT_OK(HloDCE().Run(module.get()));
  EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: all-gather{{.*}}, replica_groups={{[{]}}{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}{{[}]}}
    // CHECK: all-to-all{{.*}}, replica_groups={{[{]}}{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}{{[}]}}
    // CHECK: ragged-all-to-all{{.*}}, replica_groups={{[{]}}{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}{{[}]}}
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest, DispatchRaggedAllToAllIsDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module, num_partitions=16

ENTRY main {
  input = bf16[128] parameter(0)
  output = bf16[256] parameter(1)
  input_offsets = s64[16] parameter(2)
  send_sizes = s64[16] parameter(3)
  output_offsets = s64[16] parameter(4)
  recv_sizes = s64[16] parameter(5)
  ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes), 
    replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));

  EXPECT_TRUE(changed);
  EXPECT_OK(VerifyHloModule(module.get(), true, true));
  EXPECT_OK(HloDCE().Run(module.get()));
  EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: all-gather{{.*}}, replica_groups={{[{]}}{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}{{[}]}}
    // CHECK: all-to-all{{.*}}, replica_groups={{[{]}}{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}{{[}]}}
    // CHECK: ragged-all-to-all{{.*}}, replica_groups={{[{]}}{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}{{[}]}}
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest,
       DispatchRaggedAllToAllWithShuffledReplicaGroupsIsDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module, num_partitions=16

ENTRY main {
  input = bf16[128] parameter(0)
  output = bf16[256] parameter(1)
  input_offsets = s64[32] parameter(2)
  send_sizes = s64[32] parameter(3)
  output_offsets = s64[32] parameter(4)
  recv_sizes = s64[32] parameter(5)
  ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes),
    replica_groups={{0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));

  EXPECT_TRUE(changed);
  EXPECT_OK(VerifyHloModule(module.get(), true, true));
  EXPECT_OK(HloDCE().Run(module.get()));
  EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: all-gather{{.*}}, replica_groups={{[{]}}{0,8},{4,12},{1,9},{5,13},{2,10},{6,14},{3,11},{7,15}{{[}]}}
    // CHECK-COUNT-4: s64[16,2]{1,0} gather
    // CHECK: all-to-all{{.*}}, replica_groups={{[{]}}{0,8},{4,12},{1,9},{5,13},{2,10},{6,14},{3,11},{7,15}{{[}]}}
    // CHECK: ragged-all-to-all{{.*}}, replica_groups={{[{]}}{0,4,1,5,2,6,3,7},{8,12,9,13,10,14,11,15}{{[}]}}
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest, SingleHostRaggedAllToAllIsNotDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
    input = bf16[128] parameter(0)
    output = bf16[256] parameter(1)
    input_offsets = s64[8] parameter(2)
    send_sizes = s64[8] parameter(3)
    output_offsets = s64[8] parameter(4)
    recv_sizes = s64[8] parameter(5)
    ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
      send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3,4,5,6,7}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedAllToAllDecomposerTest, CombineRaggedAllToAllIsDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module, replica_count=16

ENTRY main {
  input = bf16[4096,128] parameter(0)
  output = bf16[256,128] parameter(1)
  input_offsets = s64[16] parameter(2)
  send_sizes = s64[16] parameter(3)
  output_offsets = s64[16] parameter(4)
  recv_sizes = s64[16] parameter(5)
  ROOT ra2a = bf16[256,128] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes),
    replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));

  EXPECT_TRUE(changed);
  EXPECT_OK(VerifyHloModule(module.get(), true, true));
  EXPECT_OK(HloDCE().Run(module.get()));
  EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: ragged-all-to-all{{.*}}, replica_groups={{[{]}}{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}{{[}]}}
    // CHECK: all-to-all{{.*}}, replica_groups={{[{]}}{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}{{[}]}}
    // CHECK: all-to-all{{.*}}, replica_groups={{[{]}}{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}{{[}]}}
    // CHECK: ragged-all-to-all{{.*}}, replica_groups={{[{]}}{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}{{[}]}}
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest, MultipleReplicaGroupsAreSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
    input = bf16[128] parameter(0)
    output = bf16[256] parameter(1)
    input_offsets = s64[8] parameter(2)
    send_sizes = s64[8] parameter(3)
    output_offsets = s64[8] parameter(4)
    recv_sizes = s64[8] parameter(5)
    ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
      send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,2,4,6,8,10,12,14},{1,3,5,7,9,11,13,15}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_TRUE(changed);

  EXPECT_OK(VerifyHloModule(module.get(), true, true));
  EXPECT_OK(HloDCE().Run(module.get()));
  EXPECT_OK(HloCSE(true).Run(module.get()));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: all-gather{{.*}}, replica_groups={{[{]}}{0,8},{2,10},{4,12},{6,14},{1,9},{3,11},{5,13},{7,15}{{[}]}}
    // CHECK: all-to-all{{.*}}, replica_groups={{[{]}}{0,8},{2,10},{4,12},{6,14},{1,9},{3,11},{5,13},{7,15}{{[}]}}
    // CHECK: ragged-all-to-all{{.*}}, replica_groups={{[{]}}{0,2,4,6},{8,10,12,14},{1,3,5,7},{9,11,13,15}{{[}]}}
  )"));
}

TEST_F(RaggedAllToAllDecomposerTest, OnlyDecompositionForTwoHostsIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[128] parameter(0)
  output = bf16[256] parameter(1)
  input_offsets = s64[16] parameter(2)
  send_sizes = s64[16] parameter(3)
  output_offsets = s64[16] parameter(4)
  recv_sizes = s64[16] parameter(5)
  ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes),
    replica_groups=[1,16]<=[16]
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedAllToAllDecomposerTest, EmptyReplicaGroupsAreNotSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
  input = bf16[128] parameter(0)
  output = bf16[256] parameter(1)
  input_offsets = s64[16] parameter(2)
  send_sizes = s64[16] parameter(3)
  output_offsets = s64[16] parameter(4)
  recv_sizes = s64[16] parameter(5)
  ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
    send_sizes, output_offsets, recv_sizes),
    replica_groups={}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedAllToAllDecomposerTest,
       RaggedAllToAllWithinSingleHostIsNotDecomposed) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY main {
    input = bf16[128] parameter(0)
    output = bf16[256] parameter(1)
    input_offsets = s64[8] parameter(2)
    send_sizes = s64[8] parameter(3)
    output_offsets = s64[8] parameter(4)
    recv_sizes = s64[8] parameter(5)
    ROOT ra2a = bf16[256] ragged-all-to-all(input, output, input_offsets,
      send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}
}
)"));

  RaggedAllToAllMultiHostDecomposer decomposer(
      /*fast_interconnect_slice_size=*/8);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get(), {}));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
