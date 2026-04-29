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

#include "xla/backends/gpu/transforms/collectives/collective_pipelining_analyzer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using CollectivePipeliningAnalyzerTest = HloHardwareIndependentTestBase;

TEST_F(CollectivePipeliningAnalyzerTest,
       AllReduceLogicalReduceScatterIsTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[1024], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[1024], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[1024] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  all-reduce = bf16[4096] all-reduce(gte.1), replica_groups={{0,1,2,3},{4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=2
  table = s32[8]{0} constant({0,1,2,3,0,1,2,3})
  pid = u32[] partition-id()
  id = s32[1] dynamic-slice(table, pid), dynamic_slice_sizes={1}
  reshape = s32[] reshape(id)
  slice_size = s32[] constant(1024)
  offset = s32[] multiply(reshape, slice_size)
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  output-buffer = bf16[1,1024] broadcast(gte.0), dimensions={1}
  ds = bf16[1024] dynamic-slice(all-reduce, offset), dynamic_slice_sizes={1024}
  bc = bf16[1,1024] bitcast(ds)
  dus = bf16[1,1024] dynamic-update-slice(output-buffer, bc, next_ind, zero)
  rsb = bf16[1024] bitcast(dus)
  ROOT _ = (bf16[1024], bf16[4096], s32[]) tuple(rsb, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[1024] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[1024], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[1024], bf16[4096], s32[]) while(tuple), condition=condition, body=body
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  config.set_use_spmd_partitioning(true);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ar = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllReduce);

  ASSERT_NE(ar, nullptr);
  EXPECT_TRUE(IsTriviallyPipelineable(*ar));
}

TEST_F(CollectivePipeliningAnalyzerTest, AllReduceIsTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[4096], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[4096], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[4096] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  all-reduce = bf16[4096] all-reduce(gte.1), replica_groups={{0,1,2,3},{4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=2
  bc = bf16[1,4096] bitcast(all-reduce)
  output-buffer = bf16[1,4096] broadcast(gte.0), dimensions={1}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  dus = bf16[1,4096] dynamic-update-slice(output-buffer, bc, next_ind, zero)
  rsb = bf16[4096] bitcast(dus)
  ROOT _ = (bf16[4096], bf16[4096], s32[]) tuple(rsb, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[4096] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[4096], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[4096], bf16[4096], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ar = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllReduce);

  ASSERT_NE(ar, nullptr);
  EXPECT_TRUE(IsTriviallyPipelineable(*ar));
}

TEST_F(CollectivePipeliningAnalyzerTest,
       AllReduceWithExpensiveOpsToDUSIsNotTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[4096], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[4096], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[4096] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  all-reduce = bf16[4096] all-reduce(gte.1), replica_groups={{0,1,2,3},{4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=2
  elem = bf16[4096] add(all-reduce,gte.0)
  bc = bf16[1,4096] bitcast(elem)
  output-buffer = bf16[1,4096] broadcast(gte.0), dimensions={1}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  dus = bf16[1,4096] dynamic-update-slice(output-buffer, bc, next_ind, zero)
  rsb = bf16[4096] bitcast(dus)
  ROOT _ = (bf16[4096], bf16[4096], s32[]) tuple(rsb, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[4096] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[4096], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[4096], bf16[4096], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ar = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllReduce);

  ASSERT_NE(ar, nullptr);
  EXPECT_FALSE(IsTriviallyPipelineable(*ar));
}

TEST_F(CollectivePipeliningAnalyzerTest,
       AllReduceWithExpensiveOpsFromDUSIsNotTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[4096], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[4096], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[4096] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  all-reduce = bf16[4096] all-reduce(gte.1), replica_groups={{0,1,2,3},{4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=2
  bc = bf16[1,4096] bitcast(all-reduce)
  output-buffer = bf16[1,4096] broadcast(gte.0), dimensions={1}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  dus = bf16[1,4096] dynamic-update-slice(output-buffer, bc, next_ind, zero)
  rsb = bf16[4096] bitcast(dus)
  elem = bf16[4096] add(rsb, gte.0)
  ROOT _ = (bf16[4096], bf16[4096], s32[]) tuple(elem, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[4096] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[4096], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[4096], bf16[4096], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ar = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllReduce);

  ASSERT_NE(ar, nullptr);
  EXPECT_FALSE(IsTriviallyPipelineable(*ar));
}

TEST_F(CollectivePipeliningAnalyzerTest, ReduceScatterIsTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[512] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  reduce-scatter = bf16[512] reduce-scatter(gte.1), replica_groups={{0,1,2,3,4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=3,
    dimensions={0}
  bc = bf16[1,512] bitcast(reduce-scatter)
  output-buffer = bf16[1,512] broadcast(gte.0), dimensions={1}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  dus = bf16[1,512] dynamic-update-slice(output-buffer, bc, next_ind, zero)
  rsb = bf16[512] bitcast(dus)
  ROOT _ = (bf16[512], bf16[4096], s32[]) tuple(rsb, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[512] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[512], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[512], bf16[4096], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* rs = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kReduceScatter);

  ASSERT_NE(rs, nullptr);
  EXPECT_TRUE(IsTriviallyPipelineable(*rs));
}

TEST_F(CollectivePipeliningAnalyzerTest,
       ReduceScatterWithExpensiveOpsToDUSIsNotTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[512] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  reduce-scatter = bf16[512] reduce-scatter(gte.1), replica_groups={{0,1,2,3,4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=4,
    dimensions={0}
  elem = bf16[512] add(reduce-scatter, gte.0)
  bc = bf16[1,512] bitcast(elem)
  output-buffer = bf16[1,512] broadcast(gte.0), dimensions={1}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  dus = bf16[1,512] dynamic-update-slice(output-buffer, bc, next_ind, zero)
  rsb = bf16[512] bitcast(dus)
  ROOT _ = (bf16[512], bf16[4096], s32[]) tuple(rsb, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[512] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[512], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[512], bf16[4096], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* rs = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kReduceScatter);

  ASSERT_NE(rs, nullptr);
  EXPECT_FALSE(IsTriviallyPipelineable(*rs));
}

TEST_F(CollectivePipeliningAnalyzerTest,
       ReduceScatterWithExpensiveOpsFromDUSIsNotTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[512] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  reduce-scatter = bf16[512] reduce-scatter(gte.1), replica_groups={{0,1,2,3,4,5,6,7}},
    to_apply=sum,
    use_global_device_ids=true,
    channel_id=5,
    dimensions={0}
  bc = bf16[1,512] bitcast(reduce-scatter)
  output-buffer = bf16[1,512] broadcast(gte.0), dimensions={1}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  dus = bf16[1,512] dynamic-update-slice(output-buffer, bc, zero, zero)
  rsb = bf16[512] bitcast(dus)
  elem = bf16[512] add(rsb, gte.0)
  ROOT _ = (bf16[512], bf16[4096], s32[]) tuple(elem, gte.1, next_ind)
}

ENTRY main {
  p.0 = bf16[512] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[512], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[512], bf16[4096], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* rs = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kReduceScatter);

  ASSERT_NE(rs, nullptr);
  EXPECT_FALSE(IsTriviallyPipelineable(*rs));
}

TEST_F(CollectivePipeliningAnalyzerTest, AllGatherIsTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(8)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[512] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  ds = bf16[512] dynamic-slice(gte.1, next_ind), dynamic_slice_sizes={512}
  all-gather = bf16[4096] all-gather(ds), replica_groups={{0,1,2,3,4,5,6,7}},
    use_global_device_ids=true,
    channel_id=5,
    dimensions={0}
  ROOT _ = (bf16[512], bf16[4096], s32[]) tuple(gte.0, all-gather, next_ind)
}

ENTRY main {
  p.0 = bf16[512] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[512], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[512], bf16[4096], s32[]) while(tuple),
  condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"8"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ag = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllGather);

  ASSERT_NE(ag, nullptr);
  EXPECT_TRUE(IsTriviallyPipelineable(*ag));
}

TEST_F(CollectivePipeliningAnalyzerTest,
       AllGatherWithExpensiveOpsFromDSIsNotTriviallyPipelined) {
  // Expensive op between DS and AG input.
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(8)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[512] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  ds_operand = bf16[4096] add(gte.1, gte.1)
  ds = bf16[512] dynamic-slice(ds_operand, next_ind), dynamic_slice_sizes={512}
  all-gather = bf16[4096] all-gather(ds), replica_groups={{0,1,2,3,4,5,6,7}},
    use_global_device_ids=true,
    channel_id=7,
    dimensions={0}
  ROOT _ = (bf16[512], bf16[4096], s32[]) tuple(gte.0, all-gather, next_ind)
}

ENTRY main {
  p.0 = bf16[512] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[512], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[512], bf16[4096], s32[]) while(tuple),
  condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"8"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ag = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllGather);

  ASSERT_NE(ag, nullptr);
  EXPECT_FALSE(IsTriviallyPipelineable(*ag));
}

TEST_F(CollectivePipeliningAnalyzerTest,
       AllGatherWithExpensiveOpsToDSIsNotTriviallyPipelined) {
  absl::string_view hlo = R"(
HloModule m

sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT _ = bf16[] add(a, b)
}

condition {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=2
  trip_count = s32[] constant(8)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (bf16[512], bf16[4096], s32[]) parameter(0)
  gte.0 = bf16[512] get-tuple-element(input_tuple), index=0
  gte.1 = bf16[4096] get-tuple-element(input_tuple), index=1
  one = s32[] constant(1)
  ind = s32[] get-tuple-element(input_tuple), index=2
  next_ind = s32[] add(ind, one)
  ds = bf16[512] dynamic-slice(gte.1, next_ind), dynamic_slice_sizes={512}
  expensive_ag_operand = bf16[512] add(ds, ds)
  all-gather = bf16[4096] all-gather(expensive_ag_operand), replica_groups={{0,1,2,3,4,5,6,7}},
    use_global_device_ids=true,
    channel_id=6,
    dimensions={0}
  ROOT _ = (bf16[512], bf16[4096], s32[]) tuple(gte.0, all-gather, next_ind)
}

ENTRY main {
  p.0 = bf16[512] parameter(0)
  p.1 = bf16[4096] parameter(1)
  c.0 = s32[] constant(0)
  tuple = (bf16[512], bf16[4096], s32[]) tuple(p.0,p.1,c.0)
  ROOT while = (bf16[512], bf16[4096], s32[]) while(tuple),
  condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"8"},
                      "known_induction_variable":{"tuple_index":"2"}}
}
)";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(1);
  config.set_num_partitions(8);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  ASSERT_OK(CollectivePipeliningAnalyzer(/*pointer_size=*/4).Run(module.get()));

  HloInstruction* ag = hlo_query::FindInstruction(
      module->entry_computation()->root_instruction()->while_body(),
      HloOpcode::kAllGather);

  ASSERT_NE(ag, nullptr);
  EXPECT_FALSE(IsTriviallyPipelineable(*ag));
}

}  // namespace
}  // namespace xla::gpu
