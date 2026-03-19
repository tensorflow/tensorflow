/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/memory_annotations.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ConvertMemoryPlacementToInternalAnnotationsTest
    : public HloHardwareIndependentTestBase {
 public:
  ConvertMemoryPlacementToInternalAnnotationsTest() = default;
};

TEST_F(ConvertMemoryPlacementToInternalAnnotationsTest, ConvertPinnedHostTest) {
  const char* hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

region_0.9 {
  arg_tuple.10 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.11 = s32[] get-tuple-element(arg_tuple.10), index=0
  constant.15 = s32[] constant(1)
  add.33 = s32[] add(get-tuple-element.11, constant.15)
  get-tuple-element.12 = f32[16]{0} get-tuple-element(arg_tuple.10), index=1
  sine.18 = f32[16]{0} sine(get-tuple-element.12)
  sine.19 = f32[16]{0} sine(sine.18)
  sine.20 = f32[16]{0} sine(sine.19)
  get-tuple-element.13 = f32[16,16]{1,0} get-tuple-element(arg_tuple.10), index=2
  custom-call.21 = f32[16]{0} custom-call(sine.19), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="pinned_host"}
  reshape.23 = f32[1,16]{1,0} reshape(custom-call.21)
  constant.17 = s32[] constant(0)
  compare.24 = pred[] compare(get-tuple-element.11, constant.17), direction=LT
  constant.16 = s32[] constant(16)
  add.25 = s32[] add(get-tuple-element.11, constant.16)
  select.26 = s32[] select(compare.24, add.25, get-tuple-element.11)
  dynamic-update-slice.27 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.13, reshape.23, select.26, constant.17)
  get-tuple-element.14 = f32[16,16]{1,0} get-tuple-element(arg_tuple.10), index=3
  custom-call.22 = f32[16]{0} custom-call(sine.20), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="pinned_host"}
  reshape.28 = f32[1,16]{1,0} reshape(custom-call.22)
  compare.29 = pred[] compare(get-tuple-element.11, constant.17), direction=LT
  add.30 = s32[] add(get-tuple-element.11, constant.16)
  select.31 = s32[] select(compare.29, add.30, get-tuple-element.11)
  dynamic-update-slice.32 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.14, reshape.28, select.31, constant.17)
  ROOT tuple.34 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(add.33, sine.20, dynamic-update-slice.27, dynamic-update-slice.32)
}

region_1.35 {
  arg_tuple.36 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.38 = f32[16]{0} get-tuple-element(arg_tuple.36), index=1
  get-tuple-element.39 = f32[16,16]{1,0} get-tuple-element(arg_tuple.36), index=2
  get-tuple-element.40 = f32[16,16]{1,0} get-tuple-element(arg_tuple.36), index=3
  get-tuple-element.37 = s32[] get-tuple-element(arg_tuple.36), index=0
  constant.41 = s32[] constant(16)
  ROOT compare.42 = pred[] compare(get-tuple-element.37, constant.41), direction=LT
}

core_closed_call.43 {
  constant.47 = s32[] constant(0)
  Arg_0.44 = f32[16]{0} parameter(0)
  constant.45 = f32[] constant(0)
  broadcast.46 = f32[16,16]{1,0} broadcast(constant.45), dimensions={}
  tuple.48 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(constant.47, Arg_0.44, broadcast.46, broadcast.46)
  while.49 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) while(tuple.48), condition=region_1.35, body=region_0.9
  get-tuple-element.50 = s32[] get-tuple-element(while.49), index=0
  get-tuple-element.51 = f32[16]{0} get-tuple-element(while.49), index=1
  get-tuple-element.52 = f32[16,16]{1,0} get-tuple-element(while.49), index=2
  get-tuple-element.53 = f32[16,16]{1,0} get-tuple-element(while.49), index=3
  ROOT tuple.54 = (f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(get-tuple-element.52, get-tuple-element.53)
}

region_2.65 {
  arg_tuple.66 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.67 = s32[] get-tuple-element(arg_tuple.66), index=0
  constant.74 = s32[] constant(1)
  add.108 = s32[] add(get-tuple-element.67, constant.74)
  get-tuple-element.73 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=6
  constant.76 = s32[] constant(0)
  compare.82 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  constant.75 = s32[] constant(16)
  add.83 = s32[] add(get-tuple-element.67, constant.75)
  select.84 = s32[] select(compare.82, add.83, get-tuple-element.67)
  dynamic-slice.85 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.73, select.84, constant.76), dynamic_slice_sizes={1,16}
  reshape.86 = f32[16]{0} reshape(dynamic-slice.85)
  custom-call.87 = f32[16]{0} custom-call(reshape.86), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="device"}
  get-tuple-element.69 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=2
  get-tuple-element.68 = f32[16]{0} get-tuple-element(arg_tuple.66), index=1
  cosine.88 = f32[16]{0} cosine(get-tuple-element.68)
  reshape.93 = f32[1,16]{1,0} reshape(cosine.88)
  compare.94 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.95 = s32[] add(get-tuple-element.67, constant.75)
  select.96 = s32[] select(compare.94, add.95, get-tuple-element.67)
  dynamic-update-slice.97 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.69, reshape.93, select.96, constant.76)
  get-tuple-element.70 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=3
  sine.89 = f32[16]{0} sine(get-tuple-element.68)
  cosine.90 = f32[16]{0} cosine(sine.89)
  reshape.98 = f32[1,16]{1,0} reshape(cosine.90)
  compare.99 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.100 = s32[] add(get-tuple-element.67, constant.75)
  select.101 = s32[] select(compare.99, add.100, get-tuple-element.67)
  dynamic-update-slice.102 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.70, reshape.98, select.101, constant.76)
  get-tuple-element.71 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=4
  get-tuple-element.72 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=5
  compare.77 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.78 = s32[] add(get-tuple-element.67, constant.75)
  select.79 = s32[] select(compare.77, add.78, get-tuple-element.67)
  dynamic-slice.80 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.72, select.79, constant.76), dynamic_slice_sizes={1,16}
  reshape.81 = f32[16]{0} reshape(dynamic-slice.80)
  custom-call.91 = f32[16]{0} custom-call(reshape.81), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="device"}
  cosine.92 = f32[16]{0} cosine(custom-call.91)
  reshape.103 = f32[1,16]{1,0} reshape(cosine.92)
  compare.104 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.105 = s32[] add(get-tuple-element.67, constant.75)
  select.106 = s32[] select(compare.104, add.105, get-tuple-element.67)
  dynamic-update-slice.107 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.71, reshape.103, select.106, constant.76)
  ROOT tuple.109 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(add.108, custom-call.87, dynamic-update-slice.97, dynamic-update-slice.102, dynamic-update-slice.107, get-tuple-element.72, get-tuple-element.73)
}

region_3.110 {
  arg_tuple.111 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.113 = f32[16]{0} get-tuple-element(arg_tuple.111), index=1
  get-tuple-element.114 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=2
  get-tuple-element.115 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=3
  get-tuple-element.116 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=4
  get-tuple-element.117 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=5
  get-tuple-element.118 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=6
  get-tuple-element.112 = s32[] get-tuple-element(arg_tuple.111), index=0
  constant.119 = s32[] constant(16)
  ROOT compare.120 = pred[] compare(get-tuple-element.112, constant.119), direction=LT
}

region_4.130 {
  arg_tuple.131 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) parameter(0)
  get-tuple-element.132 = s32[] get-tuple-element(arg_tuple.131), index=0
  constant.140 = s32[] constant(1)
  add.164 = s32[] add(get-tuple-element.132, constant.140)
  get-tuple-element.133 = f32[16]{0} get-tuple-element(arg_tuple.131), index=1
  get-tuple-element.134 = f32[] get-tuple-element(arg_tuple.131), index=2
  broadcast.159 = f32[16]{0} broadcast(get-tuple-element.134), dimensions={}
  add.160 = f32[16]{0} add(get-tuple-element.133, broadcast.159)
  get-tuple-element.137 = f32[16,16]{1,0} get-tuple-element(arg_tuple.131), index=5
  constant.141 = s32[] constant(16)
  subtract.142 = s32[] subtract(constant.141, get-tuple-element.132)
  subtract.143 = s32[] subtract(subtract.142, constant.140)
  constant.139 = s32[] constant(0)
  compare.154 = pred[] compare(subtract.143, constant.139), direction=LT
  add.155 = s32[] add(subtract.143, constant.141)
  select.156 = s32[] select(compare.154, add.155, subtract.143)
  dynamic-slice.157 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.137, select.156, constant.139), dynamic_slice_sizes={1,16}
  reshape.158 = f32[16]{0} reshape(dynamic-slice.157)
  multiply.161 = f32[16]{0} multiply(add.160, reshape.158)
  get-tuple-element.136 = f32[16,16]{1,0} get-tuple-element(arg_tuple.131), index=4
  compare.149 = pred[] compare(subtract.143, constant.139), direction=LT
  add.150 = s32[] add(subtract.143, constant.141)
  select.151 = s32[] select(compare.149, add.150, subtract.143)
  dynamic-slice.152 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.136, select.151, constant.139), dynamic_slice_sizes={1,16}
  reshape.153 = f32[16]{0} reshape(dynamic-slice.152)
  multiply.162 = f32[16]{0} multiply(multiply.161, reshape.153)
  get-tuple-element.135 = f32[16,16]{1,0} get-tuple-element(arg_tuple.131), index=3
  compare.144 = pred[] compare(subtract.143, constant.139), direction=LT
  add.145 = s32[] add(subtract.143, constant.141)
  select.146 = s32[] select(compare.144, add.145, subtract.143)
  dynamic-slice.147 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.135, select.146, constant.139), dynamic_slice_sizes={1,16}
  reshape.148 = f32[16]{0} reshape(dynamic-slice.147)
  multiply.163 = f32[16]{0} multiply(multiply.162, reshape.148)
  constant.138 = f32[] constant(0)
  ROOT tuple.165 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) tuple(add.164, multiply.163, constant.138, get-tuple-element.135, get-tuple-element.136, get-tuple-element.137)
}

region_5.166 {
  arg_tuple.167 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) parameter(0)
  get-tuple-element.169 = f32[16]{0} get-tuple-element(arg_tuple.167), index=1
  get-tuple-element.170 = f32[] get-tuple-element(arg_tuple.167), index=2
  get-tuple-element.171 = f32[16,16]{1,0} get-tuple-element(arg_tuple.167), index=3
  get-tuple-element.172 = f32[16,16]{1,0} get-tuple-element(arg_tuple.167), index=4
  get-tuple-element.173 = f32[16,16]{1,0} get-tuple-element(arg_tuple.167), index=5
  get-tuple-element.168 = s32[] get-tuple-element(arg_tuple.167), index=0
  constant.174 = s32[] constant(16)
  ROOT compare.175 = pred[] compare(get-tuple-element.168, constant.174), direction=LT
}

ENTRY main.183 {
  constant.6 = s32[] constant(0)
  Arg_0.1 = f32[16]{0} parameter(0), sharding={devices=[2]<=[2]}
  call.55 = (f32[16,16]{1,0}, f32[16,16]{1,0}) call(Arg_0.1), to_apply=core_closed_call.43
  get-tuple-element.56 = f32[16,16]{1,0} get-tuple-element(call.55), index=0
  get-tuple-element.57 = f32[16,16]{1,0} get-tuple-element(call.55), index=1
  constant.7 = f32[] constant(1)
  tuple.58 = (f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16]{0}, f32[]) tuple(get-tuple-element.56, get-tuple-element.57, Arg_0.1, constant.7)
  opt-barrier.59 = (f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16]{0}, f32[]) opt-barrier(tuple.58)
  get-tuple-element.62 = f32[16]{0} get-tuple-element(opt-barrier.59), index=2
  constant.4 = f32[] constant(0)
  broadcast.5 = f32[16,16]{1,0} broadcast(constant.4), dimensions={}
  get-tuple-element.60 = f32[16,16]{1,0} get-tuple-element(opt-barrier.59), index=0
  get-tuple-element.61 = f32[16,16]{1,0} get-tuple-element(opt-barrier.59), index=1
  tuple.64 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(constant.6, get-tuple-element.62, broadcast.5, broadcast.5, broadcast.5, get-tuple-element.60, get-tuple-element.61)
  while.121 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) while(tuple.64), condition=region_3.110, body=region_2.65
  get-tuple-element.122 = s32[] get-tuple-element(while.121), index=0
  get-tuple-element.123 = f32[16]{0} get-tuple-element(while.121), index=1
  get-tuple-element.127 = f32[16,16]{1,0} get-tuple-element(while.121), index=5
  get-tuple-element.128 = f32[16,16]{1,0} get-tuple-element(while.121), index=6
  constant.2 = f32[] constant(0)
  broadcast.3 = f32[16]{0} broadcast(constant.2), dimensions={}
  get-tuple-element.63 = f32[] get-tuple-element(opt-barrier.59), index=3
  get-tuple-element.124 = f32[16,16]{1,0} get-tuple-element(while.121), index=2
  get-tuple-element.125 = f32[16,16]{1,0} get-tuple-element(while.121), index=3
  get-tuple-element.126 = f32[16,16]{1,0} get-tuple-element(while.121), index=4
  tuple.129 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) tuple(constant.6, broadcast.3, get-tuple-element.63, get-tuple-element.124, get-tuple-element.125, get-tuple-element.126)
  while.176 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) while(tuple.129), condition=region_5.166, body=region_4.130
  get-tuple-element.177 = s32[] get-tuple-element(while.176), index=0
  ROOT get-tuple-element.178 = f32[16]{0} get-tuple-element(while.176), index=1
  get-tuple-element.179 = f32[] get-tuple-element(while.176), index=2
  get-tuple-element.180 = f32[16,16]{1,0} get-tuple-element(while.176), index=3
  get-tuple-element.181 = f32[16,16]{1,0} get-tuple-element(while.176), index=4
  get-tuple-element.182 = f32[16,16]{1,0} get-tuple-element(while.176), index=5
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  bool changed =
      ConvertMemoryPlacementToInternalAnnotations().Run(module.get()).value();
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  int64_t custom_calls_count = 0;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      if (instr->IsCustomCall(
              memory_annotations::kMoveToHostCustomCallTarget) ||
          instr->IsCustomCall(
              memory_annotations::kMoveToDeviceCustomCallTarget)) {
        ++custom_calls_count;
      }
    }
  }
  EXPECT_EQ(custom_calls_count, 4);
}

TEST_F(ConvertMemoryPlacementToInternalAnnotationsTest,
       ConvertUnpinnedHostTest) {
  const char* hlo_string = R"(
HloModule jit_f, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

region_0.9 {
  arg_tuple.10 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.11 = s32[] get-tuple-element(arg_tuple.10), index=0
  constant.15 = s32[] constant(1)
  add.33 = s32[] add(get-tuple-element.11, constant.15)
  get-tuple-element.12 = f32[16]{0} get-tuple-element(arg_tuple.10), index=1
  sine.18 = f32[16]{0} sine(get-tuple-element.12)
  sine.19 = f32[16]{0} sine(sine.18)
  sine.20 = f32[16]{0} sine(sine.19)
  get-tuple-element.13 = f32[16,16]{1,0} get-tuple-element(arg_tuple.10), index=2
  custom-call.21 = f32[16]{0} custom-call(sine.19), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="unpinned_host"}
  reshape.23 = f32[1,16]{1,0} reshape(custom-call.21)
  constant.17 = s32[] constant(0)
  compare.24 = pred[] compare(get-tuple-element.11, constant.17), direction=LT
  constant.16 = s32[] constant(16)
  add.25 = s32[] add(get-tuple-element.11, constant.16)
  select.26 = s32[] select(compare.24, add.25, get-tuple-element.11)
  dynamic-update-slice.27 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.13, reshape.23, select.26, constant.17)
  get-tuple-element.14 = f32[16,16]{1,0} get-tuple-element(arg_tuple.10), index=3
  custom-call.22 = f32[16]{0} custom-call(sine.20), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="unpinned_host"}
  reshape.28 = f32[1,16]{1,0} reshape(custom-call.22)
  compare.29 = pred[] compare(get-tuple-element.11, constant.17), direction=LT
  add.30 = s32[] add(get-tuple-element.11, constant.16)
  select.31 = s32[] select(compare.29, add.30, get-tuple-element.11)
  dynamic-update-slice.32 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.14, reshape.28, select.31, constant.17)
  ROOT tuple.34 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(add.33, sine.20, dynamic-update-slice.27, dynamic-update-slice.32)
}

region_1.35 {
  arg_tuple.36 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.38 = f32[16]{0} get-tuple-element(arg_tuple.36), index=1
  get-tuple-element.39 = f32[16,16]{1,0} get-tuple-element(arg_tuple.36), index=2
  get-tuple-element.40 = f32[16,16]{1,0} get-tuple-element(arg_tuple.36), index=3
  get-tuple-element.37 = s32[] get-tuple-element(arg_tuple.36), index=0
  constant.41 = s32[] constant(16)
  ROOT compare.42 = pred[] compare(get-tuple-element.37, constant.41), direction=LT
}

core_closed_call.43 {
  constant.47 = s32[] constant(0)
  Arg_0.44 = f32[16]{0} parameter(0)
  constant.45 = f32[] constant(0)
  broadcast.46 = f32[16,16]{1,0} broadcast(constant.45), dimensions={}
  tuple.48 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(constant.47, Arg_0.44, broadcast.46, broadcast.46)
  while.49 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}) while(tuple.48), condition=region_1.35, body=region_0.9
  get-tuple-element.50 = s32[] get-tuple-element(while.49), index=0
  get-tuple-element.51 = f32[16]{0} get-tuple-element(while.49), index=1
  get-tuple-element.52 = f32[16,16]{1,0} get-tuple-element(while.49), index=2
  get-tuple-element.53 = f32[16,16]{1,0} get-tuple-element(while.49), index=3
  ROOT tuple.54 = (f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(get-tuple-element.52, get-tuple-element.53)
}

region_2.65 {
  arg_tuple.66 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.67 = s32[] get-tuple-element(arg_tuple.66), index=0
  constant.74 = s32[] constant(1)
  add.108 = s32[] add(get-tuple-element.67, constant.74)
  get-tuple-element.73 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=6
  constant.76 = s32[] constant(0)
  compare.82 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  constant.75 = s32[] constant(16)
  add.83 = s32[] add(get-tuple-element.67, constant.75)
  select.84 = s32[] select(compare.82, add.83, get-tuple-element.67)
  dynamic-slice.85 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.73, select.84, constant.76), dynamic_slice_sizes={1,16}
  reshape.86 = f32[16]{0} reshape(dynamic-slice.85)
  custom-call.87 = f32[16]{0} custom-call(reshape.86), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="device"}
  get-tuple-element.69 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=2
  get-tuple-element.68 = f32[16]{0} get-tuple-element(arg_tuple.66), index=1
  cosine.88 = f32[16]{0} cosine(get-tuple-element.68)
  reshape.93 = f32[1,16]{1,0} reshape(cosine.88)
  compare.94 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.95 = s32[] add(get-tuple-element.67, constant.75)
  select.96 = s32[] select(compare.94, add.95, get-tuple-element.67)
  dynamic-update-slice.97 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.69, reshape.93, select.96, constant.76)
  get-tuple-element.70 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=3
  sine.89 = f32[16]{0} sine(get-tuple-element.68)
  cosine.90 = f32[16]{0} cosine(sine.89)
  reshape.98 = f32[1,16]{1,0} reshape(cosine.90)
  compare.99 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.100 = s32[] add(get-tuple-element.67, constant.75)
  select.101 = s32[] select(compare.99, add.100, get-tuple-element.67)
  dynamic-update-slice.102 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.70, reshape.98, select.101, constant.76)
  get-tuple-element.71 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=4
  get-tuple-element.72 = f32[16,16]{1,0} get-tuple-element(arg_tuple.66), index=5
  compare.77 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.78 = s32[] add(get-tuple-element.67, constant.75)
  select.79 = s32[] select(compare.77, add.78, get-tuple-element.67)
  dynamic-slice.80 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.72, select.79, constant.76), dynamic_slice_sizes={1,16}
  reshape.81 = f32[16]{0} reshape(dynamic-slice.80)
  custom-call.91 = f32[16]{0} custom-call(reshape.81), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="device"}
  cosine.92 = f32[16]{0} cosine(custom-call.91)
  reshape.103 = f32[1,16]{1,0} reshape(cosine.92)
  compare.104 = pred[] compare(get-tuple-element.67, constant.76), direction=LT
  add.105 = s32[] add(get-tuple-element.67, constant.75)
  select.106 = s32[] select(compare.104, add.105, get-tuple-element.67)
  dynamic-update-slice.107 = f32[16,16]{1,0} dynamic-update-slice(get-tuple-element.71, reshape.103, select.106, constant.76)
  ROOT tuple.109 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(add.108, custom-call.87, dynamic-update-slice.97, dynamic-update-slice.102, dynamic-update-slice.107, get-tuple-element.72, get-tuple-element.73)
}

region_3.110 {
  arg_tuple.111 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) parameter(0)
  get-tuple-element.113 = f32[16]{0} get-tuple-element(arg_tuple.111), index=1
  get-tuple-element.114 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=2
  get-tuple-element.115 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=3
  get-tuple-element.116 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=4
  get-tuple-element.117 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=5
  get-tuple-element.118 = f32[16,16]{1,0} get-tuple-element(arg_tuple.111), index=6
  get-tuple-element.112 = s32[] get-tuple-element(arg_tuple.111), index=0
  constant.119 = s32[] constant(16)
  ROOT compare.120 = pred[] compare(get-tuple-element.112, constant.119), direction=LT
}

region_4.130 {
  arg_tuple.131 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) parameter(0)
  get-tuple-element.132 = s32[] get-tuple-element(arg_tuple.131), index=0
  constant.140 = s32[] constant(1)
  add.164 = s32[] add(get-tuple-element.132, constant.140)
  get-tuple-element.133 = f32[16]{0} get-tuple-element(arg_tuple.131), index=1
  get-tuple-element.134 = f32[] get-tuple-element(arg_tuple.131), index=2
  broadcast.159 = f32[16]{0} broadcast(get-tuple-element.134), dimensions={}
  add.160 = f32[16]{0} add(get-tuple-element.133, broadcast.159)
  get-tuple-element.137 = f32[16,16]{1,0} get-tuple-element(arg_tuple.131), index=5
  constant.141 = s32[] constant(16)
  subtract.142 = s32[] subtract(constant.141, get-tuple-element.132)
  subtract.143 = s32[] subtract(subtract.142, constant.140)
  constant.139 = s32[] constant(0)
  compare.154 = pred[] compare(subtract.143, constant.139), direction=LT
  add.155 = s32[] add(subtract.143, constant.141)
  select.156 = s32[] select(compare.154, add.155, subtract.143)
  dynamic-slice.157 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.137, select.156, constant.139), dynamic_slice_sizes={1,16}
  reshape.158 = f32[16]{0} reshape(dynamic-slice.157)
  multiply.161 = f32[16]{0} multiply(add.160, reshape.158)
  get-tuple-element.136 = f32[16,16]{1,0} get-tuple-element(arg_tuple.131), index=4
  compare.149 = pred[] compare(subtract.143, constant.139), direction=LT
  add.150 = s32[] add(subtract.143, constant.141)
  select.151 = s32[] select(compare.149, add.150, subtract.143)
  dynamic-slice.152 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.136, select.151, constant.139), dynamic_slice_sizes={1,16}
  reshape.153 = f32[16]{0} reshape(dynamic-slice.152)
  multiply.162 = f32[16]{0} multiply(multiply.161, reshape.153)
  get-tuple-element.135 = f32[16,16]{1,0} get-tuple-element(arg_tuple.131), index=3
  compare.144 = pred[] compare(subtract.143, constant.139), direction=LT
  add.145 = s32[] add(subtract.143, constant.141)
  select.146 = s32[] select(compare.144, add.145, subtract.143)
  dynamic-slice.147 = f32[1,16]{1,0} dynamic-slice(get-tuple-element.135, select.146, constant.139), dynamic_slice_sizes={1,16}
  reshape.148 = f32[16]{0} reshape(dynamic-slice.147)
  multiply.163 = f32[16]{0} multiply(multiply.162, reshape.148)
  constant.138 = f32[] constant(0)
  ROOT tuple.165 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) tuple(add.164, multiply.163, constant.138, get-tuple-element.135, get-tuple-element.136, get-tuple-element.137)
}

region_5.166 {
  arg_tuple.167 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) parameter(0)
  get-tuple-element.169 = f32[16]{0} get-tuple-element(arg_tuple.167), index=1
  get-tuple-element.170 = f32[] get-tuple-element(arg_tuple.167), index=2
  get-tuple-element.171 = f32[16,16]{1,0} get-tuple-element(arg_tuple.167), index=3
  get-tuple-element.172 = f32[16,16]{1,0} get-tuple-element(arg_tuple.167), index=4
  get-tuple-element.173 = f32[16,16]{1,0} get-tuple-element(arg_tuple.167), index=5
  get-tuple-element.168 = s32[] get-tuple-element(arg_tuple.167), index=0
  constant.174 = s32[] constant(16)
  ROOT compare.175 = pred[] compare(get-tuple-element.168, constant.174), direction=LT
}

ENTRY main.183 {
  constant.6 = s32[] constant(0)
  Arg_0.1 = f32[16]{0} parameter(0), sharding={devices=[2]<=[2]}
  call.55 = (f32[16,16]{1,0}, f32[16,16]{1,0}) call(Arg_0.1), to_apply=core_closed_call.43
  get-tuple-element.56 = f32[16,16]{1,0} get-tuple-element(call.55), index=0
  get-tuple-element.57 = f32[16,16]{1,0} get-tuple-element(call.55), index=1
  constant.7 = f32[] constant(1)
  tuple.58 = (f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16]{0}, f32[]) tuple(get-tuple-element.56, get-tuple-element.57, Arg_0.1, constant.7)
  opt-barrier.59 = (f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16]{0}, f32[]) opt-barrier(tuple.58)
  get-tuple-element.62 = f32[16]{0} get-tuple-element(opt-barrier.59), index=2
  constant.4 = f32[] constant(0)
  broadcast.5 = f32[16,16]{1,0} broadcast(constant.4), dimensions={}
  get-tuple-element.60 = f32[16,16]{1,0} get-tuple-element(opt-barrier.59), index=0
  get-tuple-element.61 = f32[16,16]{1,0} get-tuple-element(opt-barrier.59), index=1
  tuple.64 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) tuple(constant.6, get-tuple-element.62, broadcast.5, broadcast.5, broadcast.5, get-tuple-element.60, get-tuple-element.61)
  while.121 = (s32[], f32[16]{0}, f32[16,16]{1,0}, f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}, f32[16,16]{1,0}) while(tuple.64), condition=region_3.110, body=region_2.65
  get-tuple-element.122 = s32[] get-tuple-element(while.121), index=0
  get-tuple-element.123 = f32[16]{0} get-tuple-element(while.121), index=1
  get-tuple-element.127 = f32[16,16]{1,0} get-tuple-element(while.121), index=5
  get-tuple-element.128 = f32[16,16]{1,0} get-tuple-element(while.121), index=6
  constant.2 = f32[] constant(0)
  broadcast.3 = f32[16]{0} broadcast(constant.2), dimensions={}
  get-tuple-element.63 = f32[] get-tuple-element(opt-barrier.59), index=3
  get-tuple-element.124 = f32[16,16]{1,0} get-tuple-element(while.121), index=2
  get-tuple-element.125 = f32[16,16]{1,0} get-tuple-element(while.121), index=3
  get-tuple-element.126 = f32[16,16]{1,0} get-tuple-element(while.121), index=4
  tuple.129 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) tuple(constant.6, broadcast.3, get-tuple-element.63, get-tuple-element.124, get-tuple-element.125, get-tuple-element.126)
  while.176 = (s32[], f32[16]{0}, f32[], f32[16,16]{1,0}, f32[16,16]{1,0}, /*index=5*/f32[16,16]{1,0}) while(tuple.129), condition=region_5.166, body=region_4.130
  get-tuple-element.177 = s32[] get-tuple-element(while.176), index=0
  ROOT get-tuple-element.178 = f32[16]{0} get-tuple-element(while.176), index=1
  get-tuple-element.179 = f32[] get-tuple-element(while.176), index=2
  get-tuple-element.180 = f32[16,16]{1,0} get-tuple-element(while.176), index=3
  get-tuple-element.181 = f32[16,16]{1,0} get-tuple-element(while.176), index=4
  get-tuple-element.182 = f32[16,16]{1,0} get-tuple-element(while.176), index=5
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  bool changed =
      ConvertMemoryPlacementToInternalAnnotations().Run(module.get()).value();
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  int64_t custom_calls_count = 0;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      if (instr->IsCustomCall(
              memory_annotations::kMoveToHostCustomCallTarget) ||
          instr->IsCustomCall(
              memory_annotations::kMoveToDeviceCustomCallTarget)) {
        ++custom_calls_count;
      }
    }
  }
  EXPECT_EQ(custom_calls_count, 4);
}

TEST_F(ConvertMemoryPlacementToInternalAnnotationsTest,
       ConvertOutputPinnedHostTest) {
  constexpr absl::string_view hlo_string = R"(
  HloModule m, entry_computation_layout={(f32[2,2]{1,0:T(2,128)},f32[2,2]{1,0:T(2,128)})->f32[2,2]{1,0:T(2,128)S(5)}}
  ENTRY m {
    x = f32[2,2] parameter(0)
    y = f32[2,2] parameter(1)
    crs = f32[2,2] add(x, y)
    ROOT transfer = f32[2,2] custom-call(crs), custom_call_target="annotate_device_placement", frontend_attributes={_xla_buffer_placement="pinned_host"}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  bool changed =
      ConvertMemoryPlacementToInternalAnnotations().Run(module.get()).value();
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  int64_t move_to_host_count = 0;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      move_to_host_count +=
          instr->IsCustomCall(memory_annotations::kMoveToHostCustomCallTarget);
    }
  }
  EXPECT_EQ(move_to_host_count, 1);
}

TEST_F(ConvertMemoryPlacementToInternalAnnotationsTest,
       ConvertPinToDeviceSramTest) {
  constexpr absl::string_view hlo_string = R"(
  HloModule jit_f, entry_computation_layout={(s32[8,2]{0,1:T(2,128)S(1)})->s32[8,2]{0,1:T(2,128)}}, allow_spmd_sharding_propagation_to_output={true}

  ENTRY main.8 {
    Arg_0.1 = s32[8,2]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}, metadata={op_name="x"}
    constant.2 = s32[] constant(2)
    broadcast.3 = s32[8,2]{1,0} broadcast(constant.2), dimensions={}
    multiply.4 = s32[8,2]{1,0} multiply(Arg_0.1, broadcast.3), metadata={op_name="jit(f)/jit(main)/mul" source_file="third_party/py/jax/tests/memories_test.py" source_line=707}
    custom-call.5 = s32[8,2]{1,0} custom-call(multiply.4), custom_call_target="Sharding", sharding={devices=[2,1]<=[2]}, metadata={op_name="jit(f)/jit(main)/device_put" source_file="third_party/py/jax/tests/memories_test.py" source_line=708}
    custom-call.6 = s32[8,2]{1,0} custom-call(custom-call.5), custom_call_target="annotate_device_placement", custom_call_has_side_effect=true, frontend_attributes={_xla_buffer_placement="vmem"}, metadata={op_name="jit(f)/jit(main)/device_put" source_file="third_party/py/jax/tests/memories_test.py" source_line=708}
    ROOT multiply.7 = s32[8,2]{1,0} multiply(custom-call.6, broadcast.3), metadata={op_name="jit(f)/jit(main)/mul" source_file="third_party/py/jax/tests/memories_test.py" source_line=709}
  } // main.8 )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  bool changed =
      ConvertMemoryPlacementToInternalAnnotations().Run(module.get()).value();
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  int64_t pin_to_vmem_count = 0;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      pin_to_vmem_count += instr->IsCustomCall(
          memory_annotations::kPinToDeviceSramCustomCallTarget);
    }
  }
  EXPECT_EQ(pin_to_vmem_count, 1);
}

TEST_F(ConvertMemoryPlacementToInternalAnnotationsTest,
       ConvertPinToDeviceTest) {
  constexpr absl::string_view hlo_string = R"(
  HloModule jit_f, entry_computation_layout={(s32[8,2]{0,1:T(2,128)S(1)})->s32[8,2]{0,1:T(2,128)}}, allow_spmd_sharding_propagation_to_output={true}

  ENTRY main.8 {
    Arg_0.1 = s32[8,2]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}, metadata={op_name="x"}
    constant.2 = s32[] constant(2)
    broadcast.3 = s32[8,2]{1,0} broadcast(constant.2), dimensions={}
    multiply.4 = s32[8,2]{1,0} multiply(Arg_0.1, broadcast.3), metadata={op_name="jit(f)/jit(main)/mul" source_file="third_party/py/jax/tests/memories_test.py" source_line=707}
    custom-call.5 = s32[8,2]{1,0} custom-call(multiply.4), custom_call_target="Sharding", sharding={devices=[2,1]<=[2]}, metadata={op_name="jit(f)/jit(main)/device_put" source_file="third_party/py/jax/tests/memories_test.py" source_line=708}
    custom-call.6 = s32[8,2]{1,0} custom-call(custom-call.5), custom_call_target="annotate_device_placement", custom_call_has_side_effect=true, frontend_attributes={_xla_buffer_placement="pinned_device"}, metadata={op_name="jit(f)/jit(main)/device_put" source_file="third_party/py/jax/tests/memories_test.py" source_line=708}
    ROOT multiply.7 = s32[8,2]{1,0} multiply(custom-call.6, broadcast.3), metadata={op_name="jit(f)/jit(main)/mul" source_file="third_party/py/jax/tests/memories_test.py" source_line=709}
  } // main.8 )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  bool changed =
      ConvertMemoryPlacementToInternalAnnotations().Run(module.get()).value();
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  int64_t pin_todevice_count = 0;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      pin_todevice_count +=
          instr->IsCustomCall(memory_annotations::kPinToDeviceCustomCallTarget);
    }
  }
  EXPECT_EQ(pin_todevice_count, 1);
}

}  // namespace
}  // namespace xla
