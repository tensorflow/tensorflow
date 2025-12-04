/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/tools/hlo_module_loader.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class HloModuleLoaderTest : public HloHardwareIndependentTestBase {};

TEST_F(HloModuleLoaderTest, StripsLogHeaders) {
  const std::string& hlo_string = R"(
I0521 12:04:45.883483    1509 service.cc:186] HloModule test_log_stripping
I0521 12:04:45.883483    1509 service.cc:186]
I0521 12:04:45.883483    1509 service.cc:186] ENTRY entry {
I0521 12:04:45.883483    1509 service.cc:186]   p0 = f32[4]{0} parameter(0)
I0521 12:04:45.883483    1509 service.cc:186]   p1 = f32[4]{0} parameter(1)
I0521 12:04:45.883483    1509 service.cc:186]   add = f32[4]{0} add(p0, p1)
I0521 12:04:45.883483    1509 service.cc:186]   ROOT rooty = (f32[4]{0}, f32[4]{0}) tuple(p1, add)
I0521 12:04:45.883483    1509 service.cc:186] }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          LoadModuleFromData(hlo_string, "txt"));
  EXPECT_NE(FindInstruction(hlo_module.get(), "p0"), nullptr);
  EXPECT_NE(FindInstruction(hlo_module.get(), "p1"), nullptr);
  EXPECT_NE(FindInstruction(hlo_module.get(), "add"), nullptr);
  EXPECT_NE(FindInstruction(hlo_module.get(), "rooty"), nullptr);
}

TEST_F(HloModuleLoaderTest, SupportsStablehlo) {
  const std::string& stablehlo_string = R"(
module @jit_slice_data attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x5xi32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]> : tensor<2x5xi32>
    %0 = stablehlo.iota dim = 0 : tensor<5xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<5xi32>
    %2 = stablehlo.compare  LT, %0, %1,  SIGNED : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi1>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<5xi32>
    %4 = stablehlo.add %0, %3 : tensor<5xi32>
    %5 = stablehlo.select %2, %4, %0 : tensor<5xi1>, tensor<5xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<5xi32>) -> tensor<5x1xi32>
    %7 = "stablehlo.gather"(%c, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1>}> : (tensor<2x5xi32>, tensor<5x1xi32>) -> tensor<2x5xi32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<2x5xi32>
    %9 = stablehlo.add %7, %8 : tensor<2x5xi32>
    return %9 : tensor<2x5xi32>
  }
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          LoadModuleFromData(stablehlo_string, "stablehlo"));
  EXPECT_EQ(hlo_module->result_shape().ToString(), "s32[2,5]");
}

}  // namespace
}  // namespace xla
