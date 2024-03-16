/* Copyright 2023 The OpenXLA Authors.

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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "xla/literal.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace cpu {

class OneDnnSoftmaxTest : public HloTestBase {};

TEST_F(OneDnnSoftmaxTest, Softmaxtest) {
  const std::string hlo_string = R"(
        HloModule jit_softmax, entry_computation_layout={(f32[16,128,30522]{2,1,0})->f32[16,128,30522]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}
        region_0.4 {
            Arg_0.5 = f32[] parameter(0)
            Arg_1.6 = f32[] parameter(1)
            ROOT maximum.7 = f32[] maximum(Arg_0.5, Arg_1.6)
        }
        region_1.15 {
            Arg_0.16 = f32[] parameter(0)
            Arg_1.17 = f32[] parameter(1)
            ROOT add.18 = f32[] add(Arg_0.16, Arg_1.17)
        }
        ENTRY main.25 {
            Arg_0.1 = f32[16,128,30522]{2,1,0} parameter(0), sharding={replicated}
            constant.3 = f32[] constant(-inf)
            reduce.8 = f32[16,128]{1,0} reduce(Arg_0.1, constant.3), dimensions={2}, to_apply=region_0.4
            reshape.9 = f32[16,128,1]{2,1,0} reshape(reduce.8)
            broadcast.10 = f32[16,128,1]{2,1,0} broadcast(reshape.9), dimensions={0,1,2}
            reshape.11 = f32[16,128]{1,0} reshape(broadcast.10)
            broadcast.12 = f32[16,128,30522]{2,1,0} broadcast(reshape.11), dimensions={0,1}
            subtract.13 = f32[16,128,30522]{2,1,0} subtract(Arg_0.1, broadcast.12)
            exponential.14 = f32[16,128,30522]{2,1,0} exponential(subtract.13)
            constant.2 = f32[] constant(0)
            reduce.19 = f32[16,128]{1,0} reduce(exponential.14, constant.2), dimensions={2}, to_apply=region_1.15
            reshape.20 = f32[16,128,1]{2,1,0} reshape(reduce.19)
            broadcast.21 = f32[16,128,1]{2,1,0} broadcast(reshape.20), dimensions={0,1,2}
            reshape.22 = f32[16,128]{1,0} reshape(broadcast.21)
            broadcast.23 = f32[16,128,30522]{2,1,0} broadcast(reshape.22), dimensions={0,1}
            ROOT divide.24 = f32[16,128,30522]{2,1,0} divide(exponential.14, broadcast.23)
        }
    )";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(OneDnnSoftmaxTest, SoftmaxFP32) {
  const std::string hlo_string = R"(
        HloModule jit_softmax, entry_computation_layout={(f32[1,128,30522]{2,1,0})->f32[1,128,30522]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}
        region_0.4 {
            Arg_0.5 = f32[] parameter(0)
            Arg_1.6 = f32[] parameter(1)
            ROOT maximum.7 = f32[] maximum(Arg_0.5, Arg_1.6)
        }
        region_1.15 {
            Arg_0.16 = f32[] parameter(0)
            Arg_1.17 = f32[] parameter(1)
            ROOT add.18 = f32[] add(Arg_0.16, Arg_1.17)
        }
        ENTRY main.25 {
            Arg_0.1 = f32[1,128,30522]{2,1,0} parameter(0), sharding={replicated}
            constant.3 = f32[] constant(-inf)
            reduce.8 = f32[1,128]{1,0} reduce(Arg_0.1, constant.3), dimensions={2}, to_apply=region_0.4
            reshape.9 = f32[1,128,1]{2,1,0} reshape(reduce.8)
            broadcast.10 = f32[1,128,1]{2,1,0} broadcast(reshape.9), dimensions={0,1,2}
            reshape.11 = f32[1,128]{1,0} reshape(broadcast.10)
            broadcast.12 = f32[1,128,30522]{2,1,0} broadcast(reshape.11), dimensions={0,1}
            subtract.13 = f32[1,128,30522]{2,1,0} subtract(Arg_0.1, broadcast.12)
            exponential.14 = f32[1,128,30522]{2,1,0} exponential(subtract.13)
            constant.2 = f32[] constant(0)
            reduce.19 = f32[1,128]{1,0} reduce(exponential.14, constant.2), dimensions={2}, to_apply=region_1.15
            reshape.20 = f32[1,128,1]{2,1,0} reshape(reduce.19)
            broadcast.21 = f32[1,128,1]{2,1,0} broadcast(reshape.20), dimensions={0,1,2}
            reshape.22 = f32[1,128]{1,0} reshape(broadcast.21)
            broadcast.23 = f32[1,128,30522]{2,1,0} broadcast(reshape.22), dimensions={0,1}
            ROOT divide.24 = f32[1,128,30522]{2,1,0} divide(exponential.14, broadcast.23)
        }
    )";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(OneDnnSoftmaxTest, SoftmaxBF16) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const std::string hlo_string = R"(
        HloModule jit_softmax, entry_computation_layout={(bf16[1,128,30522]{2,1,0})->bf16[1,128,30522]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}
        region_0.4 {
            Arg_0.5 = bf16[] parameter(0)
            Arg_1.6 = bf16[] parameter(1)
            ROOT maximum.7 = bf16[] maximum(Arg_0.5, Arg_1.6)
        }
        region_1.15 {
            Arg_0.16 = bf16[] parameter(0)
            Arg_1.17 = bf16[] parameter(1)
            ROOT add.18 = bf16[] add(Arg_0.16, Arg_1.17)
        }
        ENTRY main.25 {
            Arg_0.1 = bf16[1,128,30522]{2,1,0} parameter(0), sharding={replicated}
            constant.3 = bf16[] constant(-inf)
            reduce.8 = bf16[1,128]{1,0} reduce(Arg_0.1, constant.3), dimensions={2}, to_apply=region_0.4
            reshape.9 = bf16[1,128,1]{2,1,0} reshape(reduce.8)
            broadcast.10 = bf16[1,128,1]{2,1,0} broadcast(reshape.9), dimensions={0,1,2}
            reshape.11 = bf16[1,128]{1,0} reshape(broadcast.10)
            broadcast.12 = bf16[1,128,30522]{2,1,0} broadcast(reshape.11), dimensions={0,1}
            subtract.13 = bf16[1,128,30522]{2,1,0} subtract(Arg_0.1, broadcast.12)
            exponential.14 = bf16[1,128,30522]{2,1,0} exponential(subtract.13)
            constant.2 = bf16[] constant(0)
            reduce.19 = bf16[1,128]{1,0} reduce(exponential.14, constant.2), dimensions={2}, to_apply=region_1.15
            reshape.20 = bf16[1,128,1]{2,1,0} reshape(reduce.19)
            broadcast.21 = bf16[1,128,1]{2,1,0} broadcast(reshape.20), dimensions={0,1,2}
            reshape.22 = bf16[1,128]{1,0} reshape(broadcast.21)
            broadcast.23 = bf16[1,128,30522]{2,1,0} broadcast(reshape.22), dimensions={0,1}
            ROOT divide.24 = bf16[1,128,30522]{2,1,0} divide(exponential.14, broadcast.23)
        }
    )";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(OneDnnSoftmaxTest, SoftmaxF32toBF16) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const std::string hlo_string = R"(
        HloModule jit_softmax, entry_computation_layout={(f32[16,128,30522]{2,1,0})->bf16[16,128,30522]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}
        region_0.4 {
            Arg_0.5 = f32[] parameter(0)
            Arg_1.6 = f32[] parameter(1)
            ROOT maximum.7 = f32[] maximum(Arg_0.5, Arg_1.6)
        }
        region_1.15 {
            Arg_0.16 = f32[] parameter(0)
            Arg_1.17 = f32[] parameter(1)
            ROOT add.18 = f32[] add(Arg_0.16, Arg_1.17)
        }
        ENTRY main.25 {
            Arg_0.1 = f32[16,128,30522]{2,1,0} parameter(0), sharding={replicated}
            constant.3 = f32[] constant(-inf)
            reduce.8 = f32[16,128]{1,0} reduce(Arg_0.1, constant.3), dimensions={2}, to_apply=region_0.4
            reshape.9 = f32[16,128,1]{2,1,0} reshape(reduce.8)
            broadcast.10 = f32[16,128,1]{2,1,0} broadcast(reshape.9), dimensions={0,1,2}
            reshape.11 = f32[16,128]{1,0} reshape(broadcast.10)
            broadcast.12 = f32[16,128,30522]{2,1,0} broadcast(reshape.11), dimensions={0,1}
            subtract.13 = f32[16,128,30522]{2,1,0} subtract(Arg_0.1, broadcast.12)
            exponential.14 = f32[16,128,30522]{2,1,0} exponential(subtract.13)
            constant.2 = f32[] constant(0)
            reduce.19 = f32[16,128]{1,0} reduce(exponential.14, constant.2), dimensions={2}, to_apply=region_1.15
            reshape.20 = f32[16,128,1]{2,1,0} reshape(reduce.19)
            broadcast.21 = f32[16,128,1]{2,1,0} broadcast(reshape.20), dimensions={0,1,2}
            reshape.22 = f32[16,128]{1,0} reshape(broadcast.21)
            broadcast.23 = f32[16,128,30522]{2,1,0} broadcast(reshape.22), dimensions={0,1}
            divide.24 = f32[16,128,30522]{2,1,0} divide(exponential.14, broadcast.23)
            ROOT convert.1 = bf16[16,128,30522]{2,1,0} convert(divide.24)
        }
    )";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
