/* Copyright 2026 The OpenXLA Authors.

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

#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {

namespace {

class GemmSyclTest : public HloPjRtInterpreterReferenceMixin<HloTestBase> {
 protected:
  void TestGemmWithTypeVariations(absl::string_view hlo_template) {
    std::vector<std::tuple<absl::string_view, absl::string_view>>
        type_combinations = {{"f32", "f32"}, {"f16", "f16"}, {"bf16", "bf16"}};

    for (const auto& type_combination : type_combinations) {
      VLOG(3) << "Testing type combination: " << std::get<0>(type_combination)
              << ", " << std::get<1>(type_combination);
      absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
      replacements["<<ABType>>"] = std::get<0>(type_combination);
      replacements["<<DType>>"] = std::get<1>(type_combination);
      const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
      double tol = 1e-2;
      if (std::get<0>(type_combination) == "f32" &&
          std::get<1>(type_combination) == "f32") {
        // f32 is more precise, so we can tighten the error bounds.
        tol = 1e-4;
      }
      EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{tol, tol}));
    }
  }
};

TEST_F(GemmSyclTest, MatmulNoFusion) {
  const char* hlo_module = R"(
  HloModule module

  ENTRY module {
    %parameter.1 = <<ABType>>[256,128]{1,0} parameter(0)
    %parameter.2 = <<ABType>>[128,8]{1,0} parameter(1)
    ROOT %dot = <<DType>>[256,8] dot(%parameter.1, %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
    )";
  TestGemmWithTypeVariations(hlo_module);
}

TEST_F(GemmSyclTest, MatmulWithBias) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test

  ENTRY matmul.biasadd.test {
    arg0.1 = <<ABType>>[32,32,40,30] parameter(0), parameter_replication={false}
    arg0.2 = <<ABType>>[32,32,30,40] parameter(1), parameter_replication={false}
    arg0.3 = <<ABType>>[40]{0} parameter(2), parameter_replication={false}
    dot.7 = <<DType>>[32,32,40,40] dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = <<DType>>[32,32,40,40] broadcast(arg0.3), dimensions={3}
    add.10 = <<DType>>[32,32,40,40] add(dot.7, broad.1)
    reshape.11 = <<DType>>[32,32,40,40] reshape(add.10)
    tuple.12 = (<<DType>>[32,32,40,40]) tuple(reshape.11)
    ROOT get-tuple-element.13 = <<DType>>[32,32,40,40] get-tuple-element(tuple.12), index=0
  })";
  TestGemmWithTypeVariations(matmul_module_str);
}

TEST_F(GemmSyclTest, MatmulWithRELU) {
  const char* hlo_module = R"(
  HloModule module

  ENTRY module {
    %parameter.1 = <<ABType>>[256,128]{1,0} parameter(0)
    %parameter.2 = <<ABType>>[128,8]{1,0} parameter(1)
    %dot = <<DType>>[256,8] dot(%parameter.1, %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    c = <<DType>>[] constant(0)
    c_bcast = <<DType>>[256,8] broadcast(c), dimensions={}
    ROOT out = <<DType>>[256,8] maximum(dot, c_bcast)
  }
    )";
  TestGemmWithTypeVariations(hlo_module);
}

TEST_F(GemmSyclTest, MatmulWithApproxGELU) {
  const char* matmul_module_str = R"(
  HloModule matmul.test
  ENTRY module {
    %parameter.1 = <<ABType>>[256,128]{1,0} parameter(0)
    %parameter.2 = <<ABType>>[128,8]{1,0} parameter(1)
    %dot = <<DType>>[256,8] dot(%parameter.1, %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    mul.0 = <<DType>>[256,8] multiply(dot, dot)
    mul.1 = <<DType>>[256,8] multiply(dot, mul.0)
    const.0 = <<DType>>[] constant(0.044715)
    bcast.0 = <<DType>>[256,8] broadcast(const.0), dimensions={}
    mul.2 = <<DType>>[256,8] multiply(mul.1, bcast.0)
    add.0 = <<DType>>[256,8] add(dot, mul.2)
    const.1 = <<DType>>[] constant(0.797884583)
    bcast.1 = <<DType>>[256,8] broadcast(const.1), dimensions={}
    mul.3 = <<DType>>[256,8] multiply(add.0, bcast.1)
    tanh = <<DType>>[256,8] tanh(mul.3)
    const.2 = <<DType>>[] constant(1)
    bcast.2 = <<DType>>[256,8] broadcast(const.2), dimensions={}
    add.2 = <<DType>>[256,8] add(tanh, bcast.2)
    const.3 = <<DType>>[] constant(0.5)
    bcast.3 = <<DType>>[256,8] broadcast(const.3), dimensions={}
    mul.4 = <<DType>>[256,8] multiply(add.2, bcast.3)
    ROOT out = <<DType>>[256,8] multiply(dot, mul.4)
  })";
  TestGemmWithTypeVariations(matmul_module_str);
}

}  // namespace
}  // namespace xla::gpu
