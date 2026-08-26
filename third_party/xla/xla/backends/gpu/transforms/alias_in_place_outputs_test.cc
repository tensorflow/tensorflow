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

#include "xla/backends/gpu/transforms/alias_in_place_outputs.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

class AliasInPlaceOutputsTest : public HloHardwareIndependentTestBase {
 protected:
  using AliasingList =
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>;

  // Output-to-operand aliasing of the single candidate instruction
  // (Triton fusion or cuBLASLt custom call) in the entry computation.
  const AliasingList& GetAliasing(const HloModule& module) {
    for (const HloInstruction* instr :
         module.entry_computation()->instructions()) {
      if (const auto* fusion = DynCast<HloFusionInstruction>(instr)) {
        return fusion->output_to_operand_aliasing();
      }
      if (const auto* call = DynCast<HloCustomCallInstruction>(instr)) {
        return call->output_to_operand_aliasing();
      }
    }
    LOG(FATAL) << "no candidate instruction in entry computation";
  }

  void ExpectAlias(absl::string_view hlo, int64_t operand,
                   ShapeIndex output = {}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         AliasInPlaceOutputs(&mlir_context_).Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_THAT(GetAliasing(*module),
                ElementsAre(Pair(output, Pair(operand, ShapeIndex{}))));
  }

  void ExpectNoAlias(absl::string_view hlo) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         AliasInPlaceOutputs(&mlir_context_).Run(module.get()));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(GetAliasing(*module).empty());
  }

  mlir::MLIRContext mlir_context_;
};

// --- Triton Fusion Tests ---

// dot(a, b) + x with x a dead intermediate => x is aliased to the output.
TEST_F(AliasInPlaceOutputsTest, TritonAliasesDeadBiasOperand) {
  ExpectAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add = f32[128,128] add(dot, x)
    }

    ENTRY e {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      c = f32[128,128] parameter(2)
      x = f32[128,128] add(c, c)
      ROOT fusion = f32[128,128] fusion(a, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    })",
              /*operand=*/2);
}

// Post-autotuning Triton GEMMs are converted to __triton_nested_gemm_fusion.
TEST_F(AliasInPlaceOutputsTest, TritonAliasesPostAutotuningNestedGemmFusion) {
  ExpectAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add = f32[128,128] add(dot, x)
    }

    ENTRY e {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      c = f32[128,128] parameter(2)
      x = f32[128,128] add(c, c)
      ROOT fusion = f32[128,128] fusion(a, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion"}}
    })",
              /*operand=*/2);
}

// The bias reaches the root through a non-identity access (transpose), so it is
// not safe to alias even though shapes match.
TEST_F(AliasInPlaceOutputsTest, TritonDoesNotAliasNonIdentityAccess) {
  ExpectNoAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      xt = f32[128,128] transpose(x), dimensions={1,0}
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add = f32[128,128] add(dot, xt)
    }

    ENTRY e {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      c = f32[128,128] parameter(2)
      x = f32[128,128] add(c, c)
      ROOT fusion = f32[128,128] fusion(a, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    })");
}

// The bias is a plain entry parameter (read-only input) => not beneficial.
TEST_F(AliasInPlaceOutputsTest, TritonDoesNotAliasParameterOperand) {
  ExpectNoAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add = f32[128,128] add(dot, x)
    }

    ENTRY e {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      ROOT fusion = f32[128,128] fusion(a, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    })");
}

// The bias has another user that is independent => not safe to alias.
TEST_F(AliasInPlaceOutputsTest, TritonDoesNotAliasIndependentLiveOperand) {
  ExpectNoAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add = f32[128,128] add(dot, x)
    }

    ENTRY e {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      c = f32[128,128] parameter(2)
      x = f32[128,128] add(c, c)
      fusion = f32[128,128] fusion(a, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
      ROOT t = (f32[128,128], f32[128,128]) tuple(fusion, x)
    })");
}

// The bias has another user, but that user is a predecessor of the fusion
// (residual pattern: a = norm(x); fusion = dot(a, b) + x). Safe to alias!
TEST_F(AliasInPlaceOutputsTest,
       TritonAliasesResidualOperandWithPredecessorUser) {
  ExpectAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add = f32[128,128] add(dot, x)
    }

    ENTRY e {
      c = f32[128,128] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] add(c, c)
      slice = f32[128,64] slice(x), slice={[0:128], [0:64]}
      ROOT fusion = f32[128,128] fusion(slice, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    })",
              /*operand=*/2);
}

// --- cuBLASLt Custom Call Tests ---

// __cublas$lt$matmul with beta=1 and dead bias operand => aliases output {0} to
// operand {2, {}}.
TEST_F(AliasInPlaceOutputsTest, CublasLtAliasesDeadBiasOperand) {
  ExpectAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      p2 = f32[128,128] parameter(2)
      bias = f32[128,128] add(p2, p2)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, bias),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })",
              /*operand=*/2, /*output=*/ShapeIndex{0});
}

// __cublas$lt$matmul with beta=0 => no aliasing.
TEST_F(AliasInPlaceOutputsTest, CublasLtDoesNotAliasWhenBetaIsZero) {
  ExpectNoAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      p2 = f32[128,128] parameter(2)
      bias = f32[128,128] add(p2, p2)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, bias),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":0.0}}
    })");
}

// __cublas$lt$matmul where bias is a parameter => no aliasing.
TEST_F(AliasInPlaceOutputsTest, CublasLtDoesNotAliasParameterOperand) {
  ExpectNoAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      p2 = f32[128,128] parameter(2)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, p2),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })");
}

// cuBLASLt in a residual chain (AlphaFold3 pattern):
// bias is used by an intermediate layer (e.g. norm/slice) that produces LHS of
// matmul. Because the other user precedes matmul, aliasing is safe!
TEST_F(AliasInPlaceOutputsTest, CublasLtAliasesResidualChain) {
  ExpectAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,128] parameter(0)
      p1 = f32[64,128] parameter(1)
      bias = f32[128,128] add(p0, p0)
      lhs = f32[128,64] slice(bias), slice={[0:128], [0:64]}
      matmul = (f32[128,128], s8[4096]) custom-call(lhs, p1, bias),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
      ROOT gte = f32[128,128] get-tuple-element(matmul), index=0
    })",
              /*operand=*/2, /*output=*/ShapeIndex{0});
}

// cuBLASLt where other user does NOT precede matmul => no aliasing.
TEST_F(AliasInPlaceOutputsTest, CublasLtDoesNotAliasIndependentUser) {
  ExpectNoAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      p2 = f32[128,128] parameter(2)
      bias = f32[128,128] add(p2, p2)
      matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, bias),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
      ROOT t = ((f32[128,128], s8[4096]), f32[128,128]) tuple(matmul, bias)
    })");
}

// cuBLASLt FP8 (__cublas$lt$matmul$f8) with beta=1.
TEST_F(AliasInPlaceOutputsTest, CublasLtF8AliasesDeadBiasOperand) {
  ExpectAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f8e4m3fn[128,64] parameter(0)
      p1 = f8e4m3fn[64,128] parameter(1)
      p2 = f32[128,128] parameter(2)
      bias = f32[128,128] add(p2, p2)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, bias),
        custom_call_target="__cublas$lt$matmul$f8",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })",
              /*operand=*/2, /*output=*/ShapeIndex{0});
}

// Grouped GEMM (__cublas$lt$groupedMatmul) with beta=1 (bias at operand index
// 3).
TEST_F(AliasInPlaceOutputsTest, CublasLtGroupedMatmulAliasesBiasOperand) {
  ExpectAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      sizes = s32[2] parameter(2)
      p3 = f32[128,128] parameter(3)
      bias = f32[128,128] add(p3, p3)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, sizes, bias),
        custom_call_target="__cublas$lt$groupedMatmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })",
              /*operand=*/3, /*output=*/ShapeIndex{0});
}

// Block-scaled MX matmul (__cublas$lt$matmul$mx) with beta=1.
TEST_F(AliasInPlaceOutputsTest, CublasLtMatmulMxAliasesDeadBiasOperand) {
  ExpectAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f8e4m3fn[128,64] parameter(0)
      p1 = f8e4m3fn[64,128] parameter(1)
      p2 = f32[128,128] parameter(2)
      bias = f32[128,128] add(p2, p2)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, bias),
        custom_call_target="__cublas$lt$matmul$mx",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })",
              /*operand=*/2, /*output=*/ShapeIndex{0});
}

// Triton fusion with tuple output is not supported for aliasing.
TEST_F(AliasInPlaceOutputsTest, TritonDoesNotAliasTupleFusion) {
  ExpectNoAlias(R"(
    HloModule m

    triton {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      x = f32[128,128] parameter(2)
      dot = f32[128,128] dot(a, b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      add = f32[128,128] add(dot, x)
      ROOT t = (f32[128,128], f32[128,128]) tuple(add, dot)
    }

    ENTRY e {
      a = f32[128,64] parameter(0)
      b = f32[64,128] parameter(1)
      c = f32[128,128] parameter(2)
      x = f32[128,128] add(c, c)
      ROOT fusion = (f32[128,128], f32[128,128]) fusion(a, b, x), kind=kCustom, calls=triton,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    })");
}

// cuBLASLt with constant bias operand => no aliasing.
TEST_F(AliasInPlaceOutputsTest, CublasLtDoesNotAliasConstantOperand) {
  ExpectNoAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      c = f32[128,128] constant({...})
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, c),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })");
}

// cuBLASLt where bias shape does not match output shape (e.g. vector broadcast
// bias) => no aliasing.
TEST_F(AliasInPlaceOutputsTest, CublasLtDoesNotAliasMismatchedShapeBias) {
  ExpectNoAlias(R"(
    HloModule m

    ENTRY e {
      p0 = f32[128,64] parameter(0)
      p1 = f32[64,128] parameter(1)
      p2 = f32[128] parameter(2)
      bias = f32[128] add(p2, p2)
      ROOT matmul = (f32[128,128], s8[4096]) custom-call(p0, p1, bias),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"beta":1.0}}
    })");
}

}  // namespace
}  // namespace xla::gpu
