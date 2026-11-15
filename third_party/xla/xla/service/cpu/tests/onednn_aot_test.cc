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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/xla.pb.h"

namespace xla::cpu {
namespace {

// Returns true if any instruction in `module` is a custom call to `target`.
bool HasCustomCall(const HloModule& module, absl::string_view target) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target() == target) {
        return true;
      }
    }
  }
  return false;
}

// End-to-end AOT test for oneDNN custom calls: JIT compiles a module that is
// rewritten into a oneDNN custom call, serializes it, loads it back, and checks
// that the loaded (AOT) executable produces the same result as the JIT compiled
// one. Serialization of a PjRt CPU executable goes through CpuCompiler::Export
// and LoadAotCompilationResult, so this exercises the oneDNN thunk serdes
// together with the rest of the export -> load -> execute path.
class OneDnnAotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CpuClientOptions options;
    options.cpu_device_count = 1;
    ASSERT_OK_AND_ASSIGN(client_, GetXlaPjrtCpuClient(std::move(options)));
  }

  static CompileOptions GetCompileOptions() {
    CompileOptions compile_options;
    DebugOptions& debug_options =
        *compile_options.executable_build_options.mutable_debug_options();
    debug_options = GetDebugOptionsFromFlags();
    debug_options.set_xla_cpu_experimental_onednn_custom_call(true);
    // YNN claims dots (and other ops) by default, which would prevent them from
    // being rewritten into oneDNN custom calls.
    debug_options.clear_xla_cpu_experimental_ynn_fusion_type();
    return compile_options;
  }

  absl::StatusOr<Literal> Execute(PjRtLoadedExecutable& executable,
                                  const std::vector<Literal>& arguments) {
    PjRtMemorySpace* memory_space = client_->memory_spaces()[0];
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    std::vector<PjRtBuffer*> buffer_ptrs;
    for (const Literal& argument : arguments) {
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer,
                       client_->BufferFromHostLiteral(argument, memory_space));
      buffer_ptrs.push_back(buffer.get());
      buffers.push_back(std::move(buffer));
    }
    ABSL_ASSIGN_OR_RETURN(auto results, executable.Execute({buffer_ptrs}, {}));
    if (results.size() != 1 || results[0].size() != 1) {
      return absl::InternalError("Expected exactly one result buffer");
    }
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<Literal> result,
                     results[0][0]->ToLiteralSync());
    return std::move(*result);
  }

  void ExportLoadAndCompareWithJit(absl::string_view hlo_string,
                                   absl::string_view onednn_target) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    ASSERT_OK_AND_ASSIGN(std::vector<Literal> arguments,
                         MakeFakeArguments(module.get()));

    // JIT compile the module.
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtLoadedExecutable> jit_executable,
        client_->CompileAndLoad(XlaComputation(module->ToProto()),
                                GetCompileOptions()));

    // Make sure the exported executable actually contains a oneDNN thunk.
    ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                         jit_executable->GetHloModules());
    ASSERT_EQ(hlo_modules.size(), 1);
    ASSERT_TRUE(HasCustomCall(*hlo_modules[0], onednn_target))
        << "Expected a " << onednn_target << " custom call in:\n"
        << hlo_modules[0]->ToString();

    // Export, serialize, deserialize and load the AOT executable.
    ASSERT_OK_AND_ASSIGN(std::string serialized,
                         jit_executable->SerializeExecutable());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> aot_executable,
                         client_->LoadSerializedExecutable(
                             serialized, std::nullopt, LoadOptions()));

    // Run both executables on the same inputs and compare the results.
    ASSERT_OK_AND_ASSIGN(Literal jit_result,
                         Execute(*jit_executable, arguments));
    ASSERT_OK_AND_ASSIGN(Literal aot_result,
                         Execute(*aot_executable, arguments));

    // Both executables run the same oneDNN primitives on the same inputs, so
    // the results are expected to match.
    EXPECT_TRUE(
        LiteralTestUtil::Near(jit_result, aot_result, ErrorSpec{1e-6, 1e-6}));
  }

  std::unique_ptr<PjRtClient> client_;
};

TEST_F(OneDnnAotTest, MatMul) {
  constexpr absl::string_view kHlo = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
    arg.0 = f32[32,8,128,64] parameter(0), parameter_replication={false}
    arg.1 = f32[32,8,64,128] parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[32,8,128,128] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  ExportLoadAndCompareWithJit(kHlo, "__onednn$matmul");
}

TEST_F(OneDnnAotTest, Convolution) {
  constexpr absl::string_view kHlo = R"(
  HloModule convolution.test

  ENTRY convolution.test {
    arg.0 = f32[1,22,22,1] parameter(0)
    reshape.0 = f32[1,22,22,1] reshape(arg.0)
    arg.1 = f32[8,8,1,1] parameter(1)
    reshape.1 = f32[8,8,1,1] reshape(arg.1)
    convolution.0 = f32[1,11,11,1] convolution(reshape.0, reshape.1),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    reshape.2 = f32[1,11,11,1] reshape(convolution.0)
    tuple.0 = (f32[1,11,11,1]) tuple(reshape.2)
    ROOT gte.0 = f32[1,11,11,1] get-tuple-element(tuple.0), index=0
  })";

  ExportLoadAndCompareWithJit(kHlo, "__onednn$convolution");
}

TEST_F(OneDnnAotTest, LayerNorm) {
  constexpr absl::string_view kHlo = R"(
  HloModule layer_norm.test, entry_computation_layout={(f32[84,197,768]{2,1,0}, f32[768]{0}, f32[768]{0})->f32[84,197,768]{2,1,0}}

  region_add {
    Arg_0.7555 = f32[] parameter(0)
    Arg_1.7556 = f32[] parameter(1)
    ROOT add.7557 = f32[] add(Arg_0.7555, Arg_1.7556)
  }

  ENTRY main {
    Arg_0.1 = f32[84,197,768]{2,1,0} parameter(0), sharding={replicated}
    Arg_0.2 = f32[768]{0} parameter(1), sharding={replicated}
    Arg_0.3 = f32[768]{0} parameter(2), sharding={replicated}

    convert.290 = f32[84,197,768]{2,1,0} convert(Arg_0.1)
    constant.291 = f32[] constant(0)
    convert.292 = f32[] convert(constant.291)
    reduce.297 = f32[84,197]{1,0} reduce(convert.290, convert.292),
                 dimensions={2}, to_apply=region_add
    constant.298 = s32[] constant(768)
    convert.299 = f32[] convert(constant.298)
    broadcast.300 = f32[84,197]{1,0} broadcast(convert.299), dimensions={}
    divide.301 = f32[84,197]{1,0} divide(reduce.297, broadcast.300)
    convert.302 = f32[84,197]{1,0} convert(divide.301)
    reshape.303 = f32[84,197,1]{2,1,0} reshape(convert.302)
    reshape.304 = f32[84,197]{1,0} reshape(reshape.303)
    broadcast.305 = f32[84,197,768]{2,1,0} broadcast(reshape.304),
                    dimensions={0,1}
    subtract.306 = f32[84,197,768]{2,1,0} subtract(Arg_0.1, broadcast.305)
    multiply.307 = f32[84,197,768]{2,1,0} multiply(subtract.306, subtract.306)
    convert.308 = f32[84,197,768]{2,1,0} convert(multiply.307)
    constant.309 = f32[] constant(0)
    convert.310 = f32[] convert(constant.309)
    reduce.315 = f32[84,197]{1,0} reduce(convert.308, convert.310),
                 dimensions={2}, to_apply=region_add
    constant.316 = s32[] constant(768)
    convert.317 = f32[] convert(constant.316)
    broadcast.318 = f32[84,197]{1,0} broadcast(convert.317), dimensions={}
    divide.319 = f32[84,197]{1,0} divide(reduce.315, broadcast.318)
    convert.320 = f32[84,197]{1,0} convert(divide.319)
    reshape.321 = f32[84,197,1]{2,1,0} reshape(convert.320)
    constant.322 = f32[] constant(1e-12)
    broadcast.323 = f32[84,197,1]{2,1,0} broadcast(constant.322), dimensions={}
    add.324 = f32[84,197,1]{2,1,0} add(reshape.321, broadcast.323)
    rsqrt.325 = f32[84,197,1]{2,1,0} rsqrt(add.324)
    reshape.328 = f32[84,197]{1,0} reshape(rsqrt.325)
    broadcast.329 = f32[84,197,768]{2,1,0} broadcast(reshape.328),
                    dimensions={0,1}
    broadcast.327 = f32[84,197,768]{2,1,0} broadcast(Arg_0.2), dimensions={2}
    multiply.330 = f32[84,197,768]{2,1,0} multiply(broadcast.329, broadcast.327)
    multiply.331 = f32[84,197,768]{2,1,0} multiply(Arg_0.1, multiply.330)
    broadcast.336 = f32[84,197,768]{2,1,0} broadcast(Arg_0.3), dimensions={2}
    reshape.332 = f32[84,197]{1,0} reshape(reshape.303)
    broadcast.333 = f32[84,197,768]{2,1,0} broadcast(reshape.332),
                    dimensions={0,1}
    multiply.334 = f32[84,197,768]{2,1,0} multiply(multiply.330, broadcast.333)
    subtract.337 = f32[84,197,768]{2,1,0} subtract(broadcast.336, multiply.334)
    ROOT add.338 = f32[84,197,768]{2,1,0} add(multiply.331, subtract.337)
  })";

  ExportLoadAndCompareWithJit(kHlo, "__onednn$layernorm");
}

TEST_F(OneDnnAotTest, Softmax) {
  constexpr absl::string_view kHlo = R"(
  HloModule softmax_module

  region_max {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT maximum = f32[] maximum(Arg_0, Arg_1)
  }

  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add = f32[] add(Arg_0, Arg_1)
  }

  ENTRY main {
    Arg_0 = f32[1,128,30522]{2,1,0} parameter(0)
    neg_inf = f32[] constant(-inf)
    reduce_max = f32[1,128]{1,0} reduce(Arg_0, neg_inf), dimensions={2},
                 to_apply=region_max
    reshape.0 = f32[1,128,1]{2,1,0} reshape(reduce_max)
    broadcast.0 = f32[1,128,1]{2,1,0} broadcast(reshape.0), dimensions={0,1,2}
    reshape.1 = f32[1,128]{1,0} reshape(broadcast.0)
    broadcast.1 = f32[1,128,30522]{2,1,0} broadcast(reshape.1),
                  dimensions={0,1}
    subtract.0 = f32[1,128,30522]{2,1,0} subtract(Arg_0, broadcast.1)
    exponential = f32[1,128,30522]{2,1,0} exponential(subtract.0)
    const_zero = f32[] constant(0)
    reduce_add = f32[1,128]{1,0} reduce(exponential, const_zero),
                 dimensions={2}, to_apply=region_add
    reshape.2 = f32[1,128,1]{2,1,0} reshape(reduce_add)
    broadcast.2 = f32[1,128,1]{2,1,0} broadcast(reshape.2), dimensions={0,1,2}
    reshape.3 = f32[1,128]{1,0} reshape(broadcast.2)
    broadcast.3 = f32[1,128,30522]{2,1,0} broadcast(reshape.3),
                  dimensions={0,1}
    ROOT divide = f32[1,128,30522]{2,1,0} divide(exponential, broadcast.3)
  })";

  ExportLoadAndCompareWithJit(kHlo, "__onednn$softmax");
}

}  // namespace
}  // namespace xla::cpu
