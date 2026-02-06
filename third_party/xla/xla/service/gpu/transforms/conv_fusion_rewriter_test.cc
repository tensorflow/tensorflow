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

#include "xla/service/gpu/transforms/conv_fusion_rewriter.h"

#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/transforms/conv_kind_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

using ::testing::HasSubstr;
using ::testing::Not;

static const std::initializer_list<absl::string_view> kf16f32{"f16", "f32"};

class ConvFusionRewriterHloTest : public HloTestBase {
 public:
  bool IsCuda() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability()
        .IsCuda();
  }
  se::CudaComputeCapability GetCudaComputeCapability() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() const {
    return GetDnnVersionInfoOrDefault(backend().default_stream_executor());
  }

  se::SemanticVersion GetToolkitVersion() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .runtime_version();
  }

  ConvFusionRewriter GetConvFusionRewriter() const {
    return ConvFusionRewriter();
  }
  ConvKindAssignment GetConvKindAssignment() const {
    return ConvKindAssignment(GetCudaComputeCapability(), GetDnnVersion());
  }
  ConvFusionRewriterHloTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    /*instruction_can_change_layout_func=*/{}) {}
};

class ConvFusionRewriterTest : public GpuCodegenTest {
 public:
  bool IsCuda() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability()
        .IsCuda();
  }
  se::CudaComputeCapability GetCudaComputeCapability() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() const {
    return GetDnnVersionInfoOrDefault(backend().default_stream_executor());
  }

  stream_executor::SemanticVersion GetToolkitVersion() const {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .runtime_version();
  }

 protected:
  std::string GetOptimizedHlo(absl::string_view hlo_string) {
    // cudnn_vectorize_convolutions transforms convolutions, making it hard to
    // match them here in this test.  What's worse, the transforms it does
    // depends on the GPU that's available!  So just disable them for this
    // function that gets the optimized HLO.  When we actually run the module
    // we'll still have this pass enabled.
    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions debug_opts = config.debug_options();
    debug_opts.add_xla_disable_hlo_passes("cudnn_vectorize_convolutions");
    debug_opts.set_xla_gpu_use_runtime_fusion(true);
    debug_opts.set_xla_gpu_experimental_enable_conv_fusion(true);
    config.set_debug_options(debug_opts);

    auto result = backend().compiler()->RunHloPasses(
        ParseAndReturnVerifiedModule(hlo_string, config).value(),
        backend().default_stream_executor(), backend().memory_allocator());
    if (!result.status().ok()) {
      TF_EXPECT_OK(result.status())
          << "HLO compilation failed: " << result.status();
      return "";
    }
    HloPrintOptions print_opts;
    print_opts.set_print_operand_shape(false);
    return (*result)->ToString(print_opts);
  }

  void TestMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : kf16f32) {
      const std::string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

      TF_ASSERT_OK_AND_ASSIGN(auto module,
                              ParseAndReturnVerifiedModule(hlo_with_new_type));
      DebugOptions debug_opts = module->config().debug_options();
      debug_opts.set_xla_gpu_use_runtime_fusion(true);
      debug_opts.set_xla_gpu_experimental_enable_conv_fusion(true);
      module->mutable_config().set_debug_options(debug_opts);
      EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01}))
          << optimized_hlo_string;
    }
  }

  void TestClamp(absl::string_view pre_hlo_string,
                 absl::string_view post_hlo_string) {
    std::string optimized_hlo_string = GetOptimizedHlo(pre_hlo_string);
    EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(pre_hlo_string));
    DebugOptions debug_opts = module->config().debug_options();
    debug_opts.set_xla_gpu_experimental_enable_conv_fusion(true);
    module->mutable_config().set_debug_options(debug_opts);

    EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01}))
        << pre_hlo_string;

    absl::StatusOr<bool> filecheck_result =
        RunFileCheck(optimized_hlo_string, post_hlo_string);
    ASSERT_TRUE(filecheck_result.ok()) << filecheck_result.status();
    EXPECT_TRUE(*filecheck_result);
  }

  void TestNotMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : kf16f32) {
      const std::string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      SCOPED_TRACE(optimized_hlo_string);
      EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));
    }
  }

  void TestF8(std::string pre_hlo_string, std::string fusion_string,
              std::string fusion_comp_string) {
    if (!IsCuda()) return;

    bool fp8_supported = GetDnnVersion() >= se::dnn::VersionInfo{9, 8, 0}
                             ? GetCudaComputeCapability().IsAtLeastAda()
                             : GetCudaComputeCapability().IsAtLeastHopper();
    LOG(INFO) << "RRR fp8_supported: " << fp8_supported;
    if (fp8_supported) {
      // On Ada/Hopper and newer architectures, test numerical correctness and
      // verify the HLO of the Custom Call with operand and return layouts and
      // the serialized graph based on the full compiler pipeline.
      std::string optimized_hlo_string = GetOptimizedHlo(pre_hlo_string);
      EXPECT_THAT(optimized_hlo_string, Not(HasSubstr("Convert")));
      EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

      TF_ASSERT_OK_AND_ASSIGN(auto module,
                              ParseAndReturnVerifiedModule(pre_hlo_string));
      DebugOptions debug_opts = module->config().debug_options();
      debug_opts.set_xla_gpu_experimental_enable_conv_fusion(true);
      module->mutable_config().set_debug_options(debug_opts);
      EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.15, 0.15}))
          << pre_hlo_string;
      absl::StatusOr<bool> filecheck_result =
          RunFileCheck(optimized_hlo_string, fusion_string);
      ASSERT_TRUE(filecheck_result.ok()) << filecheck_result.status();
      EXPECT_TRUE(*filecheck_result);

      filecheck_result = RunFileCheck(optimized_hlo_string, fusion_comp_string);
      ASSERT_TRUE(filecheck_result.ok()) << filecheck_result.status();
      EXPECT_TRUE(*filecheck_result);
    }
    return;
  }

  void TestF8Parameterized(std::string template_pre_hlo_string,
                           std::string template_fusion_string,
                           std::string template_fusion_comp_string) {
    std::array<absl::string_view, 2> types = {"f8e4m3fn", "f8e5m2"};
    std::array<absl::string_view, 2> clamp_lower = {"-448.", "-57344."};
    std::array<absl::string_view, 2> clamp_upper = {"448.", "57344."};
    absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
    for (int i = 0; i < 2; ++i) {
      replacements["<<InputType>>"] = types[i];
      for (int j = 0; j < 2; ++j) {
        replacements["<<FilterType>>"] = types[j];
        for (int k = 0; k < 2; ++k) {
          replacements["<<OutputType>>"] = types[k];
          replacements["<<ClampLower>>"] = clamp_lower[k];
          replacements["<<ClampUpper>>"] = clamp_upper[k];
          TestF8(
              absl::StrReplaceAll(template_pre_hlo_string, replacements),
              absl::StrReplaceAll(template_fusion_string, replacements),
              absl::StrReplaceAll(template_fusion_comp_string, replacements));
        }
      }
    }
  }
};

#define MAYBE_SKIP_TEST(CAUSE)                                    \
  do {                                                            \
    if (absl::string_view(CAUSE) == "F8" && IsCuda() &&           \
        GetToolkitVersion() < se::SemanticVersion{12, 0, 0}) {    \
      GTEST_SKIP() << "FP8 convolutions require CUDA 12.";        \
    }                                                             \
    if (!IsCuda()) {                                              \
      GTEST_SKIP() << CAUSE " fusion is only supported on CUDA."; \
    }                                                             \
  } while (0)

TEST_F(ConvFusionRewriterTest, TestConvOnly) {
  // max(0, conv(x, w));
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,32,9,9] broadcast(zero), dimensions={}

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,17,32] parameter(1)

      conv = TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      ROOT relu = TYPE[1,32,9,9] maximum(zeros, conv)
    })");
}

TEST_F(ConvFusionRewriterTest, TestBias) {
  // max(0, conv(x, w) + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(ConvFusionRewriterTest, Test3D) {
  // max(0, conv(x, w) + bias);
  std::string body = R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,5,7,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,5,7,64] parameter(0)
      filter = TYPE[3,3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,5,7,64] convolution(input, filter), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,5,7,64] broadcast(bias), dimensions={4}
      add1 = TYPE[1,3,5,7,64] add(conv, broadcasted_bias)
    )";

  std::string relu = R"(
      ROOT relu = TYPE[1,3,5,7,64] maximum(zeros, add1)
    })";

  std::string elu = R"(
      cmp = pred[1,3,5,7,64] compare(add1, zeros), direction=GT
      expm1 = TYPE[1,3,5,7,64] exponential-minus-one(add1)
      ROOT elu = TYPE[1,3,5,7,64] select(cmp, add1, expm1)
    })";

  TestMatchWithAllTypes(body + relu);
  if (!IsCuda()) TestMatchWithAllTypes(body + elu);
}

TEST_F(ConvFusionRewriterTest, TestBiasMultiCall) {
  // max(0, conv(x, w) + bias);
  std::string code = R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,<<<format>>>,64] broadcast(zero), dimensions={}

      input = TYPE[1,<<<format>>>,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,<<<format>>>,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,<<<format>>>,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,<<<format>>>,64] add(conv, broadcasted_bias)
      ROOT relu = TYPE[1,<<<format>>>,64] maximum(zeros, add1)
    })";
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
  replacements["<<<format>>>"] = "3,3";
  TestMatchWithAllTypes(absl::StrReplaceAll(code, replacements));
  replacements["<<<format>>>"] = "5,5";
  TestMatchWithAllTypes(absl::StrReplaceAll(code, replacements));
  replacements["<<<format>>>"] = "3,3";
  TestMatchWithAllTypes(absl::StrReplaceAll(code, replacements));
}

TEST_F(ConvFusionRewriterTest, TestBiasNoRelu) {
  // conv(x, w) + bias;
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      ROOT add1 = TYPE[1,3,3,64] add(conv, broadcasted_bias)
    })");
}

TEST_F(ConvFusionRewriterTest, TestElu) {
  // sum = conv(x, w) + bias
  // select(compare(sum, 0, GT), sum, exponential-minus-one(sum));
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      sum = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      cmp = pred[1,3,3,64] compare(sum, zeros), direction=GT
      expm1 = TYPE[1,3,3,64] exponential-minus-one(sum)
      ROOT elu = TYPE[1,3,3,64] select(cmp, sum, expm1)
    })");
}

TEST_F(ConvFusionRewriterTest, TestRelu6) {
  if (IsCuda() && !GetCudaComputeCapability().IsAtLeast(
                      se::CudaComputeCapability::kAmpere)) {
    GTEST_SKIP() << "Conv-Bias-Relu6 fusion is supported and recommended with "
                    "the Nvidia Ampere+ GPUs.";
  }
  // sum = conv(x, w) + bias
  // clamp(0, sum, 6);
  TestMatchWithAllTypes(R"(
    HloModule Test
    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      six = TYPE[] constant(6)
      sixes = TYPE[1,3,3,64] broadcast(six), dimensions={}
      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)
      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      sum = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      ROOT relu6 = TYPE[1,3,3,64] clamp(zeros, sum, sixes)
    })");
}

// At time of writing, cudnn runtime fusion cannot handle f16 convs with an odd
// number of input/output channels.  Check that we don't try to run this conv
// with runtime fusion (or, if we do, that it works!).
TEST_F(ConvFusionRewriterTest, TestRelu6OddChannels) {
  if (IsCuda() && !GetCudaComputeCapability().IsAtLeast(
                      se::CudaComputeCapability::kAmpere)) {
    GTEST_SKIP() << "Conv-Bias-Relu6 fusion is supported and recommended with "
                    "the Nvidia Ampere+ GPUs.";
  }
  TestMatchWithAllTypes(R"(
    HloModule Test
    ENTRY Test {
      zeros = TYPE[1,384,1024,32] broadcast(TYPE[] constant(0)), dimensions={}
      sixes = TYPE[1,384,1024,32] broadcast(TYPE[] constant(6)), dimensions={}
      input = TYPE[1,769,2049,3] parameter(0)
      filter = TYPE[32,3,3,3] parameter(1)
      bias = TYPE[32] parameter(2)
      conv = TYPE[1,384,1024,32] convolution(input, filter), window={size=3x3 stride=2x2}, dim_labels=b01f_o01i->b01f
      broadcasted_bias = TYPE[1,384,1024,32] broadcast(bias), dimensions={3}
      sum = add(conv, broadcasted_bias)
      ROOT relu6 = clamp(zeros, sum, sixes)
    })");
}

TEST_F(ConvFusionRewriterTest, TestLeakyRelu) {
  if (IsCuda() && !GetCudaComputeCapability().IsAtLeast(
                      se::CudaComputeCapability::kAmpere)) {
    GTEST_SKIP()
        << "Conv-Bias-LeakyRelu fusion is supported and recommended with "
           "the Nvidia Ampere+ GPUs.";
  }
  // sum = conv(x, w) + bias
  // select(compare(sum, 0, GT), sum, multiply(sum, alpha));
  TestMatchWithAllTypes(R"(
    HloModule Test
    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha = TYPE[] constant(0.2)
      alphas = TYPE[1,3,3,64] broadcast(alpha), dimensions={}
      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      bias = TYPE[64] parameter(2)
      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      sum = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      cmp = pred[1,3,3,64] compare(sum, zeros), direction=GT
      mul = TYPE[1,3,3,64] multiply(sum, alphas)
      ROOT elu = TYPE[1,3,3,64] select(cmp, sum, mul)
    })");
}

TEST_F(ConvFusionRewriterTest, TestSideInputOnly) {
  // max(0, conv(x, w) + side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      add1 = TYPE[1,3,3,64] add(conv, side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(ConvFusionRewriterTest, TestBiasAndSideInput) {
  // max(0, conv(x, w) + side_input + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)
      bias = TYPE[64] parameter(3)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      add2 = TYPE[1,3,3,64] add(add1, side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(ConvFusionRewriterTest, TestScaledConv) {
  // max(0, 0.999994934 * conv(x, w));
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,32,9,9] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,17,32] parameter(1)

      conv = TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      alpha_conv = TYPE[1,32,9,9] broadcast(alpha_conv_scalar), dimensions={}
      scaled_conv = TYPE[1,32,9,9] multiply(conv, alpha_conv)
      ROOT relu = TYPE[1,32,9,9] maximum(zeros, scaled_conv)
    })");
}

TEST_F(ConvFusionRewriterTest, TestNoCrashOnInf) {
  EXPECT_TRUE(RunAndCompare(R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(inf)
      zeros = f32[1,32,9,9] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)

      input = f32[1,17,9,9] parameter(0)
      filter = f32[3,3,17,32] parameter(1)

      conv = f32[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      alpha_conv = f32[1,32,9,9] broadcast(alpha_conv_scalar), dimensions={}
      scaled_conv = f32[1,32,9,9] multiply(conv, alpha_conv)
      ROOT relu = f32[1,32,9,9] maximum(zeros, scaled_conv)
    })",
                            ErrorSpec{0.01}));
}

TEST_F(ConvFusionRewriterTest, TestConvAndScaledSideInput) {
  // max(0, conv(x, w) + 0.899994934 * side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = TYPE[1,3,3,64] add(conv, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(ConvFusionRewriterTest, TestScaledConvAndScaledSideInput) {
  // max(0, 0.999994934 * conv(x, w) + 0.899994934 * side_input);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)
      alpha_conv = TYPE[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_conv = TYPE[1,3,3,64] multiply(conv, alpha_conv)
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = TYPE[1,3,3,64] add(scaled_conv, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(ConvFusionRewriterTest, TestScaledConvAndScaledSideInputWithBias) {
  // max(0, 0.999994934 * conv(x, w) + 0.899994934 * side_input + bias);
  TestMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)
      alpha_conv = TYPE[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,64,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)
      bias = TYPE[64] parameter(3)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1
      scaled_conv = TYPE[1,3,3,64] multiply(conv, alpha_conv)
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = TYPE[1,3,3,64] add(add1, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add2)
    })");
}

TEST_F(ConvFusionRewriterTest, TestMatchMaxZeroOnly) {
  // max(0.1, conv(x, w)) shouldn't match.
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      point_one = TYPE[] constant(0.1)
      point_ones = TYPE[1,32,9,9] broadcast(point_one), dimensions={}

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,17,32] parameter(1)

      conv = TYPE[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
      ROOT relu = TYPE[1,32,9,9] maximum(point_ones, conv)
    })");
}

TEST_F(ConvFusionRewriterTest, TestPreservesFeatureGroupCount) {
  // The convolution below would crash if feature_count is not preserved.
  const char* kHloString = R"(
    HloModule jaxpr_computation__6.19

    primitive_computation__1.4 {
      parameter.5 = f32[] parameter(0)
      parameter.6 = f32[] parameter(1)
      ROOT add.7 = f32[] add(parameter.5, parameter.6)
    }

    ENTRY jaxpr_computation__7.8 {
      parameter.11 = f32[2,64,64,53]{3,2,1,0} parameter(1)
      parameter.10 = f32[3,3,1,53]{3,2,1,0} parameter(0)
      convolution.12 = f32[2,64,64,53]{3,2,1,0} convolution(parameter.11, parameter.10), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=53
      constant.13 = f32[] constant(0)
      broadcast.14 = f32[2,64,64,53]{3,2,1,0} broadcast(constant.13), dimensions={}
      maximum.15 = f32[2,64,64,53]{3,2,1,0} maximum(convolution.12, broadcast.14)
      ROOT reduce.17 = f32[] reduce(maximum.15, constant.13), dimensions={0,1,2,3}, to_apply=primitive_computation__1.4
    }
  )";
  EXPECT_TRUE(RunAndCompare(kHloString, ErrorSpec{0.01}));
}

TEST_F(ConvFusionRewriterTest, TestConvF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       ROOT conv_a = f8e4m3fn[1,16,6,6] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledOutputF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: clamp(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvInvscaledOutputF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled = f32[1,16,6,6] divide(conv_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: divide(
// CHECK: clamp(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledF8Parameterized) {
  MAYBE_SKIP_TEST("F8");
  TestF8Parameterized(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = <<InputType>>[1,128,6,6] parameter(0)
       filter = <<FilterType>>[3,3,128,16] parameter(1)
       input_scale = f32[] parameter(2)
       input_scale_bcast = f32[1,128,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[3,3,128,16] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       input_unscaled = f32[1,128,6,6] multiply(input_f32, input_scale_bcast)
       filter_f32 = f32[3,3,128,16] convert(filter)
       filter_unscaled = f32[3,3,128,16] multiply(filter_f32, filter_scale_bcast)
       z_scale = f32[] parameter(4)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_unscaled, filter_unscaled), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a, z_scale_bcast)
       c1 = f32[] constant(<<ClampLower>>)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(<<ClampUpper>>)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       ROOT conv_f8 = <<OutputType>>[1,16,6,6] convert(conv_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = <<OutputType>>[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
      )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: clamp(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledBiasF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_scale = f32[] parameter(2)
       input_scale_bcast = f32[1,128,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[3,3,128,16] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       input_unscaled = f32[1,128,6,6] multiply(input_f32, input_scale_bcast)
       filter_f32 = f32[3,3,128,16] convert(filter)
       filter_unscaled = f32[3,3,128,16] multiply(filter_f32, filter_scale_bcast)
       bias = f32[1,16,6,6] parameter(4)
       z_scale = f32[] parameter(5)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_unscaled, filter_unscaled), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_bias = f32[1,16,6,6] add(conv_a, bias)
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a_bias, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]])
      )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: add(
// CHECK: multiply(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledReluF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       c = f32[] constant(0)
       c_bcast = f32[1,16,6,6] broadcast(c), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       relu_a = f32[1,16,6,6] maximum(conv_a, c_bcast)
       relu_a_scaled = f32[1,16,6,6] multiply(relu_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       relu_a_clamped = f32[1,16,6,6] clamp(c1_bcast, relu_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(relu_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: maximum(
// CHECK: multiply(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledRelu6F8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       c0 = f32[] constant(0)
       c0_bcast = f32[1,16,6,6] broadcast(c0), dimensions={}
       c6 = f32[] constant(6)
       c6_bcast = f32[1,16,6,6] broadcast(c6), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       relu6_a = f32[1,16,6,6] clamp(c0_bcast, conv_a, c6_bcast)
       relu6_a_scaled = f32[1,16,6,6] multiply(relu6_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       relu6_a_clamped = f32[1,16,6,6] clamp(c1_bcast, relu6_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(relu6_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: clamp(
// CHECK: multiply(
// CHECK: clamp(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledEluF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       c0 = f32[] constant(0)
       c0_bcast = f32[1,16,6,6] broadcast(c0), dimensions={}
       alpha = f32[] constant(0.2)
       alpha_bcast = f32[1,16,6,6] broadcast(alpha), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_compare = compare(conv_a, c0_bcast), direction=GT
       expm1_a = exponential-minus-one(conv_a)
       elu_a = select(conv_a_compare, conv_a, expm1_a)
       elu_a_scaled = f32[1,16,6,6] multiply(elu_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       elu_a_clamped = f32[1,16,6,6] clamp(c1_bcast, elu_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(elu_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: compare(
// CHECK: select(
// CHECK: multiply(
// CHECK: clamp(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledLeakyReluF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       c0 = f32[] constant(0)
       c0_bcast = f32[1,16,6,6] broadcast(c0), dimensions={}
       alpha = f32[] constant(0.2)
       alpha_bcast = f32[1,16,6,6] broadcast(alpha), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_compare = compare(conv_a, c0_bcast), direction=GT
       conv_a_alpha = multiply(conv_a, alpha_bcast)
       lrelu_a = select(conv_a_compare, conv_a, conv_a_alpha)
       lrelu_a_scaled = f32[1,16,6,6] multiply(lrelu_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       lrelu_a_clamped = f32[1,16,6,6] clamp(c1_bcast, lrelu_a_scaled, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(lrelu_a_clamped)

    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: compare(
// CHECK: select(
// CHECK: multiply(
// CHECK: clamp(
// CHECK: convert(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvAmaxF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_scale = f32[] parameter(2)
       input_scale_bcast = f32[1,16,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[1,16,6,6] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_input_scaled = f32[1,16,6,6] multiply(conv_a, input_scale_bcast)
       conv_a_filter_scaled = f32[1,16,6,6] multiply(conv_a_input_scaled, filter_scale_bcast)
       z_scale = f32[] parameter(4)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a_filter_scaled, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       conv_a_clamped_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
       abs_conv_a = f32[1,16,6,6] abs(conv_a_filter_scaled)
       c0 = f32[] constant(-inf)
       amax = f32[] reduce(abs_conv_a, c0), dimensions={0,1,2,3}, to_apply=apply
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f32[]) tuple(conv_a_clamped_f8, amax)
    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, f32[]) fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]])
    )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: abs(
// CHECK: reduce(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvReluAmaxF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_scale = f32[] parameter(2)
       input_scale_bcast = f32[1,16,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[1,16,6,6] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       c = f32[] constant(0)
       c_bcast = f32[1,16,6,6] broadcast(c), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_input_scaled = f32[1,16,6,6] multiply(conv_a, input_scale_bcast)
       conv_a_filter_scaled = f32[1,16,6,6] multiply(conv_a_input_scaled, filter_scale_bcast)
       relu_a = f32[1,16,6,6] maximum(conv_a_filter_scaled, c_bcast)
       z_scale = f32[] parameter(4)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       relu_a_scaled = f32[1,16,6,6] multiply(relu_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       relu_a_clamped = f32[1,16,6,6] clamp(c1_bcast, relu_a_scaled, c2_bcast)
       relu_a_clamped_f8 = f8e4m3fn[1,16,6,6] convert(relu_a_clamped)
       abs_relu_a = f32[1,16,6,6] abs(relu_a)
       c0 = f32[] constant(-inf)
       amax = f32[] reduce(abs_relu_a, c0), dimensions={0,1,2,3}, to_apply=apply
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f32[]) tuple(relu_a_clamped_f8, amax)
    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, f32[]) fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]])
    )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: maximum(
// CHECK: reduce(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledOutputMultipleUsersF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale0 = f32[] parameter(2)
       z_scale0_bcast = f32[1,16,6,6] broadcast(z_scale0), dimensions={}
       z_scale1 = f32[] parameter(3)
       z_scale1_bcast = f32[1,16,6,6] broadcast(z_scale1), dimensions={}
       z_scale2 = f32[] parameter(4)
       z_scale2_bcast = f32[1,16,6,6] broadcast(z_scale2), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled0 = f32[1,16,6,6] multiply(conv_a, z_scale0_bcast)
       conv_a_scaled1 = f32[1,16,6,6] multiply(conv_a_scaled0, z_scale1_bcast)
       conv_a_scaled2 = f32[1,16,6,6] multiply(conv_a_scaled0, z_scale2_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped0 = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled1, c2_bcast)
       conv_a_clamped1 = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled2, c2_bcast)
       conv_a_convert0 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped0)
       conv_a_convert1 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped1)
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f8e4m3fn[1,16,6,6]) tuple(conv_a_convert0, conv_a_convert1)

    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f32[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply({{.*}}
// CHECK-NEXT: }
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledOutputMultipleUsersInGraphAddF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale0 = f32[] parameter(2)
       z_scale0_bcast = f32[1,16,6,6] broadcast(z_scale0), dimensions={}
       z_scale1 = f32[] parameter(3)
       z_scale1_bcast = f32[1,16,6,6] broadcast(z_scale1), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled0 = f32[1,16,6,6] multiply(conv_a, z_scale0_bcast)
       conv_a_scaled1 = f32[1,16,6,6] multiply(conv_a, z_scale1_bcast)
       conv_a_scaled_sum = f32[1,16,6,6] add(conv_a_scaled0, conv_a_scaled1)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled_sum, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: add(
// CHECK: clamp(
// CHECK: convert({{.*}}
// CHECK-NEXT: }
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest,
       TestConvScaledOutputMultipleUsersInGraphDoubleAddF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale0 = f32[] parameter(2)
       z_scale0_bcast = f32[1,16,6,6] broadcast(z_scale0), dimensions={}
       z_scale1 = f32[] parameter(3)
       z_scale1_bcast = f32[1,16,6,6] broadcast(z_scale1), dimensions={}
       z_scale2 = f32[] parameter(4)
       z_scale2_bcast = f32[1,16,6,6] broadcast(z_scale2), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled0 = f32[1,16,6,6] multiply(conv_a, z_scale0_bcast)
       conv_a_scaled1 = f32[1,16,6,6] multiply(conv_a, z_scale1_bcast)
       conv_a_scaled_sum0 = f32[1,16,6,6] add(conv_a_scaled0, conv_a_scaled1)
       conv_a_scaled2 = f32[1,16,6,6] multiply(conv_a, z_scale2_bcast)
       conv_a_scaled_sum1 = f32[1,16,6,6] add(conv_a_scaled_sum0, conv_a_scaled2)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled_sum1, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: add(
// CHECK: multiply(
// CHECK: add(
// CHECK: clamp(
// CHECK: convert({{.*}}
// CHECK-NEXT: }
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest,
       TestConvScaledOutputMultipleUsersInGraphTripleAddF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale0 = f32[] parameter(2)
       z_scale0_bcast = f32[1,16,6,6] broadcast(z_scale0), dimensions={}
       z_scale1 = f32[] parameter(3)
       z_scale1_bcast = f32[1,16,6,6] broadcast(z_scale1), dimensions={}
       z_scale2 = f32[] parameter(4)
       z_scale2_bcast = f32[1,16,6,6] broadcast(z_scale2), dimensions={}
       z_scale3 = f32[] parameter(5)
       z_scale3_bcast = f32[1,16,6,6] broadcast(z_scale3), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_scaled0 = f32[1,16,6,6] multiply(conv_a, z_scale0_bcast)
       conv_a_scaled1 = f32[1,16,6,6] multiply(conv_a, z_scale1_bcast)
       conv_a_scaled_sum0 = f32[1,16,6,6] add(conv_a_scaled0, conv_a_scaled1)
       conv_a_scaled2 = f32[1,16,6,6] multiply(conv_a, z_scale2_bcast)
       conv_a_scaled3 = f32[1,16,6,6] multiply(conv_a, z_scale3_bcast)
       conv_a_scaled_sum1 = f32[1,16,6,6] add(conv_a_scaled2, conv_a_scaled3)
       conv_a_scaled_sum2 = f32[1,16,6,6] add(conv_a_scaled_sum0, conv_a_scaled_sum1)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled_sum2, c2_bcast)
       ROOT conv_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)

    })",
      // custom_call
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f8e4m3fn[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]], /*index=5*/[[OPERAND5:%[^ ]+]])
  )",
      // serialized_graph
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: add(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: add(
// CHECK: add(
// CHECK: clamp(
// CHECK: convert({{.*}}
// CHECK-NEXT: }
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvScaledOutputUnsupportedUserF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       z_scale = f32[] parameter(2)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_tr = f32[1,16,6,6] transpose(conv_a), dimensions={0,1,2,3}
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       conv_a_convert = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f32[1,16,6,6]) tuple(conv_a_convert, conv_a_tr)

    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f32[1,6,6,16]{3,2,1,0} fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]])
  )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution({{.*}}
// CHECK-NEXT: }
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvAddOperandReachableFromAmaxF8) {
  MAYBE_SKIP_TEST("F8");
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    apply {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT c = f32[] maximum(a, b)
    }

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       input_scale = f32[] parameter(2)
       input_scale_bcast = f32[1,16,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[1,16,6,6] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       filter_f32 = f32[3,3,128,16] convert(filter)
       conv_a = f32[1,16,6,6] convolution(input_f32, filter_f32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       conv_a_0 = f32[1,16,6,6] multiply(conv_a, input_scale_bcast)
       conv_a_1 = f32[1,16,6,6] multiply(conv_a_0, filter_scale_bcast)
       z_scale = f32[] parameter(4)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a_1, z_scale_bcast)
       abs_conv_a = f32[1,16,6,6] abs(conv_a_1)
       c0 = f32[] constant(-inf)
       amax = f32[] reduce(abs_conv_a, c0), dimensions={0,1,2,3}, to_apply=apply
       amax_bcast = f32[1,16,6,6] broadcast(amax), dimensions={}
       conv_a_scaled_amax = f32[1,16,6,6] add(conv_a_scaled, amax_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled_amax, c2_bcast)
       conv_a_clamped_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f32[]) tuple(conv_a_clamped_f8, amax)
    })",
      // fusion
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = (f32[1,6,6,16]{3,2,1,0}, f32[]) fusion([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]])
    )",
      // fusion_comp
      R"(
// CHECK: conv_fprop_fusion_comp{{.*}} {
// CHECK: convolution(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: multiply(
// CHECK: reduce(
// CHECK: ENTRY
      )");
}

TEST_F(ConvFusionRewriterTest, TestConvInt8ToInt8) {
  MAYBE_SKIP_TEST("I8");
  // max(0, clamp(conv(x, w)))); for int8_t
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = s8[] constant(0)
      zeros = s8[1,32,9,9] broadcast(zero), dimensions={}

      input = s8[1,17,9,9] parameter(0)
      filter = s8[3,3,17,32] parameter(1)

      inputs32 = s32[1,17,9,9] convert(input)
      filters32 = s32[3,3,17,32] convert(filter)

      conv = s32[1,32,9,9] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1

      lower = s32[] constant(-128)
      lowers = s32[1,32,9,9] broadcast(lower), dimensions={}
      upper = s32[] constant(127)
      uppers = s32[1,32,9,9] broadcast(upper), dimensions={}

      clamp = s32[1,32,9,9] clamp(lowers, conv, uppers)

      ROOT convert = s8[1,32,9,9] convert(clamp)
    })",
      // post_hlo
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = s8[1,9,9,32]{3,2,1,0} fusion([[fusion_2_1:%[^ ]+]], [[fusion_1_2:%[^ ]+]])
      )");
}

TEST_F(ConvFusionRewriterHloTest, TestConvInt8ToFloat) {
  MAYBE_SKIP_TEST("I8");
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s8[1,17,9,9] parameter(0)
      filter = s8[3,3,17,32] parameter(1)

      inputs32 = s32[1,17,9,9] convert(input)
      filters32 = s32[3,3,17,32] convert(filter)

      conv = s32[1,32,9,9] convolution(inputs32, filters32),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01

      ROOT convert = f32[1,32,9,9] convert(conv)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion()
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)
                     .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, TestConvInt8ToInt8BiasSideInput) {
  MAYBE_SKIP_TEST("I8");
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s32[1,17,9,9] convert(s8[1,17,9,9] parameter(0))
      filter = s32[3,3,17,32] convert(s8[3,3,17,32] parameter(1))
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      side_input = f32[1,32,9,9] convert(s8[1,32,9,9] parameter(3))

      conv = s32[1,32,9,9] convolution(input, filter),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      conv_f32 = f32[1,32,9,9] convert(conv)
      ROOT root = s8[1,32,9,9] convert(clamp(f32[1,32,9,9] broadcast(f32[] constant(-128)),
                                             add(add(conv_f32, bias), side_input),
                                             f32[1,32,9,9] broadcast(f32[] constant(127))))
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1), m::Broadcast(),
                           m::Parameter(3))
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)
                     .WithShape(S8, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, TestConvInt8ToInt8BiasSideInputWithoutClamp) {
  MAYBE_SKIP_TEST("I8");
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s32[1,17,9,9] convert(s8[1,17,9,9] parameter(0))
      filter = s32[3,3,17,32] convert(s8[3,3,17,32] parameter(1))
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      side_input = f32[1,32,9,9] convert(s8[1,32,9,9] parameter(3))

      conv = s32[1,32,9,9] convolution(input, filter),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      conv_f32 = f32[1,32,9,9] convert(conv)
      ROOT root = s8[1,32,9,9] convert(add(add(conv_f32, bias), side_input))
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1), m::Broadcast(),
                           m::Parameter(3))
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)
                     .WithShape(S8, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, TestReluAfterConvert) {
  MAYBE_SKIP_TEST("I8");
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s32[1,17,9,9] convert(s8[1,17,9,9] parameter(0))
      filter = s32[3,3,17,32] convert(s8[3,3,17,32] parameter(1))

      conv = s32[1,32,9,9] convolution(input, filter),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      conv_s8 = s8[1,32,9,9] convert(clamp(s32[1,32,9,9] broadcast(s32[] constant(-128)),
                                           conv,
                                           s32[1,32,9,9] broadcast(s32[] constant(127))))
      zeros = s8[1,32,9,9] broadcast(s8[] constant(0)), dimensions={}
      ROOT root = s8[1,32,9,9] maximum(conv_s8, zeros)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1))
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)
                     .WithShape(S8, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, TestConvInt8ToFloatBiasSideInput) {
  MAYBE_SKIP_TEST("I8");
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s8[1,17,9,9] parameter(0)
      filter = s8[3,3,17,32] parameter(1)
      bias = f32[32] parameter(2)
      bias_broadcast = f32[1,32,9,9] broadcast(bias), dimensions={1}
      side_input_f32 = f32[1,32,9,9] parameter(3)

      inputs32 = s32[1,17,9,9] convert(input)
      filters32 = s32[3,3,17,32] convert(filter)

      conv = s32[1,32,9,9] convolution(inputs32, filters32),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      conv_f32 = f32[1,32,9,9] convert(conv)
      sum1 = add(conv_f32, bias_broadcast)
      ROOT sum2 = add(sum1, side_input_f32)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1), m::Broadcast(),
                           m::Parameter(3))
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)
                     .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, FuseAlpha) {
  MAYBE_SKIP_TEST("I8");
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s8[1,17,9,9] parameter(0)
      filter = s8[3,3,17,32] parameter(1)
      inputs32 = s32[1,17,9,9] convert(input)
      filters32 = s32[3,3,17,32] convert(filter)
      alpha = f32[] constant(42)
      alpha_broadcast = f32[1,32,9,9] broadcast(alpha), dimensions={}

      conv = s32[1,32,9,9] convolution(inputs32, filters32),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      convert = f32[1,32,9,9] convert(conv)
      ROOT root = multiply(convert, alpha_broadcast)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  SCOPED_TRACE(m->ToString());
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1))
                     .WithFusionKind(HloInstruction::FusionKind::kCustom)
                     .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, FuseRelu) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[32] parameter(2)
      bias_broadcast = f32[1,32,9,9] broadcast(bias), dimensions={1}
      zero = f32[] constant(0)
      zeros = f32[1,32,9,9] broadcast(zero), dimensions={}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias_broadcast)
      ROOT relu = maximum(sum, zeros)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  SCOPED_TRACE(m->ToString());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(

                  m::Fusion(m::Parameter(0), m::Parameter(1), m::Broadcast())
                      .WithFusionKind(HloInstruction::FusionKind::kCustom)
                      .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(ConvFusionRewriterHloTest, StrengthReduceF32ToF16) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f16[1,17,9,9] parameter(0)
      filters = f16[3,3,17,32] parameter(1)
      bias = f16[32] parameter(2)
      side_input = f16[1,32,9,9] parameter(3)

      inputs_f32 = f32[1,17,9,9] convert(inputs)
      filters_f32 = f32[3,3,17,32] convert(filters)
      bias_f32 = f32[32] convert(bias)
      bias_broadcast = f32[1,32,9,9] broadcast(bias_f32), dimensions={1}
      side_input_f32 = f32[1,32,9,9] convert(side_input)
      conv = f32[1,32,9,9] convolution(inputs_f32, filters_f32),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, side_input_f32)
      sum2 = add(sum, bias_broadcast)
      ROOT conv_f16 = f16[1,32,9,9] convert(sum2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ConvKindAssignment assigner = GetConvKindAssignment();
  TF_ASSERT_OK(RunHloPass(&assigner, m.get()).status());

  ConvFusionRewriter rewriter = GetConvFusionRewriter();
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

  SCOPED_TRACE(m->ToString());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(0), m::Parameter(1),
                                   m::Parameter(3), m::Broadcast())
                             .WithShape(F16, {1, 32, 9, 9})));
}

// Disabled per b/190854862 or nvbugs/3326122.
TEST_F(ConvFusionRewriterTest, DISABLED_TestFusedConvInt8ToFloat) {
  MAYBE_SKIP_TEST("I8");
  // max(0, convert<float>(conv<int32_t>(int8_x),
  // conv<int32_t>(int8_w))+float_bias)); int8_t to float via bias.
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      bias = f32[64] parameter(2)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(convfloat, broadcasted_bias)
      ROOT relu = f32[1,3,3,64] maximum(zeros, add1)
    })",
      // post_hlo
      R"(
      ; CHECK-LABEL: ENTRY %Test (input: s8[1,3,3,64], filter: s8[3,3,64,64], bias: f32[64]) -> f32[1,3,3,64] {
      ; CHECK:  [[custom_call_0:%[^ ]+]]{{(\.[0-9])?}} = (f32[1,3,3,64]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call([[input_1:%[^ ]+]], [[copy_2:%[^ ]+]]{{(\.[0-9])?}}, [[bias_3:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config=
      ; CHECK-NEXT:  ROOT [[get_tuple_element_4:%[^ ]+]]{{(\.[0-9])?}} = f32[1,3,3,64]{3,2,1,0} get-tuple-element([[custom_call_0]]{{(\.[0-9])?}}), index=0
      )");
}

TEST_F(ConvFusionRewriterTest,
       TestFusedConvWithScaledInt8SideInputBiasInt8ToInt8) {
  MAYBE_SKIP_TEST("I8");
  // clamp(max(0, alpha_conv * conv(x, w) + alpha_side *
  // convert<int32_t>(int8_side_input) + bias)); for int8_t
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)
      alpha_conv = f32[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = f32[] constant(0.899994934)
      alpha_side_input = f32[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      side_input = s8[1,3,3,64] parameter(2)
      bias = f32[64] parameter(3)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)
      side_input_f32 = f32[1,3,3,64] convert(side_input)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      scaled_conv = f32[1,3,3,64] multiply(convfloat, alpha_conv)
      scaled_side_input = f32[1,3,3,64] multiply(side_input_f32, alpha_side_input)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = f32[1,3,3,64] add(add1, scaled_side_input)
      relu = f32[1,3,3,64] maximum(zeros, add2)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)

      ROOT convert = s8[1,3,3,64] convert(clamp)
    })",
      // post_hlo
      R"(
// CHECK: [[conv_fusion:%[^ ]+]] = s8[1,3,3,64]{3,2,1,0} fusion([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]], [[side_input_4:%[^ ]+]])
      )");
}

TEST_F(ConvFusionRewriterTest,
       TestFusedConvWithScaledFloatSideInputBiasInt8ToInt8) {
  MAYBE_SKIP_TEST("I8");
  // From:
  // convert<int8_t>(clamp(max(0, alpha_conv * conv(x, w) + alpha_side *
  // float_side_input + bias))); To: convert<int8_t>(clamp(conv(int8_x, int8_w,
  // float_alpha_side, float_side_input, float_bias)));
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)
      alpha_conv = f32[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = f32[] constant(0.899994934)
      alpha_side_input = f32[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      side_input = f32[1,3,3,64] parameter(2)
      bias = f32[64] parameter(3)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      scaled_conv = f32[1,3,3,64] multiply(convfloat, alpha_conv)
      scaled_side_input = f32[1,3,3,64] multiply(side_input, alpha_side_input)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = f32[1,3,3,64] add(add1, scaled_side_input)
      relu = f32[1,3,3,64] maximum(zeros, add2)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)

      ROOT convert = s8[1,3,3,64] convert(clamp)
    })",
      //  post_hlo
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = s8[1,3,3,64]{3,2,1,0} fusion([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]], [[side_input_4:%[^ ]+]])
      )");
}

TEST_F(ConvFusionRewriterTest,
       TestFusedConvWithScaledInt8SideInputBiasInt8ToFloat) {
  MAYBE_SKIP_TEST("I8");
  // From:
  // clamp(max(0, alpha_conv * conv(x, w) + alpha_side *
  // convert<float>(int8_side_input) + bias)); To: clamp(conv(int8_x, int8_w,
  // float_alpha_side, convert<float>(int8_side_input), float_bias));
  TestClamp(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,3,3,64] broadcast(zero), dimensions={}
      alpha_conv_scalar = f32[] constant(0.999994934)
      alpha_conv = f32[1,3,3,64] broadcast(alpha_conv_scalar), dimensions={}
      alpha_side_input_scalar = f32[] constant(0.899994934)
      alpha_side_input = f32[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = s8[1,3,3,64] parameter(0)
      filter = s8[3,3,64,64] parameter(1)
      side_input = s8[1,3,3,64] parameter(2)
      bias = f32[64] parameter(3)

      inputs32 = s32[1,3,3,64] convert(input)
      filters32 = s32[3,3,64,64] convert(filter)
      side_input_f32 = f32[1,3,3,64] convert(side_input)

      conv = s32[1,3,3,64] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=1

      convfloat = f32[1,3,3,64] convert(conv)
      scaled_conv = f32[1,3,3,64] multiply(convfloat, alpha_conv)
      scaled_side_input = f32[1,3,3,64] multiply(side_input_f32, alpha_side_input)
      broadcasted_bias = f32[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = f32[1,3,3,64] add(scaled_conv, broadcasted_bias)
      add2 = f32[1,3,3,64] add(add1, scaled_side_input)
      relu = f32[1,3,3,64] maximum(zeros, add2)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      ROOT clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)
    })",
      // post_hlo
      R"(
// CHECK: [[cudnn_fusion:%[^ ]+]] = f32[1,3,3,64]{3,2,1,0} fusion([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]], [[fusion_1_4:%[^ ]+]])
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
