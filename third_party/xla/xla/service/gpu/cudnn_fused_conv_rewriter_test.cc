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

#include "xla/service/gpu/cudnn_fused_conv_rewriter.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

#include "xla/service/algebraic_simplifier.h"
#include "xla/service/convert_mover.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/service/reshape_mover.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

// TODO(b/210165681): The tests in this file are fragile to HLO op names.

namespace m = match;

using ::testing::HasSubstr;
using ::testing::Not;

class CudnnFusedConvRewriterHloTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  CudnnFusedConvRewriterHloTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false,
                    /*instruction_can_change_layout_func=*/{}) {}
};

class CudnnFusedConvRewriterTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
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
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      const std::string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      EXPECT_THAT(optimized_hlo_string,
                  Not(HasSubstr(kCudnnConvForwardCallTarget)))
          << optimized_hlo_string;
      EXPECT_THAT(optimized_hlo_string,
                  HasSubstr(kCudnnConvBiasActivationForwardCallTarget));

      TF_ASSERT_OK_AND_ASSIGN(auto module,
                              ParseAndReturnVerifiedModule(hlo_with_new_type));
      DebugOptions debug_opts = module->config().debug_options();
      debug_opts.set_xla_gpu_use_runtime_fusion(true);
      module->mutable_config().set_debug_options(debug_opts);
      EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01}))
          << optimized_hlo_string;
    }
  }

  void TestClamp(absl::string_view pre_hlo_string,
                 absl::string_view post_hlo_string) {
    std::string alpha_conv_scalar, alpha_side_input_scalar;
    std::string elementwise_type;

    std::string optimized_hlo_string = GetOptimizedHlo(pre_hlo_string);
    EXPECT_THAT(optimized_hlo_string, Not(HasSubstr("Convert")));
    EXPECT_THAT(optimized_hlo_string, HasSubstr("__cudnn$conv"));
    EXPECT_TRUE(RunAndCompare(pre_hlo_string, ErrorSpec{0.01}))
        << pre_hlo_string;

    absl::StatusOr<bool> filecheck_result =
        RunFileCheck(optimized_hlo_string, post_hlo_string);
    ASSERT_TRUE(filecheck_result.ok()) << filecheck_result.status();
    EXPECT_TRUE(*filecheck_result);
  }

  void TestNotMatchWithAllTypes(absl::string_view hlo_string) {
    for (absl::string_view type : {"f16", "f32", "f64"}) {
      const std::string hlo_with_new_type =
          absl::StrReplaceAll(hlo_string, {{"TYPE", type}});
      std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
      SCOPED_TRACE(optimized_hlo_string);
      EXPECT_THAT(optimized_hlo_string, HasSubstr(kCudnnConvForwardCallTarget));
      EXPECT_THAT(optimized_hlo_string,
                  Not(HasSubstr(kCudnnConvBiasActivationForwardCallTarget)));
    }
  }

  void TestF8(std::string pre_hlo_string, std::string custom_call_string,
              std::string serialized_graph_string) {
    if (GetCudaComputeCapability().IsAtLeast(
            se::CudaComputeCapability::HOPPER)) {
      // On Hopper and newer architectures, test numerical correctness and
      // verify the HLO of the Custom Call with operand and return layouts and
      // the serialized graph based on the full compiler pipeline.
      std::string optimized_hlo_string = GetOptimizedHlo(pre_hlo_string);
      EXPECT_THAT(optimized_hlo_string, Not(HasSubstr("Convert")));
      EXPECT_THAT(optimized_hlo_string, HasSubstr("__cudnn$conv"));
      EXPECT_TRUE(RunAndCompare(pre_hlo_string, ErrorSpec{0.15, 0.15}))
          << pre_hlo_string;

      absl::StatusOr<bool> filecheck_result =
          RunFileCheck(optimized_hlo_string, custom_call_string);
      ASSERT_TRUE(filecheck_result.ok()) << filecheck_result.status();
      EXPECT_TRUE(*filecheck_result);

      filecheck_result =
          RunFileCheck(optimized_hlo_string, serialized_graph_string);
      ASSERT_TRUE(filecheck_result.ok()) << filecheck_result.status();
      EXPECT_TRUE(*filecheck_result);
    } else {
      // On older architectures, disregard layout information and only verify
      // the basic configuration of the convolution Custom Call using the number
      // of operands and the window_size and serialized graph attributes based
      // on the GpuConvRewriter and CudnnFusedConvRewriter passes.
      std::string::size_type p0 = custom_call_string.find(':');
      std::string::size_type p1 = custom_call_string.find("custom-call");
      custom_call_string.erase(p0 + 1, p1 - p0 - 2);
      p0 = custom_call_string.find(", dim_labels");
      custom_call_string.erase(p0);

      TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                              ParseAndReturnVerifiedModule(pre_hlo_string));
      TF_ASSERT_OK_AND_ASSIGN(bool changed,
                              RunHloPass(GpuConvRewriter(), module.get()));
      EXPECT_TRUE(changed);
      RunAndFilecheckHloRewrite(
          module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
          CudnnFusedConvRewriter(
              se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0}),
          custom_call_string);
      RunAndFilecheckHloRewrite(
          module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
          CudnnFusedConvRewriter(
              se::CudaComputeCapability{se::CudaComputeCapability::HOPPER, 0}),
          serialized_graph_string);
    }
  }

  void TestF8Parameterized(std::string template_pre_hlo_string,
                           std::string template_custom_call_string,
                           std::string template_serialized_graph_string) {
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
          TestF8(absl::StrReplaceAll(template_pre_hlo_string, replacements),
                 absl::StrReplaceAll(template_custom_call_string, replacements),
                 absl::StrReplaceAll(template_serialized_graph_string,
                                     replacements));
        }
      }
    }
  }
};

TEST_F(CudnnFusedConvRewriterTest, TestConvOnly) {
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

TEST_F(CudnnFusedConvRewriterTest, DontFuseReluWithDepthwiseConv) {
  // max(0, conv(x, w));
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,17,9,9] broadcast(zero), dimensions={}

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,1,17] parameter(1)

      conv = TYPE[1,17,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=17
      ROOT relu = TYPE[1,17,9,9] maximum(zeros, conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestBias) {
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

TEST_F(CudnnFusedConvRewriterTest, DontFuseBiasWithDepthwiseConv) {
  // conv(x, w) + bias;
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,1,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=64
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      add1 = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestElu) {
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

TEST_F(CudnnFusedConvRewriterTest, DontFuseEluWithDepthwiseConv) {
  // sum = conv(x, w) + bias
  // select(compare(sum, 0, GT), sum, exponential-minus-one(sum));
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,1,64] parameter(1)
      bias = TYPE[64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=64
      broadcasted_bias = TYPE[1,3,3,64] broadcast(bias), dimensions={3}
      sum = TYPE[1,3,3,64] add(conv, broadcasted_bias)
      cmp = pred[1,3,3,64] compare(sum, zeros), direction=GT
      expm1 = TYPE[1,3,3,64] exponential-minus-one(sum)
      ROOT elu = TYPE[1,3,3,64] select(cmp, sum, expm1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestRelu6) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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
TEST_F(CudnnFusedConvRewriterTest, TestRelu6OddChannels) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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

TEST_F(CudnnFusedConvRewriterTest, TestLeakyRelu) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
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

TEST_F(CudnnFusedConvRewriterTest, TestSideInputOnly) {
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

TEST_F(CudnnFusedConvRewriterTest, DontFuseSideInputWithDepthwiseConv) {
  // max(0, conv(x, w) + side_input);
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,1,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=64
      add1 = TYPE[1,3,3,64] add(conv, side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestBiasAndSideInput) {
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

TEST_F(CudnnFusedConvRewriterTest, TestScaledConv) {
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

TEST_F(CudnnFusedConvRewriterTest, DontFuseScaledDepthwiseConv) {
  // max(0, 0.999994934 * conv(x, w));
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,17,9,9] broadcast(zero), dimensions={}
      alpha_conv_scalar = TYPE[] constant(0.999994934)

      input = TYPE[1,17,9,9] parameter(0)
      filter = TYPE[3,3,1,17] parameter(1)

      conv = TYPE[1,17,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=17
      alpha_conv = TYPE[1,17,9,9] broadcast(alpha_conv_scalar), dimensions={}
      scaled_conv = TYPE[1,17,9,9] multiply(conv, alpha_conv)
      ROOT relu = TYPE[1,17,9,9] maximum(zeros, scaled_conv)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestNoCrashOnInf) {
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

TEST_F(CudnnFusedConvRewriterTest, TestConvAndScaledSideInput) {
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

TEST_F(CudnnFusedConvRewriterTest, DontFuseDepthwiseConvWithScaledSideInput) {
  // max(0, conv(x, w) + 0.899994934 * side_input);
  TestNotMatchWithAllTypes(R"(
    HloModule Test

    ENTRY Test {
      zero = TYPE[] constant(0)
      zeros = TYPE[1,3,3,64] broadcast(zero), dimensions={}
      alpha_side_input_scalar = TYPE[] constant(0.899994934)
      alpha_side_input = TYPE[1,3,3,64] broadcast(alpha_side_input_scalar), dimensions={}

      input = TYPE[1,3,3,64] parameter(0)
      filter = TYPE[3,3,1,64] parameter(1)
      side_input = TYPE[1,3,3,64] parameter(2)

      conv = TYPE[1,3,3,64] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, feature_group_count=64
      scaled_side_input = TYPE[1,3,3,64] multiply(side_input, alpha_side_input)
      add1 = TYPE[1,3,3,64] add(conv, scaled_side_input)
      ROOT relu = TYPE[1,3,3,64] maximum(zeros, add1)
    })");
}

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndScaledSideInput) {
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

TEST_F(CudnnFusedConvRewriterTest, TestScaledConvAndScaledSideInputWithBias) {
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

TEST_F(CudnnFusedConvRewriterTest, TestMatchMaxZeroOnly) {
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

TEST_F(CudnnFusedConvRewriterTest, PreservesMetadata) {
  const char* kHloString = R"(
    HloModule Test

    ENTRY Test {
      zero = f32[] constant(0)
      zeros = f32[1,32,9,9] broadcast(zero), dimensions={}

      input = f32[1,17,9,9] parameter(0)
      filter = f32[3,3,17,32] parameter(1)

      conv = f32[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1, metadata={op_type="foo" op_name="bar"}
      ROOT relu = f32[1,32,9,9] maximum(zeros, conv)
    })";

  const std::string optimized_hlo_string =
      backend()
          .compiler()
          ->RunHloPasses(
              ParseAndReturnVerifiedModule(kHloString, GetModuleConfigForTest())
                  .value(),
              backend().default_stream_executor(), backend().memory_allocator())
          .value()
          ->ToString();
  EXPECT_THAT(optimized_hlo_string,
              ::testing::ContainsRegex(
                  R"(custom-call.*metadata=\{op_type="foo" op_name="bar"\})"));
}

TEST_F(CudnnFusedConvRewriterTest, TestPreservesFeatureGroupCount) {
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

TEST_F(CudnnFusedConvRewriterTest, TestConvF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
  TestF8(
      // pre_hlo
      R"(
    HloModule Test

    ENTRY Test {
       input = f8e4m3fn[1,128,6,6] parameter(0)
       filter = f8e4m3fn[3,3,128,16] parameter(1)
       ROOT conv_a = f8e4m3fn[1,16,6,6] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1

    })",
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f8e4m3fn]conv();"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvScaledOutputF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[SCALE_UID:[0-9]+]]:[f8e4m3fn]scale([[CONV_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvInvscaledOutputF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[SCALE0_UID:[0-9]+]]:[f8e4m3fn]invscale([[CONV_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvScaledF8Parameterized) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (<<OutputType>>[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
      )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[SCALE0_UID:[0-9]+]]:[f32]scale([[CONV_UID]]);[[SCALE1_UID:[0-9]+]]:[f32]scale([[SCALE0_UID]]);[[SCALE2_UID:[0-9]+]]:[<<OutputType>>]scale([[SCALE1_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvScaledBiasF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]], /*index=5*/[[OPERAND5:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
      )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[SCALE0_UID:[0-9]+]]:[f32]scale([[CONV_UID]]);[[SCALE1_UID:[0-9]+]]:[f32]scale([[SCALE0_UID]]);[[ADD_UID:[0-9]+]]:[f32]add([[SCALE1_UID]]);[[SCALE2_UID:[0-9]+]]:[f8e4m3fn]scale([[ADD_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvScaledReluF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[RELU_UID:[0-9]+]]:[f32]relu([[CONV_UID]]);[[SCALE0_UID:[0-9]+]]:[f8e4m3fn]scale([[RELU_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvAmaxF8) {
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
       input_scale_bcast = f32[1,128,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[3,3,128,16] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       input_unscaled = f32[1,128,6,6] multiply(input_f32, input_scale_bcast)
       filter_f32 = f32[3,3,128,16] convert(filter)
       filter_unscaled = f32[3,3,128,16] multiply(filter_f32, filter_scale_bcast)
       conv_a = f32[1,16,6,6] convolution(input_unscaled, filter_unscaled), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       z_scale = f32[] parameter(4)
       z_scale_bcast = f32[1,16,6,6] broadcast(z_scale), dimensions={}
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       conv_a_clamped_f8 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
       abs_conv_a = f32[1,16,6,6] abs(conv_a)
       c0 = f32[] constant(-inf)
       amax = f32[] reduce(abs_conv_a, c0), dimensions={0,1,2,3}, to_apply=apply
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f32[]) tuple(conv_a_clamped_f8, amax)

    })",
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, f32[], u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
    )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[SCALE0_UID:[0-9]+]]:[f32]scale([[CONV_UID]]);[[SCALE1_UID:[0-9]+]]:[f32]scale([[SCALE0_UID]]);[[SCALE2_UID:[0-9]+]]:[f8e4m3fn]scale([[SCALE1_UID]]);[[AMAX_UID:[0-9]+]]:[f32]amax([[SCALE1_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvReluAmaxF8) {
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
       input_scale_bcast = f32[1,128,6,6] broadcast(input_scale), dimensions={}
       filter_scale = f32[] parameter(3)
       filter_scale_bcast = f32[3,3,128,16] broadcast(filter_scale), dimensions={}
       input_f32 = f32[1,128,6,6] convert(input)
       input_unscaled = f32[1,128,6,6] multiply(input_f32, input_scale_bcast)
       filter_f32 = f32[3,3,128,16] convert(filter)
       filter_unscaled = f32[3,3,128,16] multiply(filter_f32, filter_scale_bcast)
       c = f32[] constant(0)
       c_bcast = f32[1,16,6,6] broadcast(c), dimensions={}
       conv_a = f32[1,16,6,6] convolution(input_unscaled, filter_unscaled), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
       relu_a = f32[1,16,6,6] maximum(conv_a, c_bcast)
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
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f8e4m3fn[1,6,6,16]{3,2,1,0}, f32[], u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]], [[OPERAND2:%[^ ]+]], [[OPERAND3:%[^ ]+]], [[OPERAND4:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
    )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();[[SCALE0_UID:[0-9]+]]:[f32]scale([[CONV_UID]]);[[SCALE1_UID:[0-9]+]]:[f32]scale([[SCALE0_UID]]);[[RELU_UID:[0-9]+]]:[f32]relu([[SCALE1_UID]]);[[SCALE2_UID:[0-9]+]]:[f8e4m3fn]scale([[RELU_UID]]);[[AMAX_UID:[0-9]+]]:[f32]amax([[RELU_UID]]);"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvScaledOutputMultipleUsersF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped0 = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled0, c2_bcast)
       conv_a_clamped1 = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled1, c2_bcast)
       conv_a_convert0 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped0)
       conv_a_convert1 = f8e4m3fn[1,16,6,6] convert(conv_a_clamped1)
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f8e4m3fn[1,16,6,6]) tuple(conv_a_convert0, conv_a_convert1)

    })",
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f32[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvScaledOutputUnsupportedUserF8) {
#if (CUDA_VERSION < 12000 || CUDNN_VERSION < 8900)
  GTEST_SKIP() << "FP8 convolutions require CUDA 12 and cuDNN 8.9.";
#endif
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
       conv_a_cos = f32[1,16,6,6] cosine(conv_a)
       conv_a_scaled = f32[1,16,6,6] multiply(conv_a, z_scale_bcast)
       c1 = f32[] constant(-448.)
       c1_bcast = f32[1,16,6,6] broadcast(c1), dimensions={}
       c2 = f32[] constant(448.)
       c2_bcast = f32[1,16,6,6] broadcast(c2), dimensions={}
       conv_a_clamped = f32[1,16,6,6] clamp(c1_bcast, conv_a_scaled, c2_bcast)
       conv_a_convert = f8e4m3fn[1,16,6,6] convert(conv_a_clamped)
       ROOT conv_f8 = (f8e4m3fn[1,16,6,6], f32[1,16,6,6]) tuple(conv_a_convert, conv_a_cos)

    })",
      // custom_call
      R"(
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (f32[1,6,6,16]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[OPERAND0:%[^ ]+]], [[OPERAND1:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForwardGraph"
  )",
      // serialized_graph
      R"(
// CHECK: "serialized_graph":"[[CONV_UID:[0-9]+]]:[f32]conv();"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvInt8ToInt8) {
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
// CHECK: [[cudnn_conv_4_0:%[^ ]+]] = (s8[1,9,9,32]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[fusion_2_1:%[^ ]+]], [[fusion_1_2:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
      )");
}

TEST_F(CudnnFusedConvRewriterHloTest, TestConvInt8ToFloat) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                             m::CustomCall({kCudnnConvForwardCallTarget}), 0)
                             .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(CudnnFusedConvRewriterHloTest, TestConvInt8ToInt8BiasSideInput) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall({kCudnnConvBiasActivationForwardCallTarget},
                                   m::Parameter(0), m::Parameter(1),
                                   m::Parameter(2), m::Parameter(3)),
                     0)
                     .WithShape(S8, {1, 32, 9, 9})));
}

TEST_F(CudnnFusedConvRewriterHloTest, TestReluAfterConvert) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(
                  &conv, {kCudnnConvBiasActivationForwardCallTarget},
                  m::Parameter(0),  //
                  m::Parameter(1),  //
                  m::Broadcast(
                      m::ConstantEffectiveScalar(0).WithElementType(F32))),
              0)
              .WithShape(S8, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kRelu);
}

TEST_F(CudnnFusedConvRewriterHloTest, TestConvInt8ToFloatBiasSideInput) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall({kCudnnConvBiasActivationForwardCallTarget},
                                   m::Parameter(0), m::Parameter(1),
                                   m::Parameter(2), m::Parameter(3)),
                     0)
                     .WithShape(F32, {1, 32, 9, 9})));
}

// The ReshapeMover pass changes
//   reshape(side_input) * alpha -->
//   reshape(side_input * alpha).
// Make sure we can pattern-match this.
TEST_F(CudnnFusedConvRewriterHloTest, Int8SideInputWithScaleAndReshape) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = s32[1,17,9,9] convert(s8[1,17,9,9] parameter(0))
      filter = s32[3,3,17,32] convert(s8[3,3,17,32] parameter(1))
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      side_input_scale = f32[2592] broadcast(f32[] constant(0.25)), dimensions={}
      side_input = f32[1,32,9,9] reshape(multiply(f32[2592] convert(s8[2592] parameter(3)), side_input_scale))

      conv = s32[1,32,9,9] convolution(input, filter),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = s8[1,32,9,9] convert(clamp(f32[1,32,9,9] broadcast(f32[] constant(-128)),
                                             add(add(f32[1,32,9,9] convert(conv), bias), side_input),
                                             f32[1,32,9,9] broadcast(f32[] constant(127))))
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  HloPassFix<HloPassPipeline> simplify("simplify");
  simplify.AddPass<AlgebraicSimplifier>(AlgebraicSimplifierOptions{});
  simplify.AddPass<ReshapeMover>();
  simplify.AddPass<ConvertMover>();
  TF_ASSERT_OK(RunHloPass(&simplify, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(
                  &conv, {kCudnnConvBiasActivationForwardCallTarget},
                  m::Parameter(0),  //
                  m::Parameter(1),  //
                  m::Parameter(2),  //
                  m::Reshape(m::Parameter(3)).WithShape(S8, {1, 32, 9, 9})),
              0)
              .WithShape(S8, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 1);
  EXPECT_EQ(config.side_input_scale(), 0.25);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseAlpha) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget}),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 42);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseRelu) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2)),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kRelu);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseReluIfMultipleUses) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      zeros = f32[1,32,9,9] broadcast(f32[] constant(0)), dimensions={}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias)
      relu = maximum(sum, zeros)
      not_relu = minimum(sum, zeros)
      ROOT root = tuple(relu, not_relu)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::MaximumAnyOrder(
              m::Broadcast(m::ConstantEffectiveScalar(0)),
              m::GetTupleElement(
                  m::CustomCall(
                      &conv, {kCudnnConvBiasActivationForwardCallTarget},
                      m::Parameter(0), m::Parameter(1), m::Parameter(2)),
                  0)
                  .WithShape(F32, {1, 32, 9, 9})),
          m::Minimum())));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseElu) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f16[1,16,9,9] parameter(0)
      filters = f16[3,3,16,32] parameter(1)
      bias = f16[32] parameter(2)
      bias_broadcast = f16[1,32,9,9] broadcast(bias), dimensions={1}
      zero = f16[] constant(0)
      zeros = f16[1,32,9,9] broadcast(zero), dimensions={}
      conv = f16[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias_broadcast)
      cmp = compare(sum, zeros), direction=GT
      expm1 = exponential-minus-one(sum)
      ROOT elu = select(cmp, sum, expm1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  DebugOptions debug_opts = m->config().debug_options();
  debug_opts.set_xla_gpu_use_runtime_fusion(true);
  m->mutable_config().set_debug_options(debug_opts);

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  // elu fusion is only active on Ampere+.
  CudnnFusedConvRewriter fuser{se::CudaComputeCapability(8, 0)};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2)),
              0)
              .WithShape(F16, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kElu);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseEluIfMultipleUses) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f16[1,16,9,9] parameter(0)
      filters = f16[3,3,16,32] parameter(1)
      bias = f16[32] parameter(2)
      bias_broadcast = f16[1,32,9,9] broadcast(bias), dimensions={1}
      zero = f16[] constant(0)
      zeros = f16[1,32,9,9] broadcast(zero), dimensions={}
      conv = f16[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias_broadcast)
      cmp = compare(sum, zeros), direction=GT
      expm1 = exponential-minus-one(sum)
      elu = select(cmp, sum, expm1)
      not_elu = minimum(sum, zeros)
      ROOT root = tuple(elu, not_elu)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  DebugOptions debug_opts = m->config().debug_options();
  debug_opts.set_xla_gpu_use_runtime_fusion(true);
  m->mutable_config().set_debug_options(debug_opts);

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  auto gte_pattern =
      m::GetTupleElement(
          m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                        m::Parameter(0), m::Parameter(1), m::Parameter(2)),
          0)
          .WithShape(F16, {1, 32, 9, 9});
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Select(m::Compare(gte_pattern,
                               m::Broadcast(m::ConstantEffectiveScalar(0)))
                        .WithComparisonDirection(ComparisonDirection::kGt),
                    gte_pattern,
                    m::Op()
                        .WithPredicate(HloPredicateIsOp<HloOpcode::kExpm1>)
                        .WithOperand(0, gte_pattern)),
          m::Minimum())));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseRelu6) {
  const std::string module_str = R"(
    HloModule Test
    ENTRY Test {
      inputs = f16[1,18,9,9] parameter(0)
      filters = f16[3,3,18,32] parameter(1)
      bias = f16[32] parameter(2)
      bias_broadcast = f16[1,32,9,9] broadcast(bias), dimensions={1}
      zero = f16[] constant(0)
      zeros = f16[1,32,9,9] broadcast(zero), dimensions={}
      sixes = f16[1,32,9,9] broadcast(f16[] constant(6)), dimensions={}
      conv = f16[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias_broadcast)
      ROOT relu = clamp(zeros, sum, sixes)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  DebugOptions debug_opts = m->config().debug_options();
  debug_opts.set_xla_gpu_use_runtime_fusion(true);
  m->mutable_config().set_debug_options(debug_opts);

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  // relu6 fusion is only enabled on Ampere+.
  CudnnFusedConvRewriter fuser{se::CudaComputeCapability(8, 0)};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());
  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2)),
              0)
              .WithShape(F16, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kRelu6);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseRelu6IfMultipleUses) {
  const std::string module_str = R"(
    HloModule Test
    ENTRY Test {
      inputs = f16[1,18,9,9] parameter(0)
      filters = f16[3,3,18,32] parameter(1)
      bias = f16[1,32,9,9] broadcast(f16[32] parameter(2)), dimensions={1}
      zeros = f16[1,32,9,9] broadcast(f16[] constant(0)), dimensions={}
      sixes = f16[1,32,9,9] broadcast(f16[] constant(6)), dimensions={}
      conv = f16[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias)
      relu = clamp(zeros, sum, sixes)
      not_relu = minimum(sum, zeros)
      ROOT root = tuple(relu, not_relu)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  DebugOptions debug_opts = m->config().debug_options();
  debug_opts.set_xla_gpu_use_runtime_fusion(true);
  m->mutable_config().set_debug_options(debug_opts);

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Clamp(m::Broadcast(m::ConstantEffectiveScalar(0)),
                   m::GetTupleElement(
                       m::CustomCall(
                           &conv, {kCudnnConvBiasActivationForwardCallTarget},
                           m::Parameter(0), m::Parameter(1), m::Parameter(2)),
                       0)
                       .WithShape(F16, {1, 32, 9, 9}),
                   m::Broadcast(m::ConstantEffectiveScalar(6))),
          m::Minimum())));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseLeakyRelu) {
  const std::string module_str = R"(
    HloModule Test
    ENTRY Test {
      inputs = f16[1,16,9,9] parameter(0)
      filters = f16[3,3,16,32] parameter(1)
      bias = f16[1,32,9,9] broadcast(f16[32] parameter(2)), dimensions={1}
      zeros = f16[1,32,9,9] broadcast(f16[] constant(0)), dimensions={}
      alphas = f16[1,32,9,9] broadcast(f16[] constant(0.2)), dimensions={}
      conv = f16[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias)
      cmp = compare(sum, zeros), direction=GT
      mul = multiply(sum, alphas)
      ROOT leaky_relu = select(cmp, sum, mul)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  DebugOptions debug_opts = m->config().debug_options();
  debug_opts.set_xla_gpu_use_runtime_fusion(true);
  m->mutable_config().set_debug_options(debug_opts);

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  // Leaky-relu fusion is only enabled on Ampere+.
  CudnnFusedConvRewriter fuser{se::CudaComputeCapability(8, 0)};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2)),
              0)
              .WithShape(F16, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kLeakyRelu);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseLeakyReluIfMultipleUses) {
  const std::string module_str = R"(
    HloModule Test
    ENTRY Test {
      inputs = f16[1,16,9,9] parameter(0)
      filters = f16[3,3,16,32] parameter(1)
      bias = f16[1,32,9,9] broadcast(f16[32] parameter(2)), dimensions={1}
      zeros = f16[1,32,9,9] broadcast(f16[] constant(0)), dimensions={}
      alphas = f16[1,32,9,9] broadcast(f16[] constant(0.2)), dimensions={}
      conv = f16[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, bias)
      cmp = compare(sum, zeros), direction=GT
      mul = multiply(sum, alphas)
      leaky_relu = select(cmp, sum, mul)
      not_leaky_relu = minimum(sum, zeros)
      ROOT root = tuple(leaky_relu, not_leaky_relu)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  DebugOptions debug_opts = m->config().debug_options();
  debug_opts.set_xla_gpu_use_runtime_fusion(true);
  m->mutable_config().set_debug_options(debug_opts);

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  auto gte_pattern =
      m::GetTupleElement(
          m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                        m::Parameter(0), m::Parameter(1), m::Parameter(2)),
          0)
          .WithShape(F16, {1, 32, 9, 9});
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Select(m::Compare(gte_pattern,
                               m::Broadcast(m::ConstantEffectiveScalar(0)))
                        .WithComparisonDirection(ComparisonDirection::kGt)
                        .WithOneUse(),
                    gte_pattern,
                    m::Multiply(gte_pattern,
                                m::Broadcast(m::ConstantEffectiveScalar()))),
          m::Minimum())));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseAlphaIfMultipleUsers) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      alpha = f32[1,32,9,9] broadcast(f32[] parameter(3)), dimensions={}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(multiply(alpha, conv), bias)
      ROOT root = tuple(conv, sum)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv1;
  const HloInstruction* conv2;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&conv1), 0),
          m::AddAnyOrder(m::Broadcast(m::Parameter(2)),
                         m::MultiplyAnyOrder(
                             m::Broadcast(m::Parameter(3)),
                             m::GetTupleElement(m::CustomCall(&conv2), 0))))));
  EXPECT_EQ(conv1, conv2);
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv1->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 1);
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseBiasIfMultipleUsers) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = tuple(conv, add(conv, bias))
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv1;
  const HloInstruction* conv2;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&conv1), 0),
          m::AddAnyOrder(m::Broadcast(m::Parameter(2)),
                         m::GetTupleElement(m::CustomCall(&conv2), 0)))));
  EXPECT_EQ(conv1, conv2);
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv1->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 1);
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseSideInputThroughRelu) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      side_input = f32[1,32,9,9] parameter(2)
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      relu = maximum(conv, f32[1,32,9,9] broadcast(f32[] constant(0)))
      ROOT root = add(relu, side_input)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::AddAnyOrder(
          m::Parameter(2),
          m::GetTupleElement(
              m::CustomCall(&conv, m::Parameter(0), m::Parameter(1),
                            m::Broadcast(m::ConstantEffectiveScalar(0))),
              0))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 1);
  EXPECT_EQ(config.activation_mode(), se::dnn::kRelu);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseBiasThroughRelu) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[1,32,9,9] broadcast(f32[32] parameter(2)), dimensions={1}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      relu = maximum(conv, f32[1,32,9,9] broadcast(f32[] constant(0)))
      ROOT root = add(relu, bias)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::AddAnyOrder(
                  m::Broadcast(m::Parameter(2)),
                  m::GetTupleElement(m::CustomCall(
                      &conv, m::Parameter(0), m::Parameter(1),
                      m::Broadcast(m::ConstantEffectiveScalar(0)))))));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 1);
  EXPECT_EQ(config.activation_mode(), se::dnn::kRelu);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseSideInputIfMultipleUsers) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      side_input = f32[1,32,9,9] parameter(2)
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = tuple(conv, add(conv, side_input))
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv1;
  const HloInstruction* conv2;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&conv1), 0),
          m::AddAnyOrder(m::Parameter(2),
                         m::GetTupleElement(m::CustomCall(&conv2), 0)))));
  EXPECT_EQ(conv1, conv2);
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv1->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.conv_result_scale(), 1);
  EXPECT_EQ(config.activation_mode(), se::dnn::kNone);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseConvertToF16IfMultipleUsers) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] convert(f16[1,17,9,9] parameter(0))
      filters = f32[3,3,17,32] convert(f16[3,3,17,32] parameter(1))
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = tuple(conv, f16[1,32,9,9] convert(conv))
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv1;
  const HloInstruction* conv2;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::GetTupleElement(m::CustomCall(&conv1), 0),
                  m::Convert(m::GetTupleElement(m::CustomCall(&conv2), 0)))));
  EXPECT_EQ(conv1, conv2);
}

TEST_F(CudnnFusedConvRewriterHloTest, DontFuseToS8IfMultipleUsers) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] convert(s8[1,17,9,9] parameter(0))
      filters = f32[3,3,17,32] convert(s8[3,3,17,32] parameter(1))
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      conv_s8 = s8[1,32,9,9] convert(clamp(
                  f32[1,32,9,9] broadcast(f32[] constant(-128)),
                  conv,
                  f32[1,32,9,9] broadcast(f32[] constant(127))))
      ROOT root = tuple(conv, conv_s8)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv1;
  const HloInstruction* conv2;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::CustomCall(&conv1), 0),
          m::Convert(m::Clamp(m::Op(),  //
                              m::GetTupleElement(m::CustomCall(&conv2), 0),
                              m::Op())))));
  EXPECT_EQ(conv1, conv2);
}

TEST_F(CudnnFusedConvRewriterHloTest, RemoveConvertByFusingS32ToF32) {
  const std::string_view module_str = R"(
    HloModule Test

    ENTRY test_entry {
      inputs = s8[1, 17, 9, 9] parameter(0)
      filters = s8[3, 3, 17, 32] parameter(1)
      mult_op  = f32[1, 32, 9, 9] parameter(2)
      conv = s32[1, 32, 9, 9] convolution(inputs, filters), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01
      ROOT ret = multiply(f32[1, 32, 9, 9] convert(conv), mult_op)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());
  SCOPED_TRACE(m->ToString());
  HloInstruction* conv1 = nullptr;
  // Checks that it removed the Convert inside multiply around conv.
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::GetTupleElement(m::CustomCall(&conv1)),
                                     m::Parameter(2))));
}

TEST_F(CudnnFusedConvRewriterHloTest, RemoveConvertByFusingS8ToF32) {
  const std::string_view module_str = R"(
    HloModule Test

    ENTRY test_entry {
      inputs = s8[1, 17, 9, 9] parameter(0)
      filters = s8[3, 3, 17, 32] parameter(1)
      mult_op  = f32[1, 32, 9, 9] parameter(2)
      conv = convolution(inputs, filters), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01
      ROOT ret = multiply(f32[1, 32, 9, 9] convert(conv), mult_op)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());
  SCOPED_TRACE(m->ToString());
  HloInstruction* conv1 = nullptr;
  // Checks that it removed the Convert inside multiply around conv.
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::GetTupleElement(m::CustomCall(&conv1)),
                                     m::Parameter(2))));
}

TEST_F(CudnnFusedConvRewriterHloTest, RemoveConvertByFusingF32ToS8) {
  const std::string_view module_str = R"(
    HloModule Test

    ENTRY test_entry {
      inputs = f32[1, 17, 9, 9] parameter(0)
      filters = f32[3, 3, 17, 32] parameter(1)
      mult_op  = s8[1, 32, 9, 9] parameter(2)
      conv = convolution(inputs, filters), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01
      ROOT ret = multiply(s8[1, 32, 9, 9] convert(conv), mult_op)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());
  SCOPED_TRACE(m->ToString());
  HloInstruction* conv1 = nullptr;
  // Checks that it removed the Convert inside multiply around conv.
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::GetTupleElement(m::CustomCall(&conv1)),
                                     m::Parameter(2))));
}

TEST_F(CudnnFusedConvRewriterHloTest, DontRemoveConvertDuetoMultpleUser) {
  const std::string_view module_str = R"(
    HloModule Test

    ENTRY test_entry {
      inputs = f32[1, 17, 9, 9] parameter(0)
      filters = f32[3, 3, 17, 32] parameter(1)
      mult_op  = s8[1, 32, 9, 9] parameter(2)
      sub_op = s8[1, 32, 9, 9] parameter(3)
      conv = convolution(inputs, filters), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01
      another = subtract(s8[1, 32, 9, 9] convert(conv), sub_op)
      ROOT ret = multiply(s8[1, 32, 9, 9] convert(conv), mult_op)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());
  SCOPED_TRACE(m->ToString());
  HloInstruction* conv1 = nullptr;
  // Checks that it removed the Convert inside multiply around conv.
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Convert(m::GetTupleElement(m::CustomCall(&conv1))),
                  m::Parameter(2))));
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseBias) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[32] parameter(2)
      bias_broadcast = f32[1,32,9,9] broadcast(bias), dimensions={1}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = add(conv, bias_broadcast)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall({kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2)),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseSideInput) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      side_input = f32[1,32,9,9] parameter(2)
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = add(conv, side_input)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1),
                            m::Broadcast(m::ConstantEffectiveScalar(0))
                                .WithShape(F32, {32}),
                            m::Parameter(2)),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseScaledSideInput) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      side_input = f32[1,32,9,9] parameter(2)
      side_input_scale = f32[] constant(42)
      side_input_scale_broadcast = f32[1,32,9,9] broadcast(side_input_scale), dimensions={}
      side_input_product = multiply(side_input, side_input_scale_broadcast)
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = add(conv, side_input_product)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1),
                            m::Broadcast(m::ConstantEffectiveScalar(0))
                                .WithShape(F32, {32}),
                            m::Parameter(2)),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 42);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseBiasAndSideInput) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[32] parameter(2)
      side_input = f32[1,32,9,9] parameter(3)
      bias_broadcast = f32[1,32,9,9] broadcast(bias), dimensions={1}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, side_input)
      ROOT sum2 = add(sum, bias_broadcast)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2),
                            m::Parameter(3)),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

TEST_F(CudnnFusedConvRewriterHloTest, EffectiveScalarBias) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] parameter(0)
      filters = f32[3,3,17,32] parameter(1)
      bias = f32[1,32,9,9] broadcast(f32[] parameter(2)), dimensions={}
      conv = f32[1,32,9,9] convolution(inputs, filters),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = add(conv, bias)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1),
                            m::Broadcast(m::Parameter(2)).WithShape(F32, {32})),
              0)
              .WithShape(F32, {1, 32, 9, 9})));
}

TEST_F(CudnnFusedConvRewriterHloTest, StrengthReduceF32ToF16) {
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

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0), m::Parameter(1), m::Parameter(2),
                            m::Parameter(3)),
              0)
              .WithShape(F16, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

// We should be able to lower this to an f16 convolution even though the
// f16-ness of the inputs is hidden behind broadcast/transpose/reshape.
TEST_F(CudnnFusedConvRewriterHloTest, BroadcastReshapeTransposeAfterConvert) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f32[1,17,9,9] reshape(f32[1377] convert(f16[1377] parameter(0)))
      filters = f32[3,3,17,32] transpose(f32[17,32,3,3] convert(f16[17,32,3,3] parameter(1))), dimensions={2,3,0,1}
      bias = f16[1,32,9,9] broadcast(f16[32] parameter(2)), dimensions={1}
      side_input = f16[1,32,9,9] reshape(f16[2592] parameter(3))

      conv_f32 = f32[1,32,9,9] convolution(inputs, filters),
                 window={size=3x3 pad=1_1x1_1},
                 dim_labels=bf01_01io->bf01
      conv_f16 = f16[1,32,9,9] convert(conv_f32)
      ROOT root = f16[1,32,9,9] add(add(conv_f16, side_input), bias)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(
                         &conv, {kCudnnConvBiasActivationForwardCallTarget},
                         m::Convert(m::Reshape(m::Convert(m::Parameter(0))))
                             .WithElementType(F16),
                         m::Convert(m::Transpose(m::Convert(m::Parameter(1))))
                             .WithElementType(F16),
                         m::Parameter(2), m::Reshape(m::Parameter(3))),
                     0)
                     .WithShape(F16, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

TEST_F(CudnnFusedConvRewriterHloTest, NoStrengthReduceF32ToF16IfBiasIsF32) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f16[1,17,9,9] parameter(0)
      filters = f16[3,3,17,32] parameter(1)
      bias = f32[32] parameter(2)
      side_input = f16[1,32,9,9] parameter(3)

      inputs_f32 = f32[1,17,9,9] convert(inputs)
      filters_f32 = f32[3,3,17,32] convert(filters)
      bias_broadcast = f32[1,32,9,9] broadcast(bias), dimensions={1}
      side_input_f32 = f32[1,32,9,9] convert(side_input)
      conv = f32[1,32,9,9] convolution(inputs_f32, filters_f32),
               window={size=3x3 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      sum = add(conv, side_input_f32)
      sum2 = add(sum, bias_broadcast)
      ROOT conv_f16 = f16[1,32,9,9] convert(sum2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  // fp16 convs only support fp16 biases.  Because bias is fp32, it doesn't get
  // fused in, and we get an fp32 conv.
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Convert(m::GetTupleElement(
                         m::CustomCall(
                             &conv, {kCudnnConvBiasActivationForwardCallTarget},
                             m::Convert(m::Parameter(0)).WithElementType(F32),
                             m::Convert(m::Parameter(1)).WithElementType(F32),
                             m::Parameter(2),
                             m::Convert(m::Parameter(3)).WithElementType(F32)),
                         0))
              .WithShape(F16, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

TEST_F(CudnnFusedConvRewriterHloTest, F32Constants) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f16[1,2,2,2] parameter(0)
      filters_f32 = f32[1,1,2,2] constant({{{{1, 2},{3, 4}}}})
      bias = f16[2] parameter(1)
      bias_f32 = f32[2] convert(bias)
      side_input_f32 = f32[1,2,2,2] constant({{
        {{0.5, 0.25}, {0.125, 0.0625}},
        {{0.5, 0.25}, {0.125, 0.0625}}
      }})

      inputs_f32 = f32[1,2,2,2] convert(inputs)
      bias_broadcast = f32[1,2,2,2] broadcast(bias_f32), dimensions={1}
      conv = f32[1,2,2,2] convolution(inputs_f32, filters_f32),
               window={size=1x1}, dim_labels=bf01_01io->bf01
      sum = add(conv, side_input_f32)
      sum2 = add(sum, bias_broadcast)
      ROOT conv_f16 = f16[1,2,2,2] convert(sum2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph, and fold
  // convert back into constants.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());
  HloConstantFolding constant_folding;
  TF_ASSERT_OK(RunHloPass(&constant_folding, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
                     m::CustomCall(
                         &conv, {kCudnnConvBiasActivationForwardCallTarget},
                         m::Parameter(0), m::Constant().WithElementType(F16),
                         m::Parameter(1), m::Constant().WithElementType(F16)),
                     0)
                     .WithShape(F16, {1, 2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

TEST_F(CudnnFusedConvRewriterHloTest, F32ConstantsNotLosslesslyConvertible) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      inputs = f16[1,2,2,2] parameter(0)
      filters_f32 = f32[1,1,2,2] constant({{{{1, 2.123456789},{3, 4}}}})
      bias = f16[2] parameter(1)
      bias_f32 = f32[2] convert(bias)
      side_input_f32 = f32[1,2,2,2] constant({{
        {{0.1, 0.2}, {0.3, 0.4}},
        {{0.5, 0.6}, {0.7, 0.8}}
      }})

      inputs_f32 = f32[1,2,2,2] convert(inputs)
      bias_broadcast = f32[1,2,2,2] broadcast(bias_f32), dimensions={1}
      conv = f32[1,2,2,2] convolution(inputs_f32, filters_f32),
               window={size=1x1}, dim_labels=bf01_01io->bf01
      sum = add(conv, side_input_f32)
      sum2 = add(sum, bias_broadcast)
      ROOT conv_f16 = f16[1,2,2,2] convert(sum2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph, and fold
  // convert back into constants.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());
  HloConstantFolding constant_folding;
  TF_ASSERT_OK(RunHloPass(&constant_folding, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  // This doesn't get transformed into an f16 conv because the filters param is
  // not losslessly expressible as f16.
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Convert(m::GetTupleElement(
                         m::CustomCall(
                             &conv, {kCudnnConvBiasActivationForwardCallTarget},
                             m::Convert(m::Parameter(0)).WithElementType(F32),
                             m::Constant().WithElementType(F32),
                             m::Convert(m::Parameter(1)).WithElementType(F32),
                             m::Constant().WithElementType(F32)),
                         0)
                         .WithShape(F32, {1, 2, 2, 2}))
              .WithElementType(F16)));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.side_input_scale(), 1);
}

TEST_F(CudnnFusedConvRewriterHloTest, FuseReluBeforeConvert) {
  const std::string module_str = R"(
  HloModule Test

  ENTRY Test {
    input = s8[1,17,9,9] parameter(0)
    filter = s8[3,3,17,32] parameter(1)
    inputs32 = s32[1,17,9,9] convert(input)
    filters32 = s32[3,3,17,32] convert(filter)

    conv = s32[1,32,9,9] convolution(inputs32, filters32), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1

    zero = s32[] constant(0)
    zeros = s32[1,32,9,9] broadcast(zero), dimensions={}
    relu = maximum(conv, zeros)

    lower = s32[] constant(-128)
    lowers = s32[1,32,9,9] broadcast(lower), dimensions={}
    upper = s32[] constant(127)
    uppers = s32[1,32,9,9] broadcast(upper), dimensions={}

    clamp = s32[1,32,9,9] clamp(lowers, relu, uppers)

    ROOT convert = s8[1,32,9,9] convert(clamp)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0),  //
                            m::Parameter(1),  //
                            m::Broadcast(m::ConstantEffectiveScalar(0))
                                .WithShape(F32, {32})),
              0)
              .WithShape(S8, {1, 32, 9, 9})));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_config,
                          conv->backend_config<GpuBackendConfig>());
  const CudnnConvBackendConfig& config = gpu_config.cudnn_conv_backend_config();
  EXPECT_EQ(config.activation_mode(), se::dnn::kRelu);
}

TEST_F(CudnnFusedConvRewriterHloTest, BiasTypeMatchesConvTypeIfFp) {
  const std::string module_str = R"(
  HloModule Test

  ENTRY Test {
    input = f64[1,17,9,9] parameter(0)
    filter = f64[3,3,17,32] parameter(1)
    bias = f64[1,32,9,9] broadcast(f64[32] convert(f32[32] parameter(2))), dimensions={1}
    conv = f64[1,32,9,9] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, feature_group_count=1
    ROOT root = f64[1,32,9,9] add(conv, bias)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  GpuConvRewriter rewriter;
  TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());
  CudnnFusedConvRewriter fuser{GetCudaComputeCapability()};
  TF_ASSERT_OK(RunHloPass(&fuser, m.get()).status());

  // Simplify new `convert`'s that may be added to the graph.
  AlgebraicSimplifier algsimp(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK(RunHloPass(&algsimp, m.get()).status());

  SCOPED_TRACE(m->ToString());
  const HloInstruction* conv;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::GetTupleElement(
              m::CustomCall(&conv, {kCudnnConvBiasActivationForwardCallTarget},
                            m::Parameter(0),  //
                            m::Parameter(1),  //
                            m::Convert(m::Parameter(2)).WithShape(F64, {32})),
              0)
              .WithShape(F64, {1, 32, 9, 9})));
}

TEST_F(CudnnFusedConvRewriterTest, TestFusedConvInt8ToInt8) {
  // clamp(max(0, conv(x, w)+bias)); for int8_t
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
      relu = f32[1,3,3,64] maximum(zeros, add1)

      lower = f32[] constant(-128)
      lowers = f32[1,3,3,64] broadcast(lower), dimensions={}
      upper = f32[] constant(127)
      uppers = f32[1,3,3,64] broadcast(upper), dimensions={}

      clamp = f32[1,3,3,64] clamp(lowers, relu, uppers)

      ROOT convert = s8[1,3,3,64] convert(clamp)
    })",
      // post_hlo
      R"(
// CHECK: [[cudnn_conv_bias_activation_7_0:%[^ ]+]] = (s8[1,3,3,64]{3,2,1,0}, u8[{{[0-9]+}}]{0}) custom-call([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBiasActivationForward"
      )");
}

// Disabled per b/190854862 or nvbugs/3326122.
TEST_F(CudnnFusedConvRewriterTest, DISABLED_TestFusedConvInt8ToFloat) {
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

TEST_F(CudnnFusedConvRewriterTest,
       TestFusedConvWithScaledInt8SideInputBiasInt8ToInt8) {
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
// CHECK: [[cudnn_conv_bias_activation_11_0:%[^ ]+]] = (s8[1,3,3,64]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]], [[side_input_4:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBiasActivationForward"
      )");
}

TEST_F(CudnnFusedConvRewriterTest,
       TestFusedConvWithScaledFloatSideInputBiasInt8ToInt8) {
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
// CHECK: [[cudnn_conv_bias_activation_9_0:%[^ ]+]] = (f32[1,3,3,64]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]], [[side_input_4:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBiasActivationForward"
      )");
}

TEST_F(CudnnFusedConvRewriterTest,
       TestFusedConvWithScaledInt8SideInputBiasInt8ToFloat) {
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
// CHECK: [[cudnn_conv_bias_activation_9_0:%[^ ]+]] = (f32[1,3,3,64]{3,2,1,0}, u8[{{.*}}]{0}) custom-call([[input_1:%[^ ]+]], [[transpose_2:%[^ ]+]], [[bias_3:%[^ ]+]], [[fusion_1_4:%[^ ]+]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convBiasActivationForward"
      )");
}

TEST_F(CudnnFusedConvRewriterTest, TestConvInt8ToInt8NoClamp) {
  // Check that integer convolution without clamp to int8_t is not allowed.
  // convert<int8_t>(custom_call<int32_t>(int32_x, int32_w,
  // cudnnConvolutionForward))
  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test (input: s8[1,17,9,9], filter: s8[3,3,17,32]) -> s8[1,32,9,9] {
      zero = s8[] constant(0)
      zeros = s8[1,32,9,9]{3,2,1,0} broadcast(s8[] zero), dimensions={}
      input = s8[1,17,9,9]{3,2,1,0} parameter(0)
      filter = s8[3,3,17,32]{3,2,1,0} parameter(1)
      custom-call = (s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call(s8[1,17,9,9]{3,2,1,0} input, s8[3,3,17,32]{3,2,1,0} filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config="{\"convResultScale\":1}"
      get-tuple-element = s32[1,32,9,9]{3,2,1,0} get-tuple-element((s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call), index=0
      convert = s8[1,32,9,9]{3,2,1,0} convert(s32[1,32,9,9]{3,2,1,0} get-tuple-element)
      ROOT relu = s8[1,32,9,9]{3,2,1,0} maximum(s8[1,32,9,9]{3,2,1,0} zeros, s8[1,32,9,9]{3,2,1,0} convert)
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ASSERT_FALSE(
      CudnnFusedConvRewriter(GetCudaComputeCapability()).Run(m.get()).ok());
}

TEST_F(CudnnFusedConvRewriterTest, TestFusedConvInt8ToInt8NoClamp) {
  // Although bias and so on are fused with forward convolution,
  // it is still not allowed if the output is not clampped/converted to int8_t
  // max(0, alpha_conv * conv(x, w) + alpha_side * side_input + bias); for
  // int8_t

  const std::string module_str = absl::StrFormat(R"(
    HloModule Test

    ENTRY Test (input: s8[1,17,9,9], filter: s8[3,3,17,32]) -> s8[1,32,9,9] {
      zero = s8[] constant(0)
      zeros = s8[1,32,9,9]{3,2,1,0} broadcast(s8[] zero), dimensions={}
      input = s8[1,17,9,9]{3,2,1,0} parameter(0)
      filter = s8[3,3,17,32]{3,2,1,0} parameter(1)
      custom-call = (s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call(s8[1,17,9,9]{3,2,1,0} input, s8[3,3,17,32]{3,2,1,0} filter), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convForward", backend_config="{\"convResultScale\":1}"
      get-tuple-element = s32[1,32,9,9]{3,2,1,0} get-tuple-element((s32[1,32,9,9]{3,2,1,0}, u8[0]{0}) custom-call), index=0
      convert = s8[1,32,9,9]{3,2,1,0} convert(s32[1,32,9,9]{3,2,1,0} get-tuple-element)
      ROOT relu = s8[1,32,9,9]{3,2,1,0} maximum(s8[1,32,9,9]{3,2,1,0} zeros, s8[1,32,9,9]{3,2,1,0} convert)
    })");
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));

  ASSERT_FALSE(
      CudnnFusedConvRewriter(GetCudaComputeCapability()).Run(m.get()).ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
