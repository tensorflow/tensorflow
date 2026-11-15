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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {

namespace {

class GemmSyclTest : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  void TestGemmWithTypeVariations(absl::string_view hlo_template,
                                  bool include_complex) {
    std::vector<std::tuple<absl::string_view, absl::string_view>>
        type_combinations = {
            {"f32", "f32"}, {"f16", "f16"}, {"bf16", "bf16"}, {"f64", "f64"}};
    if (include_complex) {
      type_combinations.push_back({"c64", "c64"});
      type_combinations.push_back({"c128", "c128"});
    }

    for (const auto& type_combination : type_combinations) {
      VLOG(3) << "Testing type combination: " << std::get<0>(type_combination)
              << ", " << std::get<1>(type_combination);
      absl::flat_hash_map<absl::string_view, absl::string_view> replacements;
      replacements["<<ABType>>"] = std::get<0>(type_combination);
      replacements["<<DType>>"] = std::get<1>(type_combination);
      const auto hlo_text = absl::StrReplaceAll(hlo_template, replacements);
      double tol;
      if (std::get<0>(type_combination) == "f32" ||
          std::get<0>(type_combination) == "f64" ||
          std::get<0>(type_combination) == "c64" ||
          std::get<0>(type_combination) == "c128") {
        // Use tighter error bounds.
        tol = 1e-4;
      } else {
        tol = 1e-2;
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
  TestGemmWithTypeVariations(hlo_module, /*include_complex=*/true);
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
  TestGemmWithTypeVariations(matmul_module_str, /*include_complex=*/false);
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
  TestGemmWithTypeVariations(hlo_module, /*include_complex=*/false);
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
  TestGemmWithTypeVariations(matmul_module_str, /*include_complex=*/false);
}

// Note that oneAPI does not fuse F64 activations,
// so the f64 variant exercises BIAS only.
TEST_F(GemmSyclTest, MatmulWithBiasAndRELU) {
  const char* matmul_module_str = R"(
  HloModule matmul.bias.relu.test

  ENTRY module {
    %parameter.1 = <<ABType>>[256,128]{1,0} parameter(0)
    %parameter.2 = <<ABType>>[128,8]{1,0} parameter(1)
    %parameter.3 = <<DType>>[8]{0} parameter(2)
    %dot = <<DType>>[256,8] dot(%parameter.1, %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    bias_bcast = <<DType>>[256,8] broadcast(%parameter.3), dimensions={1}
    biased = <<DType>>[256,8] add(%dot, bias_bcast)
    c = <<DType>>[] constant(0)
    c_bcast = <<DType>>[256,8] broadcast(c), dimensions={}
    ROOT out = <<DType>>[256,8] maximum(biased, c_bcast)
  })";
  TestGemmWithTypeVariations(matmul_module_str, /*include_complex=*/false);
}

TEST_F(GemmSyclTest, MatmulWithBiasAndApproxGELU) {
  const char* matmul_module_str = R"(
  HloModule matmul.bias.gelu.test

  ENTRY module {
    %parameter.1 = <<ABType>>[256,128]{1,0} parameter(0)
    %parameter.2 = <<ABType>>[128,8]{1,0} parameter(1)
    %parameter.3 = <<DType>>[8]{0} parameter(2)
    %dot = <<DType>>[256,8] dot(%parameter.1, %parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    bias_bcast = <<DType>>[256,8] broadcast(%parameter.3), dimensions={1}
    biased = <<DType>>[256,8] add(%dot, bias_bcast)
    mul.0 = <<DType>>[256,8] multiply(biased, biased)
    mul.1 = <<DType>>[256,8] multiply(biased, mul.0)
    const.0 = <<DType>>[] constant(0.044715)
    bcast.0 = <<DType>>[256,8] broadcast(const.0), dimensions={}
    mul.2 = <<DType>>[256,8] multiply(mul.1, bcast.0)
    add.0 = <<DType>>[256,8] add(biased, mul.2)
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
    ROOT out = <<DType>>[256,8] multiply(biased, mul.4)
  })";
  TestGemmWithTypeVariations(matmul_module_str, /*include_complex=*/false);
}

class SyclBlasLtTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        platform_, stream_executor::PlatformManager::PlatformWithName("SYCL"));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    LOG(INFO) << "Device name: " << executor_->GetDeviceDescription().name();
    ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    ASSERT_OK_AND_ASSIGN(blas_lt_,
                         stream_executor::gpu::BlasLt::Get(executor_));
  }

  template <typename LhsT, typename RhsT, typename OutputT>
  void TestComplexTypeMismatch() {
    int64_t m = 32, n = 32, k = 32;

    std::vector<LhsT> a(m * k, 1);
    std::vector<RhsT> b(k * n, 1);
    std::vector<LhsT> c(m * n, static_cast<LhsT>(0));

    auto a_buf = executor_->AllocateArray<LhsT>(a.size());
    auto b_buf = executor_->AllocateArray<RhsT>(b.size());
    auto c_buf = executor_->AllocateArray<OutputT>(c.size());

    ASSERT_OK(stream_->Memcpy(&a_buf, a.data(), a.size() * sizeof(LhsT)));
    ASSERT_OK(stream_->Memcpy(&b_buf, b.data(), b.size() * sizeof(RhsT)));
    ASSERT_OK(stream_->MemZero(&c_buf, c.size() * sizeof(OutputT)));

    stream_executor::gpu::MatrixLayout a_layout(
        xla::primitive_util::NativeToPrimitiveType<LhsT>(), m, k,
        stream_executor::gpu::MatrixLayout::Order::kColumnMajor);
    stream_executor::gpu::MatrixLayout b_layout(
        xla::primitive_util::NativeToPrimitiveType<RhsT>(), k, n,
        stream_executor::gpu::MatrixLayout::Order::kColumnMajor);
    stream_executor::gpu::MatrixLayout c_layout(
        xla::primitive_util::NativeToPrimitiveType<OutputT>(), m, n,
        stream_executor::gpu::MatrixLayout::Order::kColumnMajor);

    stream_executor::gpu::GemmConfig cfg = {
        a_layout,                                // lhs_layout
        b_layout,                                // rhs_layout
        c_layout,                                // c_layout
        c_layout,                                // output_layout
        {1.0, 0.0},                              // alpha
        0.0,                                     // beta
        0,                                       // compute_precision
        xla::PrecisionConfig::ALG_UNSET,         // precision_algorithm
        std::nullopt,                            // algorithm
        false,                                   // grad_x
        false,                                   // grad_y
        stream_executor::gpu::ScaleMode::kNone,  // scale_mode
    };

    ASSERT_OK_AND_ASSIGN(
        auto plan, blas_lt_->GetMatmulPlan(
                       cfg, stream_executor::gpu::BlasLt::Epilogue::kDefault));

    stream_executor::gpu::BlasLt::MemoryArgs args{
        /*a=*/a_buf,
        /*b=*/b_buf,
        /*c=*/c_buf,
        /*d=*/c_buf,
        /*bias=*/stream_executor::DeviceAddressBase{},
        /*aux=*/stream_executor::DeviceAddressBase{},
        /*a_scale=*/stream_executor::DeviceAddressBase{},
        /*b_scale=*/stream_executor::DeviceAddressBase{},
        /*c_scale=*/stream_executor::DeviceAddressBase{},
        /*d_scale=*/stream_executor::DeviceAddressBase{},
        /*d_amax=*/{stream_executor::DeviceAddressBase{}},
        /*workspace*/ stream_executor::DeviceAddressBase{},
        /*scratch_allocator=*/nullptr,
    };

    auto status = plan->ExecuteOnStream(stream_.get(), args, nullptr);
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(
        status.message(),
        ::testing::HasSubstr("Output datatype must match input data types"));

    ASSERT_OK(stream_->BlockHostUntilDone());
    executor_->Deallocate(&a_buf);
    executor_->Deallocate(&b_buf);
    executor_->Deallocate(&c_buf);
  }

  stream_executor::Platform* platform_;
  stream_executor::StreamExecutor* executor_;
  std::unique_ptr<stream_executor::Stream> stream_;
  stream_executor::gpu::BlasLt* blas_lt_;
};

TEST_F(SyclBlasLtTest, ComplexTypeMismatch) {
  TestComplexTypeMismatch<std::complex<double>, std::complex<float>,
                          std::complex<float>>();
}

}  // namespace
}  // namespace xla::gpu
