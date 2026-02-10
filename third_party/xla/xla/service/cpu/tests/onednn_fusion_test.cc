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

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/onednn_support.h"
#include "xla/error_spec.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {
namespace {

inline constexpr bool IsOneDnnGraphEnabled() {
#if defined(XLA_ONEDNN_USE_GRAPH_API)
  // Some Aarch64 CPUs have failures. Only test on x86 for now.
  return tsl::port::IsX86CPU();
#endif  // XLA_ONEDNN_USE_GRAPH_API
  return false;
}

struct OneDnnFusionTestParams {
  PrimitiveType dtype;
  std::string op_type;
};

class OneDnnFusionTestBase
    : public HloTestBase,
      public ::testing::WithParamInterface<OneDnnFusionTestParams> {
 protected:
  void SetUp() override {
    OneDnnFusionTestParams params = GetParam();
    data_type_ = params.dtype;
    op_type_ = params.op_type;
    atol_ = (data_type_ == F32) ? 1e-4 : 1e-2;
    rtol_ = (data_type_ == F32) ? 1e-4 : 1e-2;

    if (!IsOneDnnGraphEnabled()) {
      GTEST_SKIP() << "oneDNN fusion is not supported";
    }

    if (!IsSupportedType(data_type_)) {
      GTEST_SKIP() << "CPU does not support dtype: "
                   << primitive_util::LowercasePrimitiveTypeName(data_type_);
    }
  }

  PrimitiveType data_type_;
  std::string op_type_;
  float atol_;
  float rtol_;
};

class OneDnnFusionBinaryOpTest : public OneDnnFusionTestBase {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<OneDnnFusionTestParams>& data) {
    return absl::StrCat(
        data.param.op_type, "_",
        primitive_util::LowercasePrimitiveTypeName(data.param.dtype));
  }

 protected:
  void RunTest() {
    std::string hlo_template =
        (op_type_ == "dot") ? GetMatMulHLOTemplate() : GetBinaryOpHLOTemplate();
    std::string hlo_binary_str = absl::StrReplaceAll(
        hlo_template,
        {{"$dtype", primitive_util::LowercasePrimitiveTypeName(data_type_)},
         {"$op_type", op_type_}});
    EXPECT_TRUE(RunAndCompare(hlo_binary_str, ErrorSpec{atol_, rtol_}));
  }

 private:
  const std::string GetBinaryOpHLOTemplate() {
    return R"(
    HloModule binary_op

    onednn_fusion {
      %p0 = $dtype[10, 20] parameter(0)
      %p1 = $dtype[10, 20] parameter(1)
      ROOT %op = $dtype[10, 20] $op_type(%p0, %p1)
    }

    ENTRY entry {
      %p0 = $dtype[10, 20] parameter(0)
      %p1 = $dtype[10, 20] parameter(1)
      ROOT %fusion = $dtype[10, 20] fusion(%p0, %p1), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";
  }

  const std::string GetMatMulHLOTemplate() {
    return R"(
    HloModule matmul

    onednn_fusion {
      %p0 = $dtype[1000,200] parameter(0)
      %p1 = $dtype[200,300] parameter(1)
      ROOT %mul = $dtype[1000,300] $op_type(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY entry {
      %p0 = $dtype[1000,200] parameter(0)
      %p1 = $dtype[200,300] parameter(1)
      ROOT %fusion = $dtype[1000,300] fusion(%p0, %p1), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";
  }
};

TEST_P(OneDnnFusionBinaryOpTest, BinaryOp) { RunTest(); }

std::vector<OneDnnFusionTestParams> GetOneDnnFusionBinaryOpTestSpecs() {
  std::vector<OneDnnFusionTestParams> specs;
  for (const auto& op_type : GetOneDnnSupportedBinaryOpsStrings()) {
    specs.push_back({PrimitiveType::F32, std::string(op_type)});
  }
  specs.push_back({PrimitiveType::BF16, "dot"});
  specs.push_back({PrimitiveType::F16, "dot"});
  return specs;
}

INSTANTIATE_TEST_SUITE_P(
    OneDnnFusionBinaryOpTestSuite, OneDnnFusionBinaryOpTest,
    ::testing::ValuesIn(GetOneDnnFusionBinaryOpTestSpecs()),
    OneDnnFusionBinaryOpTest::Name);

class OneDnnFusionUnaryOpTest : public OneDnnFusionTestBase {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<OneDnnFusionTestParams>& data) {
    return absl::StrCat(
        data.param.op_type, "_",
        primitive_util::LowercasePrimitiveTypeName(data.param.dtype));
  }

 protected:
  void RunTest() {
    absl::string_view hlo_unary_template = R"(
      HloModule unary_op

      onednn_fusion {
        %p0 = $dtype[40] parameter(0)
        ROOT %op = $dtype[40] $op_type(%p0)
      }

      ENTRY entry {
        %p0 = $dtype[40] parameter(0)
        ROOT %fusion = $dtype[40] fusion(%p0), kind=kCustom,
          calls=onednn_fusion,
          backend_config={"fusion_config": {kind: "__onednn_fusion"}}
      })";

    std::string hlo_unary_str = absl::StrReplaceAll(
        hlo_unary_template,
        {{"$dtype", primitive_util::LowercasePrimitiveTypeName(data_type_)},
         {"$op_type", op_type_}});

    EXPECT_TRUE(RunAndCompare(hlo_unary_str, ErrorSpec{atol_, rtol_}));
  }
};

TEST_P(OneDnnFusionUnaryOpTest, UnaryOp) { RunTest(); }

std::vector<OneDnnFusionTestParams> GetOneDnnFusionUnaryOpTestSpecs() {
  std::vector<OneDnnFusionTestParams> specs;
  for (const auto& op_type : GetOneDnnSupportedUnaryOpsStrings()) {
    specs.push_back({PrimitiveType::F32, std::string(op_type)});
  }
  return specs;
}

INSTANTIATE_TEST_SUITE_P(OneDnnFusionUnaryOpTestSuite, OneDnnFusionUnaryOpTest,
                         ::testing::ValuesIn(GetOneDnnFusionUnaryOpTestSpecs()),
                         OneDnnFusionUnaryOpTest::Name);

class OneDnnFusionMatMulFuseBinaryTest : public OneDnnFusionTestBase {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<OneDnnFusionTestParams>& data) {
    return absl::StrCat(
        data.param.op_type, "_",
        primitive_util::LowercasePrimitiveTypeName(data.param.dtype));
  }

 protected:
  void RunTest() {
    absl::string_view hlo_fusion_template = R"(
    HloModule matmul_fusion

    onednn_fusion {
      %p0 = $dtype[1000,200] parameter(0)
      %p1 = $dtype[200,300] parameter(1)
      %p2 = $dtype[1000,300] parameter(2)
      %dot = $dtype[1000,300] dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %root = $dtype[1000,300] $op_type(%dot, %p2)
    }

    ENTRY entry {
      %p0 = $dtype[1000,200] parameter(0)
      %p1 = $dtype[200,300] parameter(1)
      %p2 = $dtype[1000,300] parameter(2)
      ROOT %fusion = $dtype[1000,300] fusion(%p0, %p1, %p2), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
      })";

    std::string hlo_fusion_str = absl::StrReplaceAll(
        hlo_fusion_template,
        {{"$dtype", primitive_util::LowercasePrimitiveTypeName(data_type_)},
         {"$op_type", op_type_}});

    EXPECT_TRUE(RunAndCompare(hlo_fusion_str, ErrorSpec{atol_, rtol_}));
  }
};

TEST_P(OneDnnFusionMatMulFuseBinaryTest, MatmulFuseWith) { RunTest(); }

std::vector<OneDnnFusionTestParams> GetOneDnnFusionFuseBinaryTestSpecs() {
  std::vector<OneDnnFusionTestParams> specs;
  for (const auto& dtype : {PrimitiveType::F32}) {
    for (const auto& op_type : GetOneDnnSupportedBinaryOpsStrings()) {
      // oneDNN does not support fusing two dot instructions
      if (op_type == HloOpcodeString(HloOpcode::kDot)) continue;
      specs.push_back({dtype, std::string(op_type)});
    }
  }
  return specs;
}

INSTANTIATE_TEST_SUITE_P(
    OneDnnFusionMatMulFusionTestSuite, OneDnnFusionMatMulFuseBinaryTest,
    ::testing::ValuesIn(GetOneDnnFusionFuseBinaryTestSpecs()),
    OneDnnFusionMatMulFuseBinaryTest::Name);

class OneDnnFusionMatMulFuseUnaryTest : public OneDnnFusionTestBase {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<OneDnnFusionTestParams>& data) {
    return absl::StrCat(
        data.param.op_type, "_",
        primitive_util::LowercasePrimitiveTypeName(data.param.dtype));
  }

 protected:
  void RunTest() {
    absl::string_view hlo_fusion_template = R"(
    HloModule matmul_fusion

    onednn_fusion {
      %p0 = $dtype[1000,200] parameter(0)
      %p1 = $dtype[200,300] parameter(1)
      %dot = $dtype[1000,300] dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %root = $dtype[1000,300] $op_type(%dot)
    }

    ENTRY entry {
      %p0 = $dtype[1000,200] parameter(0)
      %p1 = $dtype[200,300] parameter(1)
      ROOT %fusion = $dtype[1000,300] fusion(%p0, %p1), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
      })";

    std::string hlo_fusion_str = absl::StrReplaceAll(
        hlo_fusion_template,
        {{"$dtype", primitive_util::LowercasePrimitiveTypeName(data_type_)},
         {"$op_type", op_type_}});

    EXPECT_TRUE(RunAndCompare(hlo_fusion_str, ErrorSpec{atol_, rtol_}));
  }
};

TEST_P(OneDnnFusionMatMulFuseUnaryTest, MatmulFuseWith) { RunTest(); }

std::vector<OneDnnFusionTestParams> GetOneDnnFusionFuseUnaryTestSpecs() {
  std::vector<OneDnnFusionTestParams> specs;
  for (const auto& dtype : {PrimitiveType::F32}) {
    for (const auto& op_type : GetOneDnnSupportedUnaryOpsStrings()) {
      specs.push_back({dtype, std::string(op_type)});
    }
  }
  return specs;
}

INSTANTIATE_TEST_SUITE_P(
    OneDnnFusionMatMulFusionTestSuite, OneDnnFusionMatMulFuseUnaryTest,
    ::testing::ValuesIn(GetOneDnnFusionFuseUnaryTestSpecs()),
    OneDnnFusionMatMulFuseUnaryTest::Name);

}  // namespace
}  // namespace xla::cpu
