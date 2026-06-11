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

#include "xla/codegen/intrinsic_lib.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/intrinsic.h"

namespace xla::codegen::intrinsics {
namespace {

using ::testing::UnorderedElementsAre;

std::string ToString(const llvm::VecDesc& vec_desc) {
  return absl::StrJoin(
      {vec_desc.getScalarFnName().str(), vec_desc.getVectorFnName().str(),
       absl::StrCat(vec_desc.getVectorizationFactor().getKnownMinValue()),
       vec_desc.getVABIPrefix()},
      ":");
}

TEST(IntrinsicLibTest, ExpVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.exp")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(vec_descs_str, UnorderedElementsAre(
                                 "xla.exp.f64:xla.exp.v2f64:2:_ZGV_LLVM_N2v",
                                 "xla.exp.f64:xla.exp.v4f64:4:_ZGV_LLVM_N4v",
                                 "xla.exp.f64:xla.exp.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, AtanVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.atan")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(
      vec_descs_str,
      UnorderedElementsAre("xla.atan.f32:xla.atan.v4f32:4:_ZGV_LLVM_N4v",
                           "xla.atan.f32:xla.atan.v8f32:8:_ZGV_LLVM_N8v",
                           "xla.atan.f32:xla.atan.v16f32:16:_ZGV_LLVM_N16v",
                           "xla.atan.f64:xla.atan.v4f64:4:_ZGV_LLVM_N4v",
                           "xla.atan.f64:xla.atan.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, SinVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.sin")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(vec_descs_str, UnorderedElementsAre(
                                 "xla.sin.f64:xla.sin.v2f64:2:_ZGV_LLVM_N2v"));
}

TEST(IntrinsicLibTest, CosVectorizations) {
  IntrinsicOptions options;
  auto lib = IntrinsicFunctionLib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with("xla.cos")) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }

  EXPECT_THAT(vec_descs_str, UnorderedElementsAre(
                                 "xla.cos.f64:xla.cos.v2f64:2:_ZGV_LLVM_N2v"));
}

TEST(IntrinsicLibTest, AreEigenIntrinsicsAvailable) {
  // On our standard Linux test environment, this should always be available.
  EXPECT_TRUE(AreEigenIntrinsicsAvailable());
}

std::vector<std::string> GetVectorizations(const IntrinsicOptions& options,
                                           absl::string_view prefix) {
  IntrinsicFunctionLib lib(options);
  std::vector<llvm::VecDesc> vec_descs = lib.Vectorizations();
  std::vector<std::string> vec_descs_str;
  for (const auto& vec_desc : vec_descs) {
    if (vec_desc.getScalarFnName().starts_with(prefix)) {
      vec_descs_str.push_back(ToString(vec_desc));
    }
  }
  return vec_descs_str;
}

TEST(IntrinsicLibTest, SinVectorizationsAvx512) {
  IntrinsicOptions options;
  options.features = "+avx512f";
  EXPECT_THAT(
      GetVectorizations(options, "xla.sin"),
      UnorderedElementsAre("xla.sin.f64:xla.sin.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.sin.f64:xla.sin.v4f64:4:_ZGV_LLVM_N4v",
                           "xla.sin.f64:xla.sin.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, CosVectorizationsAvx512) {
  IntrinsicOptions options;
  options.features = "+avx512f";
  EXPECT_THAT(
      GetVectorizations(options, "xla.cos"),
      UnorderedElementsAre("xla.cos.f64:xla.cos.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.cos.f64:xla.cos.v4f64:4:_ZGV_LLVM_N4v",
                           "xla.cos.f64:xla.cos.v8f64:8:_ZGV_LLVM_N8v"));
}

TEST(IntrinsicLibTest, SinVectorizationsAvx2) {
  IntrinsicOptions options;
  options.features = "+avx2";
  EXPECT_THAT(
      GetVectorizations(options, "xla.sin"),
      UnorderedElementsAre("xla.sin.f64:xla.sin.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.sin.f64:xla.sin.v4f64:4:_ZGV_LLVM_N4v"));
}

TEST(IntrinsicLibTest, CosVectorizationsAvx2) {
  IntrinsicOptions options;
  options.features = "+avx2";
  EXPECT_THAT(
      GetVectorizations(options, "xla.cos"),
      UnorderedElementsAre("xla.cos.f64:xla.cos.v2f64:2:_ZGV_LLVM_N2v",
                           "xla.cos.f64:xla.cos.v4f64:4:_ZGV_LLVM_N4v"));
}

}  // namespace

}  // namespace xla::codegen::intrinsics
