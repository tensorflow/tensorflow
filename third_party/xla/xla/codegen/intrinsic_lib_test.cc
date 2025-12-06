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
#include "llvm/Analysis/TargetLibraryInfo.h"
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
}  // namespace

}  // namespace xla::codegen::intrinsics
