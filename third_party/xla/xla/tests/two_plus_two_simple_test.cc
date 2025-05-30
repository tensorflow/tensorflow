/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/client/local_client.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/service.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using TwoPlusTwoSimpleTest = ClientLibraryTestBase;

TEST_F(TwoPlusTwoSimpleTest, TwoPlusTwoVector) {
  XlaBuilder builder("two_plus_two");
  auto x = ConstantR1<float>(&builder, {2.0, 2.0});
  auto y = ConstantR1<float>(&builder, {2.0, 2.0});
  Add(x, y);

  std::vector<float> expected = {4.0, 4.0};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(TwoPlusTwoSimpleTest, TwoPlusTwoScalarWithOneTransfer) {
  Literal x_literal = LiteralUtil::CreateR0<float>(1.0f);
  std::unique_ptr<GlobalData> x_data =
      client_->TransferToServer(x_literal).value();

  XlaBuilder builder("one_transfer");
  auto x = Parameter(&builder, 0,
                     ShapeUtil::MakeValidatedShape(F32, {}).value(), "x_value");
  auto y = ConstantR0<float>(&builder, 2.0);
  Add(x, y);

  float expected = 3.0f;
  ComputeAndCompareR0<float>(&builder, expected, {x_data.get()},
                             ErrorSpec(0.0001));
}

TEST_F(TwoPlusTwoSimpleTest, TwoPlusTwoScalarWithTwoTransfer) {
  Literal x_literal = LiteralUtil::CreateR0<float>(1.0f);
  std::unique_ptr<GlobalData> x_data =
      client_->TransferToServer(x_literal).value();
  Literal y_literal = LiteralUtil::CreateR0<float>(2.0f);
  std::unique_ptr<GlobalData> y_data =
      client_->TransferToServer(y_literal).value();

  XlaBuilder builder("two_transfers");
  auto x = Parameter(&builder, 0,
                     ShapeUtil::MakeValidatedShape(F32, {}).value(), "x_value");
  auto y = Parameter(&builder, 1,
                     ShapeUtil::MakeValidatedShape(F32, {}).value(), "y_value");
  Add(x, y);

  float expected = 3.0f;
  ComputeAndCompareR0<float>(&builder, expected, {x_data.get(), y_data.get()},
                             ErrorSpec(0.0001));

  auto* outputs_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputs_dir != nullptr) {
    std::vector<std::string> paths;
    TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
        tsl::io::JoinPath(outputs_dir, "*-hlo-static-bundle-profile.txt"),
        &paths));
    int64_t newest = -1;
    const std::string* newest_path = nullptr;
    for (const auto& path : paths) {
      absl::string_view file = tsl::io::Basename(path);
      size_t pos = file.find('-');
      ASSERT_NE(pos, absl::string_view::npos);
      absl::string_view timestamp_str = file.substr(0, pos);
      int64_t timestamp;
      ASSERT_TRUE(absl::SimpleAtoi(timestamp_str, &timestamp));
      if (timestamp > newest) {
        newest = timestamp;
        newest_path = &path;
      }
    }
    if (newest_path) {
      std::string contents;
      TF_ASSERT_OK(
          tsl::ReadFileToString(tsl::Env::Default(), *newest_path, &contents));
      EXPECT_TRUE(absl::StartsWith(contents, "HLO: <no-hlo-instruction>"));
    }
  }
}

}  // namespace
}  // namespace xla
