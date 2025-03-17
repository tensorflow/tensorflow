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

#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_base.h"

namespace xla {
namespace {

using NumericTest = ClientLibraryTestBase;

TEST_F(NumericTest, Remainder) {
  XlaBuilder builder("remainder");
  auto x_literal =
      LiteralUtil::CreateR1<float>({2.9375, 2.9375, -2.9375, -2.9375});
  auto y_literal =
      LiteralUtil::CreateR1<float>({2.9375, -2.9375, 2.9375, -2.9375});
  auto x = Parameter(&builder, 0, x_literal.shape(), "x");
  auto y = Parameter(&builder, 1, y_literal.shape(), "y");
  Rem(x, y);
  ComputeAndCompare(&builder, {std::move(x_literal), std::move(y_literal)});
}

}  // namespace
}  // namespace xla
