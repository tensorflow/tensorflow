// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_op_options.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert {
namespace {

TEST(OpOptionsTest, GetCompositeOptions) {
  static constexpr auto kOptsType =
      ::tflite::BuiltinOptions2_StableHLOCompositeOptions;
  static constexpr absl::string_view kName = "test.composite";
  static constexpr int kSubgraph = 1;

  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);

  tflite::StableHLOCompositeOptionsT options;
  options.name = kName;
  options.decomposition_subgraph_index = kSubgraph;

  internal::TflOptions2 tfl_options;
  tfl_options.type = kOptsType;
  tfl_options.Set(std::move(options));
  detail::SetTflOptions2(op, std::move(tfl_options));

  auto res = GetOptionsAs<CompositeOptions>(&op);
  ASSERT_TRUE(res);
  EXPECT_EQ(res->name, kName);
  EXPECT_EQ(res->subgraph, kSubgraph);
}

TEST(OpOptionsTest, GetUnsupportedOptions) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloAdd);
  ASSERT_FALSE(GetOptionsAs<CompositeOptions>(&op));
}

}  // namespace
}  // namespace litert
