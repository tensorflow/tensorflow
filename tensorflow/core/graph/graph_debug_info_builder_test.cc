/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/graph_debug_info_builder.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
using ::testing::Eq;
using ::testing::Ne;
using ::testing::UnorderedElementsAre;

class TestStackTrace : public AbstractStackTrace {
 public:
  explicit TestStackTrace(const std::vector<StackFrame> frames)
      : frames_(std::move(frames)) {}

  absl::Span<StackFrame const> ToFrames() const override { return frames_; }

  std::vector<StackFrame> GetUserFrames(int limit) const override {
    return frames_;
  }

  StackFrame LastUserFrame() const override { return frames_.back(); }

  string ToString(const TracePrintingOptions& opts) const override {
    auto frame = LastUserFrame();
    return absl::StrCat(frame.file_name, ":", frame.line_number, ":",
                        frame.function_name);
  }

  std::vector<StackFrame> frames_;
};

TEST(GraphDebugInfoBuilderTest, AccumulateStackTrace) {
  TestStackTrace stack_trace(
      std::vector<StackFrame>{{"dummy_file_alpha.cc", 20, "function_bar"},
                              {"dummy_file_beta.cc", 30, "function_sop"}});

  GraphDebugInfoBuilder builder;
  builder.AccumulateStackTrace(stack_trace, "alpha_beta");
  GraphDebugInfo debug_info = builder.Build();

  EXPECT_THAT(debug_info.files(), UnorderedElementsAre("dummy_file_alpha.cc",
                                                       "dummy_file_beta.cc"));
  EXPECT_THAT(debug_info.traces_size(), Eq(1));

  EXPECT_THAT(debug_info.traces().find("alpha_beta"),
              Ne(debug_info.traces().end()));
  auto actual_stack_trace = debug_info.traces().find("alpha_beta")->second;
  EXPECT_THAT(actual_stack_trace.file_line_cols_size(), Eq(2));
}

TEST(GraphDebugInfoBuilderTest, AccumulateStackTracesMap) {
  StackTracesMap stack_traces;
  stack_traces["two"] = std::make_shared<TestStackTrace>(
      std::vector<StackFrame>{{"dummy_file_alpha.cc", 20, "function_bar"},
                              {"dummy_file_beta.cc", 30, "function_sop"}});
  stack_traces["scale"] =
      std::make_shared<TestStackTrace>(std::vector<StackFrame>{
          {"dummy_file_alpha.cc", 10, "function_foo"},
          {"dummy_file_beta.cc", 30, "function_sop"},
      });
  stack_traces["y"] = std::make_shared<TestStackTrace>(std::vector<StackFrame>{
      {"dummy_file_alpha.cc", 15, "function_flex"},
      {"dummy_file_alpha.cc", 20, "function_bar"},
      {"dummy_file_beta.cc", 30, "function_sop"},
  });

  GraphDebugInfoBuilder builder;
  builder.AccumulateStackTracesMap(stack_traces, "@func");
  GraphDebugInfo debug_info = builder.Build();

  EXPECT_THAT(debug_info.files(), UnorderedElementsAre("dummy_file_alpha.cc",
                                                       "dummy_file_beta.cc"));
  EXPECT_THAT(debug_info.traces_size(), Eq(3));

  // Examine one of the three stack traces in detail.
  EXPECT_THAT(debug_info.traces().find("scale@func"),
              Ne(debug_info.traces().end()));
  auto stack_trace = debug_info.traces().find("scale@func")->second;
  EXPECT_THAT(stack_trace.file_line_cols_size(), Eq(2));

  // `FileLineCol.file_index` is non-deterministic because the GraphDebugInfo is
  // built by accumulating all file names into a set, and then storing that in
  // the `files` field in an arbitrary order.
  auto file_line_col_0 = stack_trace.file_line_cols(0);
  auto file_line_col_1 = stack_trace.file_line_cols(1);
  EXPECT_THAT(std::vector<int>(
                  {file_line_col_0.file_index(), file_line_col_1.file_index()}),
              UnorderedElementsAre(0, 1));
  EXPECT_THAT(file_line_col_0.line(), Eq(10));
  EXPECT_THAT(file_line_col_0.func(), Eq("function_foo"));
  EXPECT_THAT(file_line_col_1.line(), Eq(30));
  EXPECT_THAT(file_line_col_1.func(), Eq("function_sop"));
}

}  // namespace
}  // namespace tensorflow
