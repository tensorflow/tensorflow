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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/platform/stack_frame.h"
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

  std::vector<StackFrame> ToUncachedFrames() const override { return frames_; }

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
  auto stack_trace = std::make_shared<TestStackTrace>(
      std::vector<StackFrame>{{"dummy_file_alpha.cc", 20, "function_bar"},
                              {"dummy_file_beta.cc", 30, "function_sop"}});

  GraphDebugInfoBuilder builder;
  builder.AccumulateStackTrace(stack_trace, "alpha_beta");
  GraphDebugInfo debug_info = builder.Build();

  EXPECT_THAT(debug_info.files(), UnorderedElementsAre("dummy_file_alpha.cc",
                                                       "dummy_file_beta.cc"));
  EXPECT_THAT(debug_info.traces_by_id_size(), Eq(1));

  EXPECT_THAT(debug_info.name_to_trace_id().find("alpha_beta"),
              Ne(debug_info.name_to_trace_id().end()));
  auto actual_stack_trace = debug_info.traces_by_id().at(
      debug_info.name_to_trace_id().at("alpha_beta"));
  EXPECT_THAT(actual_stack_trace.frame_id_size(), Eq(2))
      << debug_info.DebugString();
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
  EXPECT_THAT(debug_info.name_to_trace_id_size(), Eq(3));

  // Examine one of the three stack traces in detail.
  EXPECT_THAT(debug_info.name_to_trace_id().find("scale@func"),
              Ne(debug_info.name_to_trace_id().end()));
  auto stack_trace = debug_info.traces_by_id().at(
      debug_info.name_to_trace_id().at("scale@func"));
  EXPECT_THAT(stack_trace.frame_id_size(), Eq(2));

  std::vector<GraphDebugInfo::FileLineCol> file_line_cols;
  for (auto& frame_id : stack_trace.frame_id()) {
    file_line_cols.push_back(debug_info.frames_by_id().at(frame_id));
  }

  // `FileLineCol.file_index` is non-deterministic because the GraphDebugInfo is
  // built by accumulating all file names into a set, and then storing that in
  // the `files` field in an arbitrary order.
  auto file_line_col_0 = file_line_cols[0];
  auto file_line_col_1 = file_line_cols[1];
  EXPECT_THAT(std::vector<int>(
                  {file_line_col_0.file_index(), file_line_col_1.file_index()}),
              UnorderedElementsAre(0, 1));
  EXPECT_THAT(file_line_col_0.line(), Eq(10));
  EXPECT_THAT(file_line_col_0.func(), Eq("function_foo"));
  EXPECT_THAT(file_line_col_1.line(), Eq(30));
  EXPECT_THAT(file_line_col_1.func(), Eq("function_sop"));
}

TEST(GraphDebugInfoBuilderTest, AppendGraphDebugInfo) {
  GraphDebugInfo a;

  // Function stack traces are commonly returned without a prefix.
  // Validate that we can accumulate these correctly.
  {
    GraphDebugInfoBuilder builder;
    StackTracesMap stack_traces;
    stack_traces["two"] = std::make_shared<TestStackTrace>(
        std::vector<StackFrame>{{"dummy_file_alpha.cc", 20, "function_bar"}});
    stack_traces["scale"] = std::make_shared<TestStackTrace>(
        std::vector<StackFrame>{{"dummy_file_alpha.cc", 10, "function_foo"}});
    builder.AccumulateStackTracesMap(stack_traces, "");
    a = builder.Build();
  }

  GraphDebugInfo b;
  {
    GraphDebugInfoBuilder builder;
    StackTracesMap stack_traces;
    stack_traces["y"] =
        std::make_shared<TestStackTrace>(std::vector<StackFrame>{
            {"dummy_file_alpha.cc", 15, "function_flex"},
        });
    builder.AccumulateStackTracesMap(stack_traces, "");
    b = builder.Build();
  }

  // With builtin prefix
  GraphDebugInfo c;
  {
    GraphDebugInfoBuilder builder;
    StackTracesMap stack_traces;
    stack_traces["z"] =
        std::make_shared<TestStackTrace>(std::vector<StackFrame>{
            {"dummy_file_alpha.cc", 15, "function_flex"},
        });
    builder.AccumulateStackTracesMap(stack_traces, "@func3");
    c = builder.Build();
  }

  GraphDebugInfoBuilder builder;
  builder.AppendGraphDebugInfo("func1", a);
  builder.AppendGraphDebugInfo("func2", b);
  builder.AppendGraphDebugInfo("", c);
  GraphDebugInfo combined = builder.Build();

  EXPECT_EQ(combined.name_to_trace_id().size(), 4);
  std::vector<std::string> keys{"two@func1", "scale@func1", "y@func2",
                                "z@func3"};

  for (const auto& key : keys) {
    EXPECT_THAT(combined.name_to_trace_id().find(key),
                Ne(combined.name_to_trace_id().end()));
  }
}

TEST(StackTracesMapToGraphDebugInfoTest, EmptyMap) {
  StackTracesMap map;
  GraphDebugInfo generated = StackTracesMapToGraphDebugInfo(map);

  EXPECT_EQ(generated.files_size(), 0);
  EXPECT_EQ(generated.traces_size(), 0);
}

TEST(StackTracesMapToGraphDebugInfoTest, EmptyFrames) {
  StackTracesMap map;
  std::vector<StackFrame> frames;
  auto stack_trace = std::make_shared<FrozenStackTrace>(frames);
  map.insert({"dummy_name", stack_trace});
  GraphDebugInfo generated = StackTracesMapToGraphDebugInfo(map);

  EXPECT_EQ(generated.files_size(), 0);
  EXPECT_EQ(generated.traces_by_id_size(), 1);
  EXPECT_TRUE(generated.name_to_trace_id().contains("dummy_name"));
}

TEST(StackTracesMapToGraphDebugInfoTest, RoundTripStackTraces) {
  StackTracesMap map;
  std::vector<StackFrame> frames = {
      StackFrame({"dummy_file_name", 10, "dummy_function_name"}),
      StackFrame({"dummy_file_name", 20, "other_function_name"})};
  auto stack_trace = std::make_shared<FrozenStackTrace>(frames);
  map.insert({"dummy_name", stack_trace});
  GraphDebugInfo generated = StackTracesMapToGraphDebugInfo(map);

  StackTracesMap output = LoadTracesFromDebugInfo(generated);

  for (auto [name, trace] : output) {
    auto orig_trace = map[name];
    EXPECT_NE(orig_trace, nullptr);
    EXPECT_EQ(orig_trace->ToFrames(), trace->ToFrames());
  }
}

TEST(StackTracesTest, ToFrames) {
  StackTracesMap map;
  std::vector<StackFrame> frames = {
      StackFrame({"dummy_file_name", 10, "dummy_function_name"}),
      StackFrame({"other_file_name", 20, "other_function_name"})};
  auto stack_trace = TestStackTrace(frames);
  EXPECT_EQ(stack_trace.ToFrames().size(), 2);
  auto uncached_frames = stack_trace.ToUncachedFrames();
  EXPECT_EQ(uncached_frames.size(), 2);
  EXPECT_EQ(frames[0], uncached_frames[0]);
  EXPECT_EQ(frames[1], uncached_frames[1]);
}

}  // namespace
}  // namespace tensorflow
