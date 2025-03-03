/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/hlo_proto_to_graph_view.h"

#include <variant>

#include <gmock/gmock.h>
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(GraphViewerParamsTest, GraphType) {
  // Default for graph type.
  ToolOptions options1;
  options1["type"] = "graph";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "graph");
  EXPECT_EQ(params1.node_name, "");
  EXPECT_EQ(params1.graph_width, 3);
  EXPECT_EQ(params1.render_options.show_backend_config, false);
  EXPECT_EQ(params1.render_options.show_fusion_subcomputations, true);
  EXPECT_EQ(params1.format, xla::RenderedGraphFormat::kUrl);

  // User defined options for graph type.
  ToolOptions options2;
  options2["type"] = "graph";
  options2["node_name"] = "fusion.111";
  options2["graph_width"] = 10;
  options2["show_metadata"] = 1;
  options2["merge_fusion"] = 1;
  options2["format"] = "html";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params2,
                          ParseGraphViewerParams(options2));
  EXPECT_EQ(params2.type, "graph");
  EXPECT_EQ(params2.node_name, "fusion.111");
  EXPECT_EQ(params2.graph_width, 10);
  EXPECT_EQ(params2.render_options.show_backend_config, true);
  EXPECT_EQ(params2.render_options.show_fusion_subcomputations, false);
  EXPECT_EQ(params2.format, xla::RenderedGraphFormat::kHtml);
}

TEST(GraphViewerParamsTest, ShortTxtType) {
  // Default for short txt type.
  ToolOptions options1;
  options1["type"] = "short_txt";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "short_txt");
  EXPECT_EQ(params1.verbose, false);
  EXPECT_EQ(params1.show_metadata, false);

  // User defined options for short txt type.
  ToolOptions options2;
  options2["type"] = "short_txt";
  options2["show_metadata"] = 1;
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params2,
                          ParseGraphViewerParams(options2));
  EXPECT_EQ(params2.type, "short_txt");
  EXPECT_EQ(params2.verbose, false);
  EXPECT_EQ(params2.show_metadata, true);
}

TEST(GraphViewerParamsTest, LongTxtType) {
  // Default for long txt type.
  ToolOptions options1;
  options1["type"] = "long_txt";
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params1,
                          ParseGraphViewerParams(options1));
  EXPECT_EQ(params1.type, "long_txt");
  EXPECT_EQ(params1.verbose, true);
  EXPECT_EQ(params1.show_metadata, false);

  // User defined options for long txt type.
  ToolOptions options2;
  options2["type"] = "long_txt";
  options2["show_metadata"] = 1;
  TF_ASSERT_OK_AND_ASSIGN(GraphViewerParams params2,
                          ParseGraphViewerParams(options2));
  EXPECT_EQ(params2.type, "long_txt");
  EXPECT_EQ(params2.verbose, true);
  EXPECT_EQ(params2.show_metadata, true);
}

TEST(GraphViewerParamsTest, OtherTypes) {
  ToolOptions options1;
  EXPECT_THAT(ParseGraphViewerParams(options1),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Graph viewer must provide a type option")));

  ToolOptions options2;
  options2["type"] = "abcd";
  EXPECT_THAT(ParseGraphViewerParams(options2),
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Unknown graph viewer type option: abcd")));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
