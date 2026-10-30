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

#include "xla/hlo/pass/hlo_pipeline_editor.h"

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;

// Temporaries would dangle.
static_assert(!std::is_constructible_v<HloPipelineEditor, HloPipelineProto&&>);
static_assert(!std::is_constructible_v<HloPipelineEditor,
                                       HloPipelineNodeSequenceProto&&>);

TEST(HloPipelineEditorTest, BuildsNestedPipeline) {
  HloPipelineProto proto;
  HloPipelineEditor root(proto);
  HloPipelineEditor spmd = root.AppendSequence("spmd");
  spmd.AppendPass("sharding-propagation");
  HloPipelineEditor simplification =
      root.AppendSequence("simplification", /*max_fixed_point_iterations=*/10);
  simplification.AppendPass("algsimp", {{"limit", 2}},
                            /*max_fixed_point_iterations=*/25);
  simplification.AppendPass("hlo-cse", {{"is_layout_sensitive", false}});
  root.AppendPass("hlo-dce");

  EXPECT_THAT(proto, EqualsProto(R"pb(
                root {
                  nodes {
                    name: "spmd"
                    sub_nodes { nodes { name: "sharding-propagation" } }
                  }
                  nodes {
                    name: "simplification"
                    max_fixed_point_iterations: 10
                    sub_nodes {
                      nodes {
                        name: "algsimp"
                        options {
                          key: "limit"
                          value { int_val: 2 }
                        }
                        max_fixed_point_iterations: 25
                      }
                      nodes {
                        name: "hlo-cse"
                        options {
                          key: "is_layout_sensitive"
                          value { bool_val: false }
                        }
                      }
                    }
                  }
                  nodes { name: "hlo-dce" }
                }
              )pb"));
}

TEST(HloPipelineEditorTest, AppendsPassWithNamedOptions) {
  HloPassOptions::ProtoMap map;
  map["flag"].set_bool_val(true);
  HloPassOptions options(map);
  HloPipelineNodeSequenceProto proto;
  HloPipelineEditor editor(proto);
  editor.AppendPass("first", options);
  editor.AppendPass("second", options);

  EXPECT_THAT(proto, EqualsProto(R"pb(
                nodes {
                  name: "first"
                  options {
                    key: "flag"
                    value { bool_val: true }
                  }
                }
                nodes {
                  name: "second"
                  options {
                    key: "flag"
                    value { bool_val: true }
                  }
                }
              )pb"));
}

TEST(HloPipelineEditorTest, EmptySequenceIsStillASequence) {
  HloPipelineNodeSequenceProto proto;
  (void)HloPipelineEditor(proto).AppendSequence("empty");

  ASSERT_EQ(proto.nodes_size(), 1);
  EXPECT_TRUE(proto.nodes(0).has_sub_nodes());
}

TEST(HloPipelineEditorTest, NestedEditorSurvivesParentGrowth) {
  HloPipelineNodeSequenceProto proto;
  HloPipelineEditor root(proto);
  HloPipelineEditor first = root.AppendSequence("first");
  for (int i = 0; i < 1000; ++i) {
    root.AppendPass("filler");
  }
  first.AppendPass("late");

  ASSERT_EQ(proto.nodes_size(), 1001);
  EXPECT_THAT(proto.nodes(0), EqualsProto(R"pb(
                name: "first"
                sub_nodes { nodes { name: "late" } }
              )pb"));
}

}  // namespace
}  // namespace xla
