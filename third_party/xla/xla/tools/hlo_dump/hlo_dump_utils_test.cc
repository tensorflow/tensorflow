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

#include "xla/tools/hlo_dump/hlo_dump_utils.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::numerics::debug_info {
namespace {

TEST(HloDumpUtilsTest, ConvertHloToHtmlBasic) {
  std::string hlo_text = "%foo = f32[10] add(%p0, %p1)";
  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;
  annotations[TensorKey::Create("foo", {})] =
      TensorAnnotation{"lightgreen", "\"my tooltip\""};

  std::string html = ConvertHloToHtml("test_dump", hlo_text, annotations);

  EXPECT_TRUE(absl::StrContains(html, "test_dump"));
  EXPECT_TRUE(absl::StrContains(html, "bg-lightgreen"));
  EXPECT_TRUE(absl::StrContains(html, "my tooltip"));
  EXPECT_TRUE(absl::StrContains(html, "f32"));
  EXPECT_TRUE(absl::StrContains(html, "["));
  EXPECT_TRUE(absl::StrContains(html, "10"));
  EXPECT_TRUE(absl::StrContains(html, "]"));
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlTuple) {
  std::string hlo_text = "%bar = (f32[5], s32[]) tuple(%p0, %p1)";
  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;
  annotations[TensorKey::Create("bar", {0})] =
      TensorAnnotation{"pink", "\"f32 tooltip\""};
  annotations[TensorKey::Create("bar", {1})] =
      TensorAnnotation{"yellow", "\"s32 tooltip\""};

  std::string html = ConvertHloToHtml("test_tuple", hlo_text, annotations);

  EXPECT_TRUE(absl::StrContains(html, "bg-pink"));
  EXPECT_TRUE(absl::StrContains(html, "f32 tooltip"));
  EXPECT_TRUE(absl::StrContains(html, "bg-yellow"));
  EXPECT_TRUE(absl::StrContains(html, "s32 tooltip"));
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlTupleWithLayout) {
  std::string hlo_text = "%bar = (f32[5], s32[]){:T(128)} tuple(%p0, %p1)";
  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;
  annotations[TensorKey::Create("bar", {0})] =
      TensorAnnotation{"pink", "\"f32 tooltip\""};
  annotations[TensorKey::Create("bar", {1})] =
      TensorAnnotation{"yellow", "\"s32 tooltip\""};

  std::string html =
      ConvertHloToHtml("test_tuple_layout", hlo_text, annotations);

  EXPECT_TRUE(absl::StrContains(html, "bg-pink"));
  EXPECT_TRUE(absl::StrContains(html, "f32 tooltip"));
  EXPECT_TRUE(absl::StrContains(html, "bg-yellow"));
  EXPECT_TRUE(absl::StrContains(html, "s32 tooltip"));
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlWithStats) {
  std::string hlo_text = "%root = f32[] parameter(0)";
  std::vector<std::pair<std::string, int64_t>> histogram = {{"pass1", 5},
                                                            {"pass2", 2}};

  OriginalValueRecoveryInfo recovery_info;
  recovery_info.percentage_recoverable = 80.0;
  recovery_info.percentage_recovered = 50.0;
  recovery_info.histogram = histogram;

  std::string html =
      ConvertHloToHtml("test_stats", hlo_text, {}, recovery_info);

  EXPECT_TRUE(absl::StrContains(html, "Recoverable tensors"));
  EXPECT_TRUE(absl::StrContains(html, "80.00%"));
  EXPECT_TRUE(absl::StrContains(html, "50.00%"));
  EXPECT_TRUE(absl::StrContains(html, "Tensors lost per pass:"));
  EXPECT_TRUE(absl::StrContains(html, "pass1"));
  EXPECT_TRUE(absl::StrContains(html, "5"));
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlRealWorld) {
  std::string hlo_text = R"hlo(HloModule jit_diverge,
    entry_computation_layout={(f32[]{:T(128)})->f32[]{:T(128)}},
    allow_spmd_sharding_propagation_to_parameters={true},
    allow_spmd_sharding_propagation_to_output={true}

FileNames
1 "experimental/users/tgeng/jax/tgeng_diverge.py"
2 "testing/pybase/googletest.py"
3 "third_party/py/absl/testing/absltest.py"
4 "third_party/py/absl/app.py"

FunctionNames
1 "<module>"
2 "main"
3 "_run_in_app"
4 "run"
5 "_run_main"
6 "_run_in_app.<locals>.main_function"
7 "RunTests"
8 "_run_and_get_tests_result"
9 "TgengDivergeTest.test_divergence"
10 "diverge"

FileLocations
1 {file_name_id=1 function_name_id=1 line=28 end_line=28 column=2 end_column=19}
2 {file_name_id=2 function_name_id=2 line=439 end_line=439 column=2 end_column=46}
3 {file_name_id=3 function_name_id=3 line=2448 end_line=2448 column=4 end_column=31}
4 {file_name_id=4 function_name_id=4 line=540 end_line=540 column=6 end_column=27}
5 {file_name_id=4 function_name_id=5 line=460 end_line=460 column=13 end_column=23}
6 {file_name_id=3 function_name_id=6 line=2446 end_line=2446 column=6 end_column=34}
7 {file_name_id=2 function_name_id=7 line=507 end_line=509 column=18 end_column=7}
8 {file_name_id=3 function_name_id=8 line=2902 end_line=2902 column=19 end_column=56}
9 {file_name_id=1 function_name_id=9 line=24 end_line=24 column=4 end_column=40}
10 {file_name_id=1 function_name_id=10 line=16 end_line=16 column=8 end_column=13}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=4}
5 {file_location_id=5 parent_frame_id=5}
6 {file_location_id=6 parent_frame_id=6}
7 {file_location_id=7 parent_frame_id=7}
8 {file_location_id=8 parent_frame_id=8}
9 {file_location_id=9 parent_frame_id=9}
10 {file_location_id=10 parent_frame_id=10}


ENTRY %main.1 (x.1: f32[]) -> f32[] {
  %x.1 = f32[] parameter(0),
    metadata={op_name="x"}
  %mul.10 = f32[] multiply(%x.1, %x.1),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.11 = f32[] multiply(%mul.10, %mul.10),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.12 = f32[] multiply(%mul.11, %mul.11),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.13 = f32[] multiply(%mul.12, %mul.12),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.14 = f32[] multiply(%mul.13, %mul.13),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.15 = f32[] multiply(%mul.14, %mul.14),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.16 = f32[] multiply(%mul.15, %mul.15),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.17 = f32[] multiply(%mul.16, %mul.16),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  %mul.18 = f32[] multiply(%mul.17, %mul.17),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
  ROOT %mul.19 = f32[] multiply(%mul.18, %mul.18),
    frontend_attributes={xla_log_for_comparison="true"},
    metadata={op_name="jit(diverge)/mul" stack_frame_id=10}
}
)hlo";

  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;
  annotations[TensorKey::Create("mul.10", {})] =
      TensorAnnotation{"lightgreen", "\"mul.10 tooltip\""};
  annotations[TensorKey::Create("mul.19", {})] =
      TensorAnnotation{"pink", "\"mul.19 tooltip\""};

  std::string html = ConvertHloToHtml("jit_diverge", hlo_text, annotations);

  EXPECT_TRUE(absl::StrContains(html, "jit_diverge"));
  EXPECT_TRUE(absl::StrContains(html, "bg-lightgreen"));
  EXPECT_TRUE(absl::StrContains(html, "mul.10 tooltip"));
  EXPECT_TRUE(absl::StrContains(html, "bg-pink"));
  EXPECT_TRUE(absl::StrContains(html, "mul.19 tooltip"));
  // Check that some other parts of the HLO are preserved.
  EXPECT_TRUE(
      absl::StrContains(html, "allow_spmd_sharding_propagation_to_parameters"));
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlArbitraryColors) {
  std::string hlo_text = "%foo = f32[10] add(%p0, %p1)";
  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;
  // #123456 quantizes to #113355
  annotations[TensorKey::Create("foo", {})] =
      TensorAnnotation{"#123456", "\"hex tooltip\""};

  std::string html = ConvertHloToHtml("test_hex", hlo_text, annotations);

  EXPECT_TRUE(
      absl::StrContains(html, ".bg-113355 { background-color: #113355; }"));
  EXPECT_TRUE(absl::StrContains(html, "class=\"kt bg-113355\""));
  EXPECT_TRUE(absl::StrContains(html, "hex tooltip"));
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlWithCompressedGraph) {
  std::string hlo_text = "%foo = f32[10] add(%p0, %p1)";
  GraphData graph_data;
  graph_data.nodes.push_back(GraphNode{0, 1.0, 2.0, 0.5, "foo", 0});
  graph_data.nodes.push_back(GraphNode{1, 2.0, 3.0, 0.6, "bar", 1});
  graph_data.edges.push_back(GraphEdge{0, 1});

  std::string html = ConvertHloToHtml("test_compressed_graph", hlo_text, {}, {},
                                      nullptr, &graph_data);

  EXPECT_TRUE(absl::StrContains(html, "window.compressedGraphData ="));
}

TEST(HloDumpUtilsTest, AnchorsAndLinks) {
  std::string hlo_text = R"hlo(
    @comp {
      %p0 = f32[] parameter(0)
      ROOT %neg = f32[] negate(%p0)
    }
    ENTRY @main {
      %c0 = f32[] constant(1)
      ROOT %call = f32[] call(%c0), to_apply=@comp
    }
  )hlo";
  std::string html = ConvertHloToHtml("test_links", hlo_text, {});

  // Check instruction anchors.
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_p0\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_neg\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_c0\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_call\""));

  // Check computation anchors.
  EXPECT_TRUE(absl::StrContains(html, "id=\"comp_comp\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"comp_main\""));

  // Check instruction links.
  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#instr_p0\"><span class=\"nv\">%p0</span></a>"));
  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#instr_c0\"><span class=\"nv\">%c0</span></a>"));

  // Check computation links (to_apply).
  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#comp_comp\"><span class=\"nc\">@comp</span></a>"));
}

TEST(HloDumpUtilsTest, AnchorsAndLinksNoPrefix) {
  // Verifies that even if % or @ are missing in the HLO text (matched as
  // kName), they still get anchors and links.
  std::string hlo_text = R"hlo(
    comp {
      p0 = f32[] parameter(0)
      ROOT neg = f32[] negate(p0)
    }
    ENTRY main {
      c0 = f32[] constant(1)
      ROOT call = f32[] call(c0), to_apply=comp
    }
  )hlo";
  std::string html = ConvertHloToHtml("test_links_no_prefix", hlo_text, {});

  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_p0\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_neg\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"comp_comp\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"comp_main\""));

  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#instr_p0\"><span class=\"n\">p0</span></a>"));
  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#comp_comp\"><span class=\"n\">comp</span></a>"));
}

TEST(HloDumpUtilsTest, TupleLinks) {
  std::string hlo_text = R"hlo(
    ENTRY %main {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %t = (f32[], f32[]) tuple(%p0, %p1)
    }
  )hlo";
  std::string html = ConvertHloToHtml("test_tuple_links", hlo_text, {});

  // Check instruction anchors for p0 and p1.
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_p0\""));
  EXPECT_TRUE(absl::StrContains(html, "id=\"instr_p1\""));

  // Check instruction links for p0 and p1 inside tuple.
  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#instr_p0\"><span class=\"nv\">%p0</span></a>"))
      << "Link for %p0 missing in: " << html;
  EXPECT_TRUE(absl::StrContains(
      html, "<a href=\"#instr_p1\"><span class=\"nv\">%p1</span></a>"))
      << "Link for %p1 missing in: " << html;
}

TEST(HloDumpUtilsTest, ConvertHloWithStackTraces) {
  std::string hlo_text = R"hlo(ENTRY %main {
  %p0 = f32[] parameter(0), metadata={stack_frame_id=1}
  ROOT %neg = f32[] negate(%p0), metadata={stack_frame_id=2}
})hlo";

  xla::StackFrameIndexProto sf_index;
  sf_index.add_file_names("file1.py");
  sf_index.add_function_names("func1");
  auto* loc = sf_index.add_file_locations();
  loc->set_file_name_id(1);
  loc->set_function_name_id(1);
  loc->set_line(10);
  auto* frame1 = sf_index.add_stack_frames();
  frame1->set_file_location_id(1);
  auto* frame2 = sf_index.add_stack_frames();
  frame2->set_file_location_id(1);
  frame2->set_parent_frame_id(1);

  std::string html = ConvertHloToHtml("test_st", hlo_text, {}, {}, &sf_index);

  // Check that data-stack-frame-id is present in the HTML.
  EXPECT_TRUE(absl::StrContains(html, "data-stack-frame-id="));
  EXPECT_TRUE(absl::StrContains(html, "1"));
  EXPECT_TRUE(absl::StrContains(html, "2"));

  // Check that the stack frame index is serialized into the JS.
  EXPECT_TRUE(absl::StrContains(html, "window.stackFrameIndex ="));
  EXPECT_TRUE(absl::StrContains(html, "\"file1.py\""));
  EXPECT_TRUE(absl::StrContains(html, "\"func1\""));
  // Check that the tooltip JS is present.
  EXPECT_TRUE(absl::StrContains(html, "No stack frame data found in HTML."));
}

TEST(HloDumpUtilsTest, MetadataStrippingAndOpNameExtraction) {
  std::string hlo_text = R"hlo(ENTRY %main {
  %p0 = f32[] parameter(0), metadata={op_name="my_op" stack_frame_id=1}
  %p1 = f32[] parameter(1), frontend_attributes={xla_log_for_comparison="true"}, metadata={op_name="other_op"}
})hlo";

  std::string html = ConvertHloToHtml("test_metadata", hlo_text, {});

  // Should have data-op-name for my_op
  EXPECT_TRUE(absl::StrContains(html, "data-op-name=\"my_op\""));
  // Should have data-stack-frame-id for my_op
  EXPECT_TRUE(absl::StrContains(html, "data-stack-frame-id=\"1\""));

  // Should have data-op-name for other_op
  EXPECT_TRUE(absl::StrContains(html, "data-op-name=\"other_op\""));

  // Should NOT have metadata= in the rendered HLO
  EXPECT_FALSE(absl::StrContains(html, "metadata="));

  // Should preserve other attributes
  EXPECT_TRUE(absl::StrContains(html, "xla_log_for_comparison"));
}

TEST(HloDumpUtilsTest, DictionaryStripping) {
  std::string hlo_text = R"hlo(HloModule foo

FileNames
1 "test.py"
2 "other.py"

FunctionNames
1 "main"
2 "helper"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10}

StackFrames
1 {file_location_id=1}

ENTRY %main {
  %p0 = f32[] parameter(0)
}
)hlo";

  std::string html = ConvertHloToHtml("test_dict_strip", hlo_text, {});

  // Should NOT have FileNames, FunctionNames, FileLocations, StackFrames
  EXPECT_FALSE(absl::StrContains(html, "FileNames"));
  EXPECT_FALSE(absl::StrContains(html, "FunctionNames"));
  EXPECT_FALSE(absl::StrContains(html, "FileLocations"));
  EXPECT_FALSE(absl::StrContains(html, "StackFrames"));

  // Should NOT have the content of the dicts
  EXPECT_FALSE(absl::StrContains(html, "\"test.py\""));
  EXPECT_FALSE(absl::StrContains(html, "\"main\""));

  // Should have the rest of the module
  EXPECT_TRUE(absl::StrContains(html, "HloModule"));
  EXPECT_TRUE(absl::StrContains(html, "foo"));
  EXPECT_TRUE(absl::StrContains(html, "ENTRY"));
}

TEST(HloDumpUtilsTest, PopulateMismatchAnnotations_Basic) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_module
ENTRY main {
  p0 = f32[10] parameter(0)
  ROOT add = f32[10] add(p0, p0)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "add";
  details.actual = 1.0;
  details.expected = 2.0;
  details.rel_error = 0.5;
  details.percentage_of_elems_exceeding_abs_error = 50.0;
  details.percentage_of_elems_exceeding_rel_error = 50.0;
  details.percentage_of_elems_exceeding_both_errors = 50.0;
  details.result_of_reduce = false;

  auto annotations = PopulateMismatchAnnotations(*module, {details});
  EXPECT_FALSE(annotations.empty());
  auto root_key = TensorKey::Create("add", ShapeIndex{});
  ASSERT_TRUE(annotations.contains(root_key));
  EXPECT_EQ(annotations[root_key].background_color, "pink");
  EXPECT_TRUE(annotations[root_key].tooltip_data.has_value());
  EXPECT_TRUE(absl::StrContains(*annotations[root_key].tooltip_data,
                                "Numeric Mismatch"));
  EXPECT_TRUE(
      absl::StrContains(*annotations[root_key].tooltip_data, "Actual: 1"));
  EXPECT_TRUE(
      absl::StrContains(*annotations[root_key].tooltip_data, "Expected: 2"));
  EXPECT_TRUE(
      absl::StrContains(*annotations[root_key].tooltip_data, "Rel Error: 0.5"));
  EXPECT_TRUE(absl::StrContains(*annotations[root_key].tooltip_data,
                                "Elems exceeding abs error: 50.00%"));
  EXPECT_TRUE(absl::StrContains(*annotations[root_key].tooltip_data,
                                "Elems exceeding rel error: 50.00%"));
  EXPECT_TRUE(absl::StrContains(*annotations[root_key].tooltip_data,
                                "Elems exceeding both errors: 50.00%"));
  EXPECT_TRUE(absl::StrContains(*annotations[root_key].tooltip_data,
                                "Result of reduce: False"));
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_Basic) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_module
ENTRY main {
  p0 = f32[10] parameter(0)
  ROOT add = f32[10] add(p0, p0)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "add";
  details.actual = 1.0;
  details.expected = 2.0;
  details.rel_error = 0.5;
  details.percentage_of_elems_exceeding_abs_error = 50.0;
  details.percentage_of_elems_exceeding_rel_error = 50.0;

  auto graph_data = PopulateMismatchGraphData(*module, {details});
  EXPECT_FALSE(graph_data.nodes.empty());
  EXPECT_FALSE(graph_data.edges.empty());

  bool found_add = false;
  bool found_p0 = false;
  for (const auto& node : graph_data.nodes) {
    if (node.key == "add") {
      found_add = true;
      EXPECT_EQ(node.diff_score, 50.0);
    } else if (node.key == "p0") {
      found_p0 = true;
      EXPECT_EQ(node.diff_score, 0.0);
    }
  }
  EXPECT_TRUE(found_add);
  EXPECT_TRUE(found_p0);
}

TEST(HloDumpUtilsTest, PopulateMismatchAnnotations_TupleAndReduce) {
  const absl::string_view hlo_string = R"hlo(
HloModule tuple_module

fused_computation {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[10] parameter(0)
  add = f32[10] add(p0, p0)
  zero = f32[] constant(0)
  reduce = f32[] reduce(p0, zero), dimensions={0}, to_apply=fused_computation
  ROOT tuple = (f32[10], f32[]) tuple(add, reduce)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "tuple";
  details.output_shape_index = 1;
  details.actual = 1.0;
  details.expected = 2.0;
  details.rel_error = 0.5;
  details.percentage_of_elems_exceeding_abs_error = 100.0;
  details.percentage_of_elems_exceeding_rel_error = 100.0;
  details.result_of_reduce = true;

  auto annotations = PopulateMismatchAnnotations(*module, {details});
  auto root_key = TensorKey::Create("tuple", ShapeIndex{1});
  ASSERT_TRUE(annotations.contains(root_key));
  EXPECT_EQ(annotations[root_key].background_color, "pink");
  EXPECT_TRUE(annotations[root_key].tooltip_data.has_value());
  EXPECT_TRUE(absl::StrContains(*annotations[root_key].tooltip_data,
                                "Result of reduce: True"));
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_FusionHierarchy) {
  const absl::string_view hlo_string = R"hlo(
HloModule fusion_module

fused_comp {
  p0.1 = f32[10] parameter(0)
  ROOT add.1 = f32[10] add(p0.1, p0.1)
}

ENTRY main {
  p0 = f32[10] parameter(0)
  ROOT fusion = f32[10] fusion(p0), kind=kLoop, calls=fused_comp
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "fusion";
  details.actual = 1.0;
  details.expected = 2.0;
  details.rel_error = 0.25;
  details.percentage_of_elems_exceeding_abs_error = 50.0;
  details.percentage_of_elems_exceeding_rel_error = 50.0;

  auto annotations = PopulateMismatchAnnotations(*module, {details});
  auto graph_data = PopulateMismatchGraphData(*module, {details});

  auto outer_key = TensorKey::Create("fusion", ShapeIndex{});
  auto inner_key = TensorKey::Create("add.1", ShapeIndex{});
  ASSERT_TRUE(annotations.contains(outer_key));
  ASSERT_FALSE(annotations.contains(inner_key));
  EXPECT_EQ(annotations[outer_key].background_color, "pink");

  bool found_fusion = false;
  bool found_add1 = false;
  for (const auto& node : graph_data.nodes) {
    if (node.key == "fusion") {
      found_fusion = true;
      EXPECT_EQ(node.diff_score, 25.0);
    } else if (node.key == "fusion/add.1") {
      found_add1 = true;
      EXPECT_EQ(node.diff_score, 0.0);
    }
  }
  EXPECT_TRUE(found_fusion);
  EXPECT_TRUE(found_add1);
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_NestedFusionHierarchy) {
  const absl::string_view hlo_string = R"hlo(
HloModule nested_fusion_module

fused_comp_0 {
  p0.2 = f32[10] parameter(0)
  mul = f32[10] multiply(p0.2, p0.2)
  ROOT add.1 = f32[10] add(mul, p0.2)
}

fused_comp_1 {
  p0.1 = f32[10] parameter(0)
  ROOT inner_fusion = f32[10] fusion(p0.1), kind=kLoop, calls=fused_comp_0
}

ENTRY main {
  p0 = f32[10] parameter(0)
  ROOT outer_fusion = f32[10] fusion(p0), kind=kLoop, calls=fused_comp_1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "mul";
  details.actual = 1.0;
  details.expected = 2.0;
  details.rel_error = 0.25;
  details.percentage_of_elems_exceeding_abs_error = 50.0;
  details.percentage_of_elems_exceeding_rel_error = 50.0;

  auto annotations = PopulateMismatchAnnotations(*module, {details});
  auto graph_data = PopulateMismatchGraphData(*module, {details});

  auto outer_key = TensorKey::Create("outer_fusion", ShapeIndex{});
  auto inner_key = TensorKey::Create("add.1", ShapeIndex{});
  auto mul_key = TensorKey::Create("mul", ShapeIndex{});
  ASSERT_FALSE(annotations.contains(outer_key));
  ASSERT_FALSE(annotations.contains(inner_key));
  ASSERT_TRUE(annotations.contains(mul_key));

  bool found_outer = false;
  bool found_add1 = false;
  bool found_mul = false;
  for (const auto& node : graph_data.nodes) {
    if (node.key == "outer_fusion") {
      found_outer = true;
      EXPECT_EQ(node.diff_score, 0.0);
    } else if (node.key == "outer_fusion/inner_fusion/add.1") {
      found_add1 = true;
      EXPECT_EQ(node.diff_score, 0.0);
    } else if (node.key == "outer_fusion/inner_fusion/mul") {
      found_mul = true;
      EXPECT_EQ(node.diff_score, 25.0);
    }
  }
  EXPECT_TRUE(found_outer);
  EXPECT_TRUE(found_add1);
  EXPECT_TRUE(found_mul);
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_SingleOpInNestedFusion) {
  const absl::string_view hlo_string = R"hlo(
HloModule nested_fusion_module

fused_comp_0 {
  p0.2 = f32[10] parameter(0)
  mul = f32[10] multiply(p0.2, p0.2)
  ROOT add.1 = f32[10] add(mul, p0.2)
}

fused_comp_1 {
  p0.1 = f32[10] parameter(0)
  ROOT inner_fusion = f32[10] fusion(p0.1), kind=kLoop, calls=fused_comp_0
}

ENTRY main {
  p0 = f32[10] parameter(0)
  ROOT outer_fusion = f32[10] fusion(p0), kind=kLoop, calls=fused_comp_1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "mul";
  details.actual = 1.0;
  details.expected = 2.0;
  details.rel_error = 0.25;
  details.percentage_of_elems_exceeding_abs_error = 50.0;
  details.percentage_of_elems_exceeding_rel_error = 50.0;

  auto annotations = PopulateMismatchAnnotations(*module, {details});
  auto graph_data = PopulateMismatchGraphData(*module, {details});

  auto mul_key = TensorKey::Create("mul", ShapeIndex{});
  auto outer_key = TensorKey::Create("outer_fusion", ShapeIndex{});
  auto inner_key = TensorKey::Create("inner_fusion", ShapeIndex{});
  auto add_key = TensorKey::Create("add.1", ShapeIndex{});

  ASSERT_TRUE(annotations.contains(mul_key));
  EXPECT_EQ(annotations[mul_key].background_color, "pink");
  EXPECT_FALSE(annotations.contains(outer_key));
  EXPECT_FALSE(annotations.contains(inner_key));
  EXPECT_FALSE(annotations.contains(add_key));

  bool found_mul = false;
  for (const auto& node : graph_data.nodes) {
    if (node.key == "outer_fusion/inner_fusion/mul") {
      found_mul = true;
      EXPECT_EQ(node.diff_score, 25.0);
    } else if (node.key == "outer_fusion" ||
               node.key == "outer_fusion/inner_fusion/add.1") {
      EXPECT_EQ(node.diff_score, 0);
    }
  }
  EXPECT_TRUE(found_mul);
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_ZeroRelError) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_module
ENTRY main {
  p0 = f32[10] parameter(0)
  ROOT add = f32[10] add(p0, p0)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails details;
  details.target_instruction_name = "add";
  details.actual = 1.0;
  details.expected = 1.0;
  details.rel_error = 0.0;
  details.percentage_of_elems_exceeding_abs_error = 0.0;
  details.percentage_of_elems_exceeding_rel_error = 0.0;

  auto graph_data = PopulateMismatchGraphData(*module, {details});
  bool found_add = false;
  for (const auto& node : graph_data.nodes) {
    if (node.key == "add") {
      found_add = true;
      EXPECT_EQ(node.diff_score, 100.0);
    }
  }
  EXPECT_TRUE(found_add);
}

TEST(HloDumpUtilsTest, ConvertHloToHtmlCompactGte) {
  std::string hlo_text =
      "ENTRY %test {\n"
      "  %p0 = (f32[10], f16[10]) parameter(0)\n"
      "  ROOT %root = f32[10] add(f32[10] %p0#0, f32[10] %p0#0)\n"
      "}\n";

  absl::flat_hash_map<TensorKey, TensorAnnotation> annotations;
  annotations[TensorKey::Create("p0_0", {})] =
      TensorAnnotation{"lightgreen", "\"my compact GTE tooltip\""};

  std::string html =
      ConvertHloToHtml("test_compact_gte", hlo_text, annotations);

  // Verify that the tooltip data is correctly injected
  EXPECT_TRUE(absl::StrContains(html, "my compact GTE tooltip"));

  // Verify that the color light green is associated with the element
  EXPECT_TRUE(absl::StrContains(html, "bg-lightgreen"));

  // Verify that the link is correctly sanitized to point to the GTE
  EXPECT_TRUE(absl::StrContains(html, "href=\"#instr_p0_0\""));
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_RuntimeNanMismatch) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_runtime_nan
ENTRY main {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT div = f32[10] divide(p0, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails nan_mismatch;
  nan_mismatch.target_instruction_name = "div";
  nan_mismatch.actual = std::numeric_limits<double>::quiet_NaN();
  nan_mismatch.expected = 1.0;
  nan_mismatch.rel_error = std::numeric_limits<double>::quiet_NaN();

  auto graph_data = PopulateMismatchGraphData(*module, {nan_mismatch});
  absl::flat_hash_map<std::string, double> scores;
  for (const auto& node : graph_data.nodes) {
    scores[node.key] = node.diff_score;
  }

  EXPECT_EQ(scores["div"], kNanInfMismatchDiffScore);
  EXPECT_EQ(scores["p0"], 0.0);
  EXPECT_EQ(scores["p1"], 0.0);
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_RuntimeInfMismatch) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_runtime_inf
ENTRY main {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT div = f32[10] divide(p0, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails inf_mismatch;
  inf_mismatch.target_instruction_name = "div";
  inf_mismatch.actual = std::numeric_limits<double>::infinity();
  inf_mismatch.expected = 1.0;
  inf_mismatch.rel_error = std::numeric_limits<double>::infinity();

  auto graph_data = PopulateMismatchGraphData(*module, {inf_mismatch});
  absl::flat_hash_map<std::string, double> scores;
  for (const auto& node : graph_data.nodes) {
    scores[node.key] = node.diff_score;
  }

  EXPECT_EQ(scores["div"], kNanInfMismatchDiffScore);
  EXPECT_EQ(scores["p0"], 0.0);
  EXPECT_EQ(scores["p1"], 0.0);
}

TEST(HloDumpUtilsTest, PopulateMismatchGraphData_RuntimeNanMismatchInFusion) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_fusion_nan
%fused_comp (p0: f32[10]) -> f32[10] {
  %p0 = f32[10] parameter(0)
  %c = f32[] constant(0.0)
  %b = f32[10] broadcast(%c), dimensions={}
  ROOT %div = f32[10] divide(%p0, %b)
}

ENTRY main {
  %p0 = f32[10] parameter(0)
  ROOT %fusion = f32[10] fusion(%p0), kind=kCustom, calls=%fused_comp
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails nan_mismatch;
  nan_mismatch.target_instruction_name = "div";
  nan_mismatch.actual = std::numeric_limits<double>::quiet_NaN();
  nan_mismatch.expected = 1.0;
  nan_mismatch.rel_error = std::numeric_limits<double>::quiet_NaN();

  auto graph_data = PopulateMismatchGraphData(*module, {nan_mismatch});
  absl::flat_hash_map<std::string, double> scores;
  for (const auto& node : graph_data.nodes) {
    scores[node.key] = node.diff_score;
  }

  EXPECT_EQ(scores["fusion"], 0.0);
  EXPECT_EQ(scores["fusion/div"], kNanInfMismatchDiffScore);
  EXPECT_EQ(scores["fusion/c"], 0.0);
  EXPECT_EQ(scores["p0"], 0.0);
}

TEST(HloDumpUtilsTest, PopulateMismatchAnnotations_NonTupleOpInsideFusion) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_fusion_annotation
%fused_comp (p0: f32[10]) -> f32[10] {
  %p0 = f32[10] parameter(0)
  ROOT %sin.0 = f32[10] sine(%p0)
}

ENTRY main {
  %p0 = f32[10] parameter(0)
  ROOT %fusion = f32[10] fusion(%p0), kind=kLoop, calls=%fused_comp
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails mismatch;
  mismatch.target_instruction_name = "sin.0";
  mismatch.output_shape_index = 0;  // Default shape index for non-tuple
  mismatch.actual = 1.0;
  mismatch.expected = 2.0;
  mismatch.rel_error = 0.5;

  auto annotations = PopulateMismatchAnnotations(*module, {mismatch});
  TensorKey expected_key = TensorKey::Create("sin.0", ShapeIndex{});
  ASSERT_TRUE(annotations.contains(expected_key));
  EXPECT_EQ(annotations[expected_key].background_color, "pink");
  EXPECT_TRUE(annotations[expected_key].tooltip_data.has_value());

  std::string html =
      ConvertHloToHtml(module->name(), module->ToString(), annotations);
  EXPECT_TRUE(absl::StrContains(html, "bg-pink"));
  EXPECT_TRUE(absl::StrContains(html, "Numeric Mismatch:"));
  EXPECT_TRUE(absl::StrContains(html, "sin.0"));
}

TEST(HloDumpUtilsTest, ComputeBoundingBoxCleanMaskReturnsZeroMismatches) {
  Literal mask = LiteralUtil::CreateR2<bool>({
      {false, false, false},
      {false, false, false},
  });
  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.mismatch_count, 0);
  EXPECT_EQ(box.total_elements, 6);
}

TEST(HloDumpUtilsTest, ComputeBoundingBox1D) {
  std::array<bool, 16> values{};
  values[3] = true;
  values[10] = true;
  Literal mask = LiteralUtil::CreateR1<bool>(absl::MakeConstSpan(values));
  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.box_min, std::vector<int64_t>{3});
  EXPECT_EQ(box.box_max, std::vector<int64_t>{10});
  EXPECT_EQ(box.mismatch_count, 2);
  EXPECT_EQ(box.total_elements, 16);
}

TEST(HloDumpUtilsTest, ComputeBoundingBox2D) {
  Literal mask = LiteralUtil::CreateR2<bool>({
      {false, false, false, false, false},
      {false, false, true, false, false},
      {false, false, false, false, false},
      {false, true, false, false, false},
  });
  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.box_min, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(box.box_max, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(box.mismatch_count, 2);
  EXPECT_EQ(box.total_elements, 20);
}

TEST(HloDumpUtilsTest, ComputeBoundingBox3D) {
  Literal mask = LiteralUtil::CreateR3<bool>({
      {
          {false, false, false, false},
          {false, false, false, false},
          {false, false, false, false},
      },
      {
          {false, false, true, false},
          {false, false, false, false},
          {false, false, false, false},
      },
  });
  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.box_min, (std::vector<int64_t>{1, 0, 2}));
  EXPECT_EQ(box.box_max, (std::vector<int64_t>{1, 0, 2}));
  EXPECT_EQ(box.mismatch_count, 1);
  EXPECT_EQ(box.total_elements, 24);
}

TEST(HloDumpUtilsTest, ComputeBoundingBoxScalar0D) {
  Literal clean_scalar = LiteralUtil::CreateR0<bool>(false);
  MismatchBoundingBox clean_box =
      ComputeBoundingBoxFromLiteralMask(clean_scalar);
  EXPECT_EQ(clean_box.mismatch_count, 0);
  EXPECT_EQ(clean_box.total_elements, 1);
  EXPECT_TRUE(clean_box.box_min.empty());
  EXPECT_TRUE(clean_box.box_max.empty());

  Literal mismatch_scalar = LiteralUtil::CreateR0<bool>(true);
  MismatchBoundingBox mismatch_box =
      ComputeBoundingBoxFromLiteralMask(mismatch_scalar);
  EXPECT_EQ(mismatch_box.mismatch_count, 1);
  EXPECT_EQ(mismatch_box.total_elements, 1);
  EXPECT_TRUE(mismatch_box.box_min.empty());
  EXPECT_TRUE(mismatch_box.box_max.empty());
  ASSERT_EQ(mismatch_box.top_mismatch_coords.size(), 1);
  EXPECT_TRUE(mismatch_box.top_mismatch_coords[0].empty());
}

TEST(HloDumpUtilsTest, ComputeBoundingBox4D) {
  Shape shape = ShapeUtil::MakeShape(PRED, {2, 3, 4, 5});
  Literal mask(shape);
  mask.PopulateWithValue(false);
  mask.Set<bool>({0, 1, 2, 3}, true);
  mask.Set<bool>({1, 2, 0, 4}, true);

  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.mismatch_count, 2);
  EXPECT_EQ(box.total_elements, 120);
  EXPECT_EQ(box.box_min, (std::vector<int64_t>{0, 1, 0, 3}));
  EXPECT_EQ(box.box_max, (std::vector<int64_t>{1, 2, 2, 4}));
  EXPECT_GE(box.top_mismatch_coords.size(), 2);
}

TEST(HloDumpUtilsTest, ComputeBoundingBoxMultipleCoordsInRow) {
  Shape shape = ShapeUtil::MakeShape(PRED, {16});
  Literal mask(shape);
  mask.PopulateWithValue(false);
  mask.Set<bool>({1}, true);
  mask.Set<bool>({3}, true);
  mask.Set<bool>({5}, true);
  mask.Set<bool>({7}, true);
  mask.Set<bool>({9}, true);
  mask.Set<bool>({11}, true);

  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.mismatch_count, 6);
  EXPECT_EQ(box.box_min, (std::vector<int64_t>{1}));
  EXPECT_EQ(box.box_max, (std::vector<int64_t>{11}));
  EXPECT_EQ(box.top_mismatch_coords.size(), 6);
  EXPECT_EQ(box.top_mismatch_coords[0], (std::vector<int64_t>{1}));
  EXPECT_EQ(box.top_mismatch_coords[1], (std::vector<int64_t>{3}));
  EXPECT_EQ(box.top_mismatch_coords[2], (std::vector<int64_t>{5}));
  EXPECT_EQ(box.top_mismatch_coords[3], (std::vector<int64_t>{7}));
  EXPECT_EQ(box.top_mismatch_coords[4], (std::vector<int64_t>{9}));
  EXPECT_EQ(box.top_mismatch_coords[5], (std::vector<int64_t>{11}));
  EXPECT_EQ(box.pattern, "STRIDED (stride 2)");
}

MismatchBoundingBox MakeBoundingBox(
    std::vector<int64_t> shape, std::vector<int64_t> box_min,
    std::vector<int64_t> box_max, int64_t mismatch_count,
    int64_t total_elements, std::string pattern = "",
    std::vector<std::vector<int64_t>> top_coords = {}) {
  MismatchBoundingBox bbox;
  bbox.tensor_shape = std::move(shape);
  bbox.box_min = std::move(box_min);
  bbox.box_max = std::move(box_max);
  bbox.mismatch_count = mismatch_count;
  bbox.total_elements = total_elements;
  bbox.pattern = std::move(pattern);
  bbox.top_mismatch_coords = std::move(top_coords);
  return bbox;
}

MismatchDetails CreateSampleMismatch(std::string instruction_name,
                                     std::optional<int64_t> output_shape_index,
                                     double actual, double expected,
                                     double rel_error,
                                     MismatchBoundingBox bounding_box) {
  MismatchDetails m;
  m.target_instruction_name = std::move(instruction_name);
  m.output_shape_index = output_shape_index;
  m.actual = actual;
  m.expected = expected;
  m.rel_error = rel_error;
  m.bounding_box = std::move(bounding_box);
  return m;
}

void ExpectSliceBox(
    const absl::flat_hash_map<std::string, SliceBoundingBox>& slice_boxes,
    absl::string_view key, const std::vector<int64_t>& expected_min,
    const std::vector<int64_t>& expected_max, int64_t expected_count) {
  SCOPED_TRACE(key);
  ASSERT_TRUE(slice_boxes.contains(key));
  const auto& sbox = slice_boxes.at(key);
  EXPECT_EQ(sbox.box_min, expected_min);
  EXPECT_EQ(sbox.box_max, expected_max);
  EXPECT_EQ(sbox.mismatch_count, expected_count);
}

TEST(HloDumpUtilsTest, ClassifyMismatchPatternTests) {
  // 1. Dense Block (density >= 70%)
  EXPECT_EQ(ClassifyMismatchPattern(
                MakeBoundingBox({10, 10}, {0, 0}, {9, 9}, 85, 100)),
            "DENSE_BLOCK");

  // 2. Sparse Outliers (density < 2% and count < 16)
  EXPECT_EQ(ClassifyMismatchPattern(
                MakeBoundingBox({100, 100}, {0, 0}, {99, 99}, 5, 10000)),
            "SPARSE_OUTLIERS");

  // 3. Strided (periodic delta between mismatch coordinates in minor dimension)
  EXPECT_EQ(ClassifyMismatchPattern(
                MakeBoundingBox({8, 32}, {0, 0}, {7, 24}, 24, 256,
                                /*pattern=*/"", {{1, 8}, {1, 16}, {2, 24}})),
            "STRIDED (stride 8)");

  // 4. Boundary (errors concentrated at min and max bounds with hollow
  // interior)
  EXPECT_EQ(ClassifyMismatchPattern(MakeBoundingBox(
                {10, 20}, {0, 2}, {9, 18}, 20, 200, /*pattern=*/"",
                {{0, 5}, {9, 5}, {0, 6}, {9, 6}})),
            "BOUNDARY");

  // 5. Single Slice (span is 1 along one or more dimensions)
  EXPECT_EQ(
      ClassifyMismatchPattern(MakeBoundingBox(
          {8, 16}, {3, 2}, {3, 10}, 8, 128, /*pattern=*/"", {{3, 2}, {3, 3}})),
      "SINGLE_SLICE");

  // 6. Boundary with coordinates that have column delta >= 2 (must remain
  // BOUNDARY, not STRIDED)
  EXPECT_EQ(ClassifyMismatchPattern(MakeBoundingBox(
                {10, 20}, {0, 4}, {9, 12}, 20, 200, /*pattern=*/"",
                {{0, 4}, {9, 4}, {0, 12}, {9, 12}})),
            "BOUNDARY");

  // 7. Sparse outliers where points have coordinate delta >= 2 (must remain
  // SPARSE_OUTLIERS, not STRIDED)
  EXPECT_EQ(ClassifyMismatchPattern(
                MakeBoundingBox({100, 100}, {0, 4}, {99, 12}, 2, 10000,
                                /*pattern=*/"", {{0, 4}, {99, 12}})),
            "SPARSE_OUTLIERS");

  // 8. Irregular deltas with common even factor (e.g. 100 and 2) must not
  // trigger STRIDED
  EXPECT_NE(ClassifyMismatchPattern(
                MakeBoundingBox({10, 200}, {1, 0}, {5, 102}, 30, 2000,
                                /*pattern=*/"", {{1, 0}, {2, 100}, {3, 102}})),
            "STRIDED (stride 2)");

  // 9. Scalar tensor (rank 0) with mismatch
  EXPECT_EQ(ClassifyMismatchPattern(MakeBoundingBox({}, {}, {}, 1, 1)),
            "DENSE_BLOCK");

  // 10. Empty / 0 mismatch count returns empty string
  EXPECT_EQ(ClassifyMismatchPattern(
                MakeBoundingBox({10, 10}, {0, 0}, {0, 0}, 0, 100)),
            "");
}

TEST(HloDumpUtilsTest, ComputeBoundingBoxFromLiteralMask_4DMismatchedSlices) {
  Literal mask = LiteralUtil::CreateR4<bool>({
      {{{true, false}, {false, false}}, {{false, false}, {false, false}}},
      {{{false, false}, {false, false}}, {{false, false}, {false, false}}},
      {{{false, false}, {false, true}}, {{false, false}, {false, false}}},
      {{{false, false}, {false, false}}, {{false, false}, {false, false}}},
  });
  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.mismatch_count, 2);
  EXPECT_EQ(box.box_min, (std::vector<int64_t>{0, 0, 0, 0}));
  EXPECT_EQ(box.box_max, (std::vector<int64_t>{2, 0, 1, 1}));
  EXPECT_EQ(box.mismatched_slices, (std::vector<int64_t>{0, 2}));
  ASSERT_EQ(box.slice_boxes.size(), 2);
  ExpectSliceBox(box.slice_boxes, "0", {0, 0, 0, 0}, {0, 0, 0, 0}, 1);
  EXPECT_EQ(box.slice_boxes.at("0").slice_index, 0);
  EXPECT_EQ(box.slice_boxes.at("0").slice_key, "0");

  ExpectSliceBox(box.slice_boxes, "2", {2, 0, 1, 1}, {2, 0, 1, 1}, 1);
  EXPECT_EQ(box.slice_boxes.at("2").slice_index, 2);
  EXPECT_EQ(box.slice_boxes.at("2").slice_key, "2");
}

TEST(HloDumpUtilsTest, ComputeBoundingBoxFromLiteralMask_5DMultiSliceKeying) {
  Shape shape = ShapeUtil::MakeShape(PRED, {2, 3, 4, 8, 16});
  Literal mask(shape);
  mask.PopulateWithValue(false);
  // Two mismatches in distinct outer batches (dim 0 = 0 vs 1) but sharing dim 1
  // = 2 (which is rank - 4).
  mask.Set<bool>({0, 2, 1, 2, 3}, true);
  mask.Set<bool>({1, 2, 2, 4, 6}, true);

  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.mismatch_count, 2);
  EXPECT_EQ(box.box_min, (std::vector<int64_t>{0, 2, 1, 2, 3}));
  EXPECT_EQ(box.box_max, (std::vector<int64_t>{1, 2, 2, 4, 6}));

  ASSERT_EQ(box.slice_boxes.size(), 2);
  ExpectSliceBox(box.slice_boxes, "0,2", {0, 2, 1, 2, 3}, {0, 2, 1, 2, 3}, 1);
  EXPECT_EQ(box.slice_boxes.at("0,2").slice_key, "0,2");
  EXPECT_EQ(box.slice_boxes.at("0,2").slice_coords,
            (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(box.slice_boxes.at("0,2").slice_index, 2);

  ExpectSliceBox(box.slice_boxes, "1,2", {1, 2, 2, 4, 6}, {1, 2, 2, 4, 6}, 1);
  EXPECT_EQ(box.slice_boxes.at("1,2").slice_key, "1,2");
  EXPECT_EQ(box.slice_boxes.at("1,2").slice_coords,
            (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(box.slice_boxes.at("1,2").slice_index, 2);
}

TEST(HloDumpUtilsTest, ComputeBoundingBoxFromLiteralMask_ReservoirSampling) {
  Shape shape = ShapeUtil::MakeShape(PRED, {100, 32});
  Literal mask(shape);
  mask.PopulateWithValue(false);

  // Set 17 mismatches across rows 0-3 (exceeding reservoir capacity k=16).
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 5; ++c) {
      mask.Set<bool>({r, c}, true);
    }
  }
  mask.Set<bool>({3, 0}, true);
  mask.Set<bool>({3, 1}, true);

  // Set 15 mismatches across later rows (rows 50, 90, 99).
  for (int c = 0; c < 5; ++c) {
    mask.Set<bool>({50, c}, true);
    mask.Set<bool>({90, c}, true);
    mask.Set<bool>({99, c}, true);
  }

  MismatchBoundingBox box = ComputeBoundingBoxFromLiteralMask(mask);
  EXPECT_EQ(box.mismatch_count, 32);
  EXPECT_EQ(box.top_mismatch_coords.size(), 16);

  // Reservoir sampling must ensure spatial representation across both early and
  // late rows.
  bool has_early_row = false;
  bool has_late_row = false;
  for (const auto& coord : box.top_mismatch_coords) {
    if (coord[0] <= 3) {
      has_early_row = true;
    }
    if (coord[0] >= 50) {
      has_late_row = true;
    }
  }
  EXPECT_TRUE(has_early_row);
  EXPECT_TRUE(has_late_row);
}

TEST(HloDumpUtilsTest, PopulateTensorVisualizationsAndSerialize) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_vis
ENTRY main {
  ROOT %p0 = f32[10,20] parameter(0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails mismatch =
      CreateSampleMismatch("p0", 0, 1.0, 2.0, 0.5,
                           MakeBoundingBox({10, 20}, {1, 2}, {5, 8}, 15, 200,
                                           /*pattern=*/"", {{2, 3}}));

  auto vis_map = PopulateTensorVisualizations(*module, {mismatch});
  ASSERT_TRUE(vis_map.contains("p0"));
  const auto& item = vis_map["p0"];
  EXPECT_EQ(item.instruction_name, "p0");
  EXPECT_EQ(item.opcode, "parameter");
  EXPECT_EQ(item.shape, (std::vector<int64_t>{10, 20}));
  EXPECT_TRUE(item.has_mismatch);
  EXPECT_EQ(item.box_min, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(item.box_max, (std::vector<int64_t>{5, 8}));
  EXPECT_EQ(item.mismatch_count, 15);
  EXPECT_EQ(item.total_elements, 200);

  std::string js = SerializeTensorVisualizationsJs(vis_map);
  EXPECT_TRUE(absl::StrContains(js, "\"has_mismatch\": true"));
  EXPECT_TRUE(absl::StrContains(js, "\"opcode\": \"parameter\""));
  EXPECT_TRUE(absl::StrContains(js, "\"box_min\": [1, 2]"));
  EXPECT_TRUE(absl::StrContains(js, "\"box_max\": [5, 8]"));
  EXPECT_TRUE(absl::StrContains(js, "\"mismatch_count\": 15"));
  EXPECT_TRUE(absl::StrContains(js, "\"total_elements\": 200"));

  std::string html = ConvertHloToHtml(
      module->name(), module->ToString(), /*annotations=*/{},
      /*recovery_info=*/{}, /*stack_frame_index=*/nullptr,
      /*graph_data=*/nullptr, /*tensor_visualizations=*/&vis_map);
  EXPECT_TRUE(absl::StrContains(html, "window.tensorVisualizations ="));
  EXPECT_TRUE(absl::StrContains(html, "\"mismatch_count\": 15"));
}

TEST(HloDumpUtilsTest, PopulateTensorVisualizationsDefaultsToSingleElement) {
  // Guards existing producers: without a bounding box, the historical fallback
  // of a single mismatching element is preserved.
  const absl::string_view hlo_string = R"hlo(
HloModule test_vis
ENTRY main {
  ROOT %p0 = f32[4] parameter(0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  MismatchDetails mismatch;
  mismatch.target_instruction_name = "p0";

  auto vis_map = PopulateTensorVisualizations(*module, {mismatch});
  ASSERT_TRUE(vis_map.contains("p0"));
  EXPECT_TRUE(vis_map["p0"].has_mismatch);
  EXPECT_EQ(vis_map["p0"].shape, (std::vector<int64_t>{4}));
  EXPECT_EQ(vis_map["p0"].mismatch_count, 1);
  EXPECT_EQ(vis_map["p0"].total_elements, 4);
}

TEST(HloDumpUtilsTest, PopulateTensorVisualizationsWithoutElementLevelData) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_vis
ENTRY main {
  ROOT %p0 = f32[10,20] parameter(0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  MismatchDetails in_module;
  in_module.target_instruction_name = "p0";
  in_module.has_element_level_data = false;
  // Even a supplied bounding box must not be rendered: the flag wins.
  in_module.bounding_box = MakeBoundingBox({10, 20}, {1, 2}, {5, 8}, 15, 200);

  MismatchDetails not_in_module;
  not_in_module.target_instruction_name = "absent";
  not_in_module.has_element_level_data = false;

  auto vis_map =
      PopulateTensorVisualizations(*module, {in_module, not_in_module});
  for (absl::string_view name : {"p0", "absent"}) {
    SCOPED_TRACE(name);
    ASSERT_TRUE(vis_map.contains(name));
    const auto& item = vis_map[name];
    // The mismatch is still reported, so HLO/graph highlighting keep working,
    // but the tensor is non-inspectable so the inspector stays closed.
    EXPECT_TRUE(item.has_mismatch);
    EXPECT_TRUE(item.shape.empty());
    EXPECT_TRUE(item.box_min.empty());
    EXPECT_TRUE(item.box_max.empty());
    EXPECT_TRUE(item.top_mismatches.empty());
    EXPECT_EQ(item.mismatch_count, 0);
    EXPECT_EQ(item.total_elements, 0);
  }
}

TEST(HloDumpUtilsTest, SerializeTensorVisualizationsJsSpecialFloats) {
  absl::flat_hash_map<std::string, TensorVisualizationInfo> vis_map;
  TensorVisualizationInfo item;
  item.instruction_name = "test_op";
  item.opcode = "custom-call";
  item.shape = {2, 2};
  item.has_mismatch = true;
  item.top_mismatches = {
      {{0, 0}, std::numeric_limits<double>::infinity()},
      {{0, 1}, -std::numeric_limits<double>::infinity()},
      {{1, 0}, std::numeric_limits<double>::quiet_NaN()},
      {{1, 1}, 0.25},
  };
  vis_map["test_op"] = item;

  std::string js = SerializeTensorVisualizationsJs(vis_map);
  EXPECT_TRUE(absl::StrContains(js, "\"rel_error\": Infinity"));
  EXPECT_TRUE(absl::StrContains(js, "\"rel_error\": -Infinity"));
  EXPECT_TRUE(absl::StrContains(js, "\"rel_error\": NaN"));
  EXPECT_TRUE(absl::StrContains(js, "\"rel_error\": 0.25"));
  EXPECT_FALSE(absl::StrContains(js, "\"rel_error\": inf"));
  EXPECT_FALSE(absl::StrContains(js, "\"rel_error\": -inf"));
  EXPECT_FALSE(absl::StrContains(js, "\"rel_error\": nan"));
}

std::vector<MismatchDetails> CreateMultiRankSampleMismatches() {
  std::vector<MismatchDetails> mismatches;

  // 1. Rank 5 tensor (add) with 3 composite slice boxes
  {
    auto bbox =
        MakeBoundingBox({2, 4, 8, 16, 32}, {0, 0, 0, 1, 2}, {1, 2, 7, 15, 30},
                        130, 2 * 4 * 8 * 16 * 32, "STRIDED (stride 8)",
                        {{0, 0, 1, 3, 8}, {0, 1, 4, 8, 16}, {1, 2, 6, 13, 24}});
    bbox.mismatched_slices = {0, 1, 2};
    // Slice 0: [0, 0, ...] -> High density cluster (72 / 84 cells = 85.71%
    // density) -> solid rich fill
    bbox.slice_boxes["0,0"] = {
        0, "0,0", {0, 0}, {0, 0, 0, 1, 2}, {0, 0, 2, 4, 8}, 72};
    // Slice 1: [0, 1, ...] -> Medium density cluster (48 / 135 cells = 35.56%
    // density) -> standard fill
    bbox.slice_boxes["0,1"] = {
        1, "0,1", {0, 1}, {0, 1, 3, 6, 12}, {0, 1, 5, 10, 20}, 48};
    // Slice 2: [1, 2, ...] -> Low density sparse lattice (10 / 135 cells
    // = 7.41% density) -> hollow ghost dashed lattice
    bbox.slice_boxes["1,2"] = {
        2, "1,2", {1, 2}, {1, 2, 5, 11, 22}, {1, 2, 7, 15, 30}, 10};
    mismatches.push_back(
        CreateSampleMismatch("add", 0, 1.45, 1.0, 0.45, std::move(bbox)));
  }

  // 2. Rank 2 matrix (sub)
  {
    auto bbox = MakeBoundingBox({64, 128}, {10, 20}, {40, 80}, 1200, 64 * 128,
                                "DENSE_BLOCK", {{25, 50}});
    mismatches.push_back(
        CreateSampleMismatch("sub", 1, 3.2, 3.0, 0.067, std::move(bbox)));
  }

  // 3. Rank 6 tensor (mul) with 3 composite slice boxes
  {
    auto bbox = MakeBoundingBox(
        {4, 5, 6, 7, 8, 6}, {1, 2, 3, 0, 1, 1}, {2, 3, 5, 5, 7, 5}, 64,
        4 * 5 * 6 * 7 * 8 * 6, "STRIDED (stride 4)",
        {{1, 2, 3, 2, 3, 2}, {1, 2, 4, 3, 2, 3}, {2, 3, 5, 1, 5, 3}});
    bbox.mismatched_slices = {3, 4, 5};
    // Slice 3: [1, 2, 3, ...] -> High density (36 / 48 cells = 75.0% density)
    bbox.slice_boxes["1,2,3"] = {
        3, "1,2,3", {1, 2, 3}, {1, 2, 3, 1, 2, 1}, {1, 2, 3, 4, 5, 3}, 36};
    // Slice 4: [1, 2, 4, ...] -> Medium density (20 / 48 cells = 41.67%
    // density)
    bbox.slice_boxes["1,2,4"] = {
        4, "1,2,4", {1, 2, 4}, {1, 2, 4, 2, 1, 2}, {1, 2, 4, 5, 4, 4}, 20};
    // Slice 5: [2, 3, 5, ...] -> Low density (8 / 96 cells = 8.33% density)
    bbox.slice_boxes["2,3,5"] = {
        5, "2,3,5", {2, 3, 5}, {2, 3, 5, 0, 4, 2}, {2, 3, 5, 5, 7, 5}, 8};
    mismatches.push_back(
        CreateSampleMismatch("mul", 2, 5.92, 4.0, 0.48, std::move(bbox)));
  }

  // 4. Rank 1 vector (neg)
  {
    auto bbox =
        MakeBoundingBox({256}, {64}, {95}, 32, 256, "CONTIGUOUS", {{72}});
    mismatches.push_back(
        CreateSampleMismatch("neg", 3, -0.575, -0.5, 0.15, std::move(bbox)));
  }

  // 5. Rank 0 scalar (abs)
  {
    auto bbox = MakeBoundingBox({}, {}, {}, 1, 1, "", {{}});
    mismatches.push_back(
        CreateSampleMismatch("abs", 4, 1.42, 1.0, 0.42, std::move(bbox)));
  }

  // 6. Rank 5 large tensor (large_scatter) with 16,777,216 elements and
  // scattered mismatches computed directly from a 16.8M-element Literal mask.
  {
    Shape shape = ShapeUtil::MakeShape(PRED, {2, 4, 64, 128, 256});
    Literal mask(shape);
    mask.PopulateWithValue(false);

    // Slice [0, 1]: wide sparse outliers across the [64, 128, 256] 3D volume
    const std::vector<std::vector<int64_t>> slice_0_1_coords = {
        {0, 1, 32, 64, 128},  // Peak mismatch coordinate
        {0, 1, 4, 8, 12},     {0, 1, 12, 24, 48},  {0, 1, 20, 44, 96},
        {0, 1, 28, 72, 160},  {0, 1, 40, 96, 200}, {0, 1, 52, 108, 224},
        {0, 1, 59, 119, 243},
    };
    for (const auto& c : slice_0_1_coords) {
      mask.Set<bool>(c, true);
    }
    for (int i = 0; i < 32; ++i) {
      mask.Set<bool>(
          {0, 1, 6 + (i * 5) % 50, 10 + (i * 11) % 105, 20 + (i * 17) % 215},
          true);
    }

    // Slice [0, 3]: localized tile cluster (e.g. attention block numerical
    // divergence)
    const std::vector<std::vector<int64_t>> slice_0_3_coords = {
        {0, 3, 24, 64, 128},
        {0, 3, 25, 68, 134},
        {0, 3, 26, 72, 139},
        {0, 3, 27, 75, 143},
    };
    for (const auto& c : slice_0_3_coords) {
      mask.Set<bool>(c, true);
    }
    for (int z = 24; z <= 27; ++z) {
      for (int y = 64; y <= 75; y += 2) {
        for (int x = 128; x <= 143; x += 2) {
          mask.Set<bool>({0, 3, z, y, x}, true);
        }
      }
    }

    // Slice [1, 0]: sparse outlier (causes component-wise global box_min
    // leading slice tuple to be [0, 0], which is a clean slice!)
    const std::vector<std::vector<int64_t>> slice_1_0_coords = {
        {1, 0, 15, 30, 60},
        {1, 0, 45, 90, 180},
    };
    for (const auto& c : slice_1_0_coords) {
      mask.Set<bool>(c, true);
    }

    // Slice [1, 2]: scattered boundary outliers
    const std::vector<std::vector<int64_t>> slice_1_2_coords = {
        {1, 2, 10, 20, 30},
        {1, 2, 30, 60, 120},
        {1, 2, 50, 100, 220},
    };
    for (const auto& c : slice_1_2_coords) {
      mask.Set<bool>(c, true);
    }

    MismatchBoundingBox bbox = ComputeBoundingBoxFromLiteralMask(mask);
    bbox.top_mismatch_coords.clear();
    for (const auto& c : slice_0_1_coords) {
      bbox.top_mismatch_coords.push_back(c);
    }
    for (const auto& c : slice_0_3_coords) {
      bbox.top_mismatch_coords.push_back(c);
    }
    for (const auto& c : slice_1_2_coords) {
      bbox.top_mismatch_coords.push_back(c);
    }
    bbox.top_mismatch_coords.push_back(slice_1_0_coords[0]);

    mismatches.push_back(CreateSampleMismatch("large_scatter", 5, 1.85, 1.0,
                                              0.85, std::move(bbox)));
  }

  // 7. Rank 5 tensor with large outer dimensions (large_outer: [64, 128, 32,
  // 64, 128]) to showcase adaptive continuous slider, numeric input, and
  // mismatch jump buttons.
  {
    MismatchBoundingBox bbox;
    bbox.box_min = {12, 24, 8, 16, 32};
    bbox.box_max = {45, 96, 12, 24, 48};
    bbox.top_mismatch_coords = {
        {12, 24, 8, 16, 32},
        {28, 60, 10, 20, 40},
        {45, 96, 12, 24, 48},
    };
    bbox.mismatch_count = 3;

    auto make_sbox = [](std::string key, int64_t d0, int64_t d1,
                        std::vector<int64_t> min_b, std::vector<int64_t> max_b,
                        int64_t count) {
      SliceBoundingBox sbox;
      sbox.slice_key = std::move(key);
      sbox.slice_coords = {d0, d1};
      sbox.slice_index = d1;
      sbox.box_min = std::move(min_b);
      sbox.box_max = std::move(max_b);
      sbox.mismatch_count = count;
      return sbox;
    };

    bbox.slice_boxes["12,24"] =
        make_sbox("12,24", 12, 24, {12, 24, 8, 16, 32}, {12, 24, 9, 18, 36}, 1);
    bbox.slice_boxes["28,60"] = make_sbox("28,60", 28, 60, {28, 60, 10, 20, 40},
                                          {28, 60, 11, 22, 44}, 1);
    bbox.slice_boxes["45,96"] = make_sbox("45,96", 45, 96, {45, 96, 11, 23, 46},
                                          {45, 96, 12, 24, 48}, 1);

    mismatches.push_back(CreateSampleMismatch("large_outer", 5, 1.75, 1.0, 0.75,
                                              std::move(bbox)));
  }

  return mismatches;
}

void VerifyMultiRankTensorVisualizations(
    const absl::flat_hash_map<std::string, TensorVisualizationInfo>&
        tensor_visualizations) {
  // Check tensor visualizations for clean vs failing operations.
  ASSERT_TRUE(tensor_visualizations.contains("p0"));
  EXPECT_FALSE(tensor_visualizations.at("p0").has_mismatch);
  ASSERT_TRUE(tensor_visualizations.contains("tuple"));
  EXPECT_FALSE(tensor_visualizations.at("tuple").has_mismatch);

  // Rank 0 scalar (abs)
  ASSERT_TRUE(tensor_visualizations.contains("abs"));
  const auto& abs_vis = tensor_visualizations.at("abs");
  EXPECT_TRUE(abs_vis.has_mismatch);
  EXPECT_EQ(abs_vis.shape, (std::vector<int64_t>{}));
  EXPECT_EQ(abs_vis.total_elements, 1);
  EXPECT_EQ(abs_vis.mismatch_count, 1);
  ASSERT_FALSE(abs_vis.top_mismatches.empty());
  EXPECT_DOUBLE_EQ(abs_vis.top_mismatches[0].rel_error, 0.42);

  // Rank 1 vector (neg)
  ASSERT_TRUE(tensor_visualizations.contains("neg"));
  const auto& neg_vis = tensor_visualizations.at("neg");
  EXPECT_TRUE(neg_vis.has_mismatch);
  EXPECT_EQ(neg_vis.shape, (std::vector<int64_t>{256}));
  EXPECT_EQ(neg_vis.box_min, (std::vector<int64_t>{64}));
  EXPECT_EQ(neg_vis.box_max, (std::vector<int64_t>{95}));
  EXPECT_EQ(neg_vis.pattern, "CONTIGUOUS");
  EXPECT_EQ(neg_vis.mismatch_count, 32);
  EXPECT_EQ(neg_vis.total_elements, 256);
  ASSERT_FALSE(neg_vis.top_mismatches.empty());
  EXPECT_DOUBLE_EQ(neg_vis.top_mismatches[0].rel_error, 0.15);

  // Rank 2 matrix (sub)
  ASSERT_TRUE(tensor_visualizations.contains("sub"));
  const auto& sub_vis = tensor_visualizations.at("sub");
  EXPECT_TRUE(sub_vis.has_mismatch);
  EXPECT_EQ(sub_vis.shape, (std::vector<int64_t>{64, 128}));
  EXPECT_EQ(sub_vis.box_min, (std::vector<int64_t>{10, 20}));
  EXPECT_EQ(sub_vis.box_max, (std::vector<int64_t>{40, 80}));
  EXPECT_EQ(sub_vis.pattern, "DENSE_BLOCK");
  EXPECT_EQ(sub_vis.mismatch_count, 1200);
  EXPECT_EQ(sub_vis.total_elements, 64 * 128);

  // Rank 5 tensor (add) with 3 slice boxes
  ASSERT_TRUE(tensor_visualizations.contains("add"));
  const auto& add_vis = tensor_visualizations.at("add");
  EXPECT_TRUE(add_vis.has_mismatch);
  EXPECT_EQ(add_vis.shape, (std::vector<int64_t>{2, 4, 8, 16, 32}));
  EXPECT_EQ(add_vis.pattern, "STRIDED (stride 8)");
  EXPECT_EQ(add_vis.slice_boxes.size(), 3);
  ExpectSliceBox(add_vis.slice_boxes, "0,0", {0, 0, 0, 1, 2}, {0, 0, 2, 4, 8},
                 72);
  ExpectSliceBox(add_vis.slice_boxes, "0,1", {0, 1, 3, 6, 12},
                 {0, 1, 5, 10, 20}, 48);
  ExpectSliceBox(add_vis.slice_boxes, "1,2", {1, 2, 5, 11, 22},
                 {1, 2, 7, 15, 30}, 10);

  // Rank 6 tensor (mul) with 3 slice boxes
  ASSERT_TRUE(tensor_visualizations.contains("mul"));
  const auto& mul_vis = tensor_visualizations.at("mul");
  EXPECT_TRUE(mul_vis.has_mismatch);
  EXPECT_EQ(mul_vis.shape, (std::vector<int64_t>{4, 5, 6, 7, 8, 6}));
  EXPECT_EQ(mul_vis.pattern, "STRIDED (stride 4)");
  EXPECT_EQ(mul_vis.slice_boxes.size(), 3);
  ExpectSliceBox(mul_vis.slice_boxes, "1,2,3", {1, 2, 3, 1, 2, 1},
                 {1, 2, 3, 4, 5, 3}, 36);
  ExpectSliceBox(mul_vis.slice_boxes, "1,2,4", {1, 2, 4, 2, 1, 2},
                 {1, 2, 4, 5, 4, 4}, 20);
  ExpectSliceBox(mul_vis.slice_boxes, "2,3,5", {2, 3, 5, 0, 4, 2},
                 {2, 3, 5, 5, 7, 5}, 8);
  ASSERT_FALSE(mul_vis.top_mismatches.empty());
  EXPECT_DOUBLE_EQ(mul_vis.top_mismatches[0].rel_error, 0.48);

  // Rank 5 large tensor (large_scatter) with 16,777,216 elements
  ASSERT_TRUE(tensor_visualizations.contains("large_scatter"));
  const auto& large_vis = tensor_visualizations.at("large_scatter");
  EXPECT_TRUE(large_vis.has_mismatch);
  EXPECT_EQ(large_vis.shape, (std::vector<int64_t>{2, 4, 64, 128, 256}));
  EXPECT_EQ(large_vis.total_elements, 16777216);
  EXPECT_EQ(large_vis.pattern, "SPARSE_OUTLIERS");
  EXPECT_EQ(large_vis.slice_boxes.size(), 4);
  EXPECT_TRUE(large_vis.slice_boxes.contains("0,1"));
  EXPECT_TRUE(large_vis.slice_boxes.contains("0,3"));
  EXPECT_TRUE(large_vis.slice_boxes.contains("1,0"));
  EXPECT_TRUE(large_vis.slice_boxes.contains("1,2"));
  ASSERT_FALSE(large_vis.top_mismatches.empty());
  EXPECT_DOUBLE_EQ(large_vis.top_mismatches[0].rel_error, 0.85);

  // Rank 5 tensor with large outer dimensions (large_outer)
  ASSERT_TRUE(tensor_visualizations.contains("large_outer"));
  const auto& outer_vis = tensor_visualizations.at("large_outer");
  EXPECT_TRUE(outer_vis.has_mismatch);
  EXPECT_EQ(outer_vis.shape, (std::vector<int64_t>{64, 128, 32, 64, 128}));
  EXPECT_EQ(outer_vis.box_min, (std::vector<int64_t>{12, 24, 8, 16, 32}));
  EXPECT_EQ(outer_vis.box_max, (std::vector<int64_t>{45, 96, 12, 24, 48}));
  EXPECT_EQ(outer_vis.mismatch_count, 3);
  ASSERT_EQ(outer_vis.top_mismatches.size(), 3);
  EXPECT_DOUBLE_EQ(outer_vis.top_mismatches[0].rel_error, 0.75);
  EXPECT_EQ(outer_vis.slice_boxes.size(), 3);
}

void VerifyDumpedHtmlReport(absl::string_view file_content) {
  constexpr absl::string_view kExpectedSubstrings[] = {
      // Expected container elements and script injections
      "window.tensorVisualizations =",
      "window.compressedGraphData =",
      "demo_pipeline",
      "id=\"graph-module-title\"",
      "id=\"tensor-inspector\"",
      "id=\"tensor-3d-canvas\"",
      "id=\"splitter-sidebar\"",
      "id=\"zoom-in-btn\"",
      "id=\"zoom-out-btn\"",
      "id=\"zoom-fit-btn\"",
      // Instruction anchors
      "id=\"instr_p0\"",
      "id=\"instr_add\"",
      "id=\"instr_sub\"",
      "id=\"instr_mul\"",
      "id=\"instr_neg\"",
      "id=\"instr_abs\"",
      "id=\"instr_large_scatter\"",
      "id=\"instr_large_outer\"",
      "id=\"instr_tuple\"",
      // JSON representations for each rank
      "\"abs\":",
      "\"rel_error\": 0.42",
      "\"neg\":",
      "\"pattern\": \"CONTIGUOUS\"",
      "\"sub\":",
      "\"pattern\": \"DENSE_BLOCK\"",
      "\"add\":",
      "\"pattern\": \"STRIDED (stride 8)\"",
      "\"0,0\":",
      "\"0,1\":",
      "\"1,2\":",
      "\"mul\":",
      "\"pattern\": \"STRIDED (stride 4)\"",
      "\"1,2,3\":",
      "\"1,2,4\":",
      "\"2,3,5\":",
      "\"large_scatter\":",
      "\"pattern\": \"SPARSE_OUTLIERS\"",
      "\"total_elements\": 16777216",
      "\"large_outer\":",
  };
  for (absl::string_view needle : kExpectedSubstrings) {
    EXPECT_TRUE(absl::StrContains(file_content, needle)) << needle;
  }

  constexpr absl::string_view kUnexpectedSubstrings[] = {
      "toggle-inspector-btn",
      "3D Tensor</button>",
  };
  for (absl::string_view needle : kUnexpectedSubstrings) {
    EXPECT_FALSE(absl::StrContains(file_content, needle)) << needle;
  }
}

TEST(HloDumpUtilsTest, DumpHloModuleMismatchWithGraphData) {
  const absl::string_view hlo_string = R"hlo(
HloModule demo_pipeline
ENTRY main {
  p0 = f32[2,4,8,16,32] parameter(0)
  p1 = f32[2,4,8,16,32] parameter(1)
  add = f32[2,4,8,16,32] add(p0, p1)
  p2 = f32[64,128] parameter(2)
  p3 = f32[64,128] parameter(3)
  sub = f32[64,128] subtract(p2, p3)
  p4 = f32[4,5,6,7,8,6] parameter(4)
  p5 = f32[4,5,6,7,8,6] parameter(5)
  mul = f32[4,5,6,7,8,6] multiply(p4, p5)
  p6 = f32[256] parameter(6)
  neg = f32[256] negate(p6)
  p7 = f32[] parameter(7)
  abs = f32[] abs(p7)
  p8 = f32[2,4,64,128,256] parameter(8)
  p9 = f32[2,4,64,128,256] parameter(9)
  large_scatter = f32[2,4,64,128,256] add(p8, p9)
  p10 = f32[64,128,32,64,128] parameter(10)
  p11 = f32[64,128,32,64,128] parameter(11)
  large_outer = f32[64,128,32,64,128] add(p10, p11)
  ROOT tuple = (f32[2,4,8,16,32], f32[64,128], f32[4,5,6,7,8,6], f32[256], f32[], f32[2,4,64,128,256], f32[64,128,32,64,128]) tuple(add, sub, mul, neg, abs, large_scatter, large_outer)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));

  std::vector<MismatchDetails> mismatches = CreateMultiRankSampleMismatches();

  ASSERT_OK_AND_ASSIGN(const std::string html_filename,
                       DumpHloModuleMismatchWithGraphData(
                           *module, mismatches, "sample_mismatch_report.html"));
  EXPECT_FALSE(html_filename.empty());

  // 1. Verify that the dumped file exists and read its contents back from disk.
  std::ifstream ifs(html_filename);
  ASSERT_TRUE(ifs.is_open()) << "Failed to open dumped file: " << html_filename;
  std::string file_content((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
  EXPECT_GT(file_content.size(), 50000);

  // 2. Check the populated data structures directly.
  GraphData graph_data = PopulateMismatchGraphData(*module, mismatches);
  EXPECT_EQ(graph_data.nodes.size(), 20);
  EXPECT_FALSE(graph_data.edges.empty());

  VerifyMultiRankTensorVisualizations(
      PopulateTensorVisualizations(*module, mismatches));

  // 3. Verify the dumped HTML file content strings.
  VerifyDumpedHtmlReport(file_content);

  // 4. Mirror to /tmp/mismatch_demo/sample_mismatch_report.html for local live
  // inspection.
  tsl::Env::Default()->RecursivelyCreateDir("/tmp/mismatch_demo").IgnoreError();
  std::ofstream ofs("/tmp/mismatch_demo/sample_mismatch_report.html");
  if (ofs.is_open()) {
    ofs << file_content;
    ofs.close();
  }
}

}  // namespace
}  // namespace xla::numerics::debug_info
