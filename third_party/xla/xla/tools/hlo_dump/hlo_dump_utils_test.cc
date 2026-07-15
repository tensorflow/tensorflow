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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
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
  ASSERT_TRUE(annotations.contains(inner_key));
  EXPECT_EQ(annotations[outer_key].background_color, "pink");
  EXPECT_EQ(annotations[inner_key].background_color, "pink");
  EXPECT_EQ(annotations[outer_key].tooltip_data,
            annotations[inner_key].tooltip_data);

  bool found_fusion = false;
  bool found_add1 = false;
  for (const auto& node : graph_data.nodes) {
    if (node.key == "fusion") {
      found_fusion = true;
      EXPECT_EQ(node.diff_score, 25.0);
    } else if (node.key == "fusion/add.1") {
      found_add1 = true;
      EXPECT_EQ(node.diff_score, 25.0);
    }
  }
  EXPECT_TRUE(found_fusion);
  EXPECT_TRUE(found_add1);
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

  // Verify that the color lightgreen is associated with the element
  EXPECT_TRUE(absl::StrContains(html, "bg-lightgreen"));

  // Verify that the link is correctly sanitized to point to the GTE
  EXPECT_TRUE(absl::StrContains(html, "href=\"#instr_p0_0\""));
}

}  // namespace
}  // namespace xla::numerics::debug_info
