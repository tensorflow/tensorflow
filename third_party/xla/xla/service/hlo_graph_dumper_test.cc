/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_graph_dumper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/io/inputstream_interface.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_inputstream.h"
#include "xla/xla.pb.h"
#include "tsl/platform/tstring.h"

namespace xla {
namespace {

using absl::StrCat;
using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::Not;
using ::testing::UnorderedElementsAre;

using HloGraphDumperTest = HloHardwareIndependentTestBase;

TEST_F(HloGraphDumperTest, NestedFusion) {
  HloComputation::Builder b("b");

  // Build param0 + param1 + param2 + param3 + param4.
  auto shape = ShapeUtil::MakeShape(F32, {10, 100});
  std::vector<HloInstruction*> params;
  for (int i = 0; i <= 4; ++i) {
    params.push_back(b.AddInstruction(
        HloInstruction::CreateParameter(i, shape, StrCat("param", i))));
  }
  std::vector<HloInstruction*> sums;
  sums.push_back(b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, params[0], params[1])));
  for (int i = 0; i <= 2; ++i) {
    sums.push_back(b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, sums[i], params[i + 2])));
  }
  HloModuleConfig config;
  HloModule m(TestName(), config);
  m.AddEntryComputation(b.Build());
  HloComputation* root_computation = m.entry_computation();

  // Fuse into fusion(param0 + param1 + param2 + param3 + param4).
  auto* outer_fusion = root_computation->CreateFusionInstruction(
      {sums[3], sums[2], sums[1], sums[0]}, HloInstruction::FusionKind::kLoop);

  // Fusing invalidates the pointers in sums -- the instructions are cloned when
  // they're moved to the new computation.  Get the updated pointers to sums.
  std::vector<HloInstruction*> fused_sums;
  for (auto* instr : outer_fusion->fused_instructions_computation()
                         ->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kAdd) {
      fused_sums.push_back(instr);
    }
  }

  // Fuse into fusion(fusion(param0 + param1 + param2) + param3 + param4).
  auto* inner_fusion =
      outer_fusion->fused_instructions_computation()->CreateFusionInstruction(
          {fused_sums[1], fused_sums[0]}, HloInstruction::FusionKind::kLoop);

  // Generate the graph; all nodes should be present.
  ASSERT_OK_AND_ASSIGN(std::string graph,
                       RenderGraph(*root_computation, /*label=*/"",
                                   DebugOptions(), RenderedGraphFormat::kDot));
  for (const HloComputation* computation :
       {root_computation,  //
        inner_fusion->fused_instructions_computation(),
        outer_fusion->fused_instructions_computation()}) {
    for (const HloInstruction* instruction : computation->instructions()) {
      EXPECT_THAT(graph, HasSubstr(instruction->name()));
    }
  }

  // Dump a neighborhood around one of the inner sum nodes.  We don't really
  // care that the outer nodes are omitted -- whether they are or not is based
  // fiddly heuristics -- but we do care that the node we asked for is printed.
  const HloInstruction* inner_sum = nullptr;
  for (const HloInstruction* instruction :
       inner_fusion->fused_instructions_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kAdd) {
      inner_sum = instruction;
      break;
    }
  }
  ASSERT_NE(inner_sum, nullptr);
  ASSERT_OK_AND_ASSIGN(std::string neighborhood_graph,
                       RenderNeighborhoodAround(*inner_sum, /*radius=*/1,
                                                RenderedGraphFormat::kDot));
  EXPECT_THAT(neighborhood_graph, HasSubstr(inner_sum->name()));
}

TEST_F(HloGraphDumperTest, Constant) {
  HloComputation::Builder b("b");
  auto instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-42)));
  instruction->SetAndSanitizeName("i_am_a_constant_root_instruction");
  HloModuleConfig config;
  HloModule m(TestName(), config);
  HloComputation* root_computation = m.AddEntryComputation(b.Build());
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*root_computation, /*label=*/"an_empty_graph", DebugOptions(),
                  RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("an_empty_graph"));
}

TEST_F(HloGraphDumperTest, TupleConstant) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(S32, {4, 5})});
  HloComputation::Builder b("b");
  auto constant = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(tuple_shape)));
  auto gte = b.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::MakeShape(F32, {3, 2}), constant, 0));

  HloModuleConfig config;
  HloModule m(TestName(), config);
  HloComputation* root_computation = m.AddEntryComputation(b.Build(gte));
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*root_computation, /*label=*/"tuple_constant", DebugOptions(),
                  RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("tuple_constant"));
  EXPECT_THAT(graph, HasSubstr("constant (f32[3,2], s32[4,5])"));
}

TEST_F(HloGraphDumperTest, Compare) {
  const char* hlo_string = R"(
    HloModule comp

    ENTRY comp {
      param.0 = f32[10] parameter(0)
      param.1 = f32[10] parameter(1)
      ROOT lt = pred[10] compare(param.0, param.1), direction=LT
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation(), /*label=*/"tuple_constant",
                  DebugOptions(), RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("direction=LT"));
}

TEST_F(HloGraphDumperTest, HasStatisticsViz) {
  const char* hlo_string = R"(
    HloModule comp

    ENTRY comp {
      param.0 = f32[10] parameter(0), statistics={visualizing_index=0,stat-0=0.5}
      param.1 = f32[10] parameter(1), statistics={visualizing_index=1,stat-0=55.5,stat-1=44.4}
      ROOT lt = pred[10] compare(param.0, param.1), direction=LT
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  // Just check that it doesn't crash.
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation(), /*label=*/"tuple_constant",
                  DebugOptions(), RenderedGraphFormat::kDot));
}

TEST_F(HloGraphDumperTest, RootIsConstant) {
  const char* hlo_string = R"(
HloModule indexed_conditional

%then_branch (empty: ()) -> f32[] {
  %empty = () parameter(0)
  ROOT %then = f32[] constant(1)
}

%else_branch (empty.1: ()) -> f32[] {
  %empty.1 = () parameter(0)
  ROOT %else = f32[] constant(2)
}

ENTRY %conditional_select (constant: pred[]) -> (f32[]) {
  %constant = pred[] parameter(0)
  %emptytuple = () tuple()
  %conditional = f32[] conditional(pred[] %constant, () %emptytuple, () %emptytuple), true_computation=%then_branch, false_computation=%else_branch
  ROOT %t = (f32[]) tuple(f32[] %conditional)
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  // Just check that it doesn't crash.
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation(), /*label=*/"tuple_constant",
                  DebugOptions(), RenderedGraphFormat::kDot));
}

TEST_F(HloGraphDumperTest, ShowCallers) {
  const char* hlo_string = R"(
    command_buffer {
      ROOT root = f32[16] parameter(0)
    }
    ENTRY comp {
      p0 = f32[16] parameter(0)
      ROOT call.1 = f32[16] call(p0), to_apply=command_buffer
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(std::string graph,
                       RenderGraph(*module->entry_computation(),
                                   /*label=*/"command_buffer", DebugOptions(),
                                   RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("ENTRY computation"));

  ASSERT_OK_AND_ASSIGN(
      graph,
      RenderGraph(*module->entry_computation()->root_instruction()->to_apply(),
                  /*label=*/"command_buffer", DebugOptions(),
                  RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("Caller instructions: call.1"));
}

TEST_F(HloGraphDumperTest, OverrideColors) {
  const char* hlo_string = R"(
    HloModule comp

    ENTRY comp {
      param.0 = f32[10] parameter(0)
      param.1 = f32[10] parameter(1)
      ROOT lt = pred[10] compare(param.0, param.1), direction=LT
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  // Create a color map with color and stats
  absl::flat_hash_map<const HloInstruction*, ColorStats> color_map;
  ColorStats color_stats_1;
  color_stats_1.color = "#A9C343";
  color_stats_1.stats = absl::StrFormat("%.3f", 1.11);
  ColorStats color_stats_2;
  color_stats_2.color = "#BC8A3F";
  color_stats_2.stats = absl::StrFormat("%.3f", 2.22);
  color_map[module->entry_computation()->GetInstructionWithName("param.0")] =
      color_stats_1;
  color_map[module->entry_computation()->GetInstructionWithName("param.1")] =
      color_stats_2;

  HloRenderOptions hlo_render_options;
  hlo_render_options.override_node_colors = true;
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation(), /*label=*/"tuple_constant",
                  DebugOptions(), RenderedGraphFormat::kDot, hlo_render_options,
                  color_map));
  EXPECT_THAT(graph, HasSubstr("#A9C343"));
  EXPECT_THAT(graph, HasSubstr("1.110"));
  EXPECT_THAT(graph, HasSubstr("#BC8A3F"));
  EXPECT_THAT(graph, HasSubstr("2.220"));
}

TEST_F(HloGraphDumperTest, AnnotateCalledComputationsParameters) {
  const char* hlo_string = R"(
    command_buffer.0 {
      p0 = f32[1024] parameter(0)
      add.123 = f32[1024] add(p0, p0)
      mul.456 = f32[1024] multiply(add.123, p0)
      ROOT tuple = (f32[1024], f32[1024]) tuple(add.123, mul.456)
    }
    command_buffer.1 {
      p0 = f32[1024] parameter(0)  // Output of add.123, but hard to tell due to
                                   // get-tuple-element, call, bitcast, etc.
      p1 = f32[1024] parameter(1)  // Output of mul.456.
      ROOT mul = f32[1024] multiply(p1, p0)
    }
    ENTRY comp {
      p0 = f32[1024] parameter(0)
      call.0 = (f32[1024], f32[1024]) call(p0), to_apply=command_buffer.0
      gte.0 = f32[1024] get-tuple-element(call.0), index=0
      gte.1 = f32[1024] get-tuple-element(call.0), index=1
      bitcast = f32[32,32] bitcast(gte.0)
      tuple = (f32[1024], f32[32,32]) tuple(gte.1, bitcast)
      gte.2 = f32[1024] get-tuple-element(tuple), index=0
      gte.3 = f32[32,32] get-tuple-element(tuple), index=1
      bitcast.1 = f32[1024] bitcast(gte.3)
      ROOT call.1 = f32[1024] call(bitcast.1, gte.2), to_apply=command_buffer.1
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation()->root_instruction()->to_apply(),
                  /*label=*/"command buffer", DebugOptions(),
                  RenderedGraphFormat::kDot));
  EXPECT_THAT(graph, HasSubstr("<b>Parameter 0</b><br/>"
                               "<i>from add.123 in command_buffer.0</i>"));
  EXPECT_THAT(graph, HasSubstr("<b>Parameter 1</b><br/>"
                               "<i>from mul.456 in command_buffer.0</i>"));
}

TEST_F(HloGraphDumperTest, AnnotateCalledComputationsParametersTuple) {
  const char* hlo_string = R"(
    command_buffer {
      p0 = (f32[1024], f32[1024]) parameter(0)
      gte.0 = f32[1024] get-tuple-element(p0), index=0
      gte.1 = f32[1024] get-tuple-element(p0), index=1
      ROOT tuple = (f32[1024], f32[1024]) tuple(gte.0, gte.1)
    }
    ENTRY comp {
      p0 = f32[1024] parameter(0)
      add.123 = f32[1024] add(p0, p0)
      mul.456 = f32[1024] multiply(p0, p0)
      tuple.1 = (f32[1024], f32[1024]) tuple(add.123, mul.456)
      call.0 = (f32[1024], f32[1024]) call(tuple.1), to_apply=command_buffer
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(
      std::string graph,
      RenderGraph(*module->entry_computation()->root_instruction()->to_apply(),
                  /*label=*/"command buffer", DebugOptions(),
                  RenderedGraphFormat::kDot));
  // Annotate the parameter as `tuple.1`, rather than `add.123` or `mul.456`,
  // because both are being passed at the same time.
  EXPECT_THAT(graph, HasSubstr("<b>Parameter 0</b><br/>"
                               "<i>from tuple.1 in the ENTRY computation</i>"));
}

uint16_t ReadLe16(absl::string_view data, size_t offset) {
  return static_cast<uint16_t>(static_cast<uint8_t>(data[offset])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(data[offset + 1])) << 8);
}

uint32_t ReadLe32(absl::string_view data, size_t offset) {
  return static_cast<uint32_t>(static_cast<uint8_t>(data[offset])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[offset + 1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[offset + 2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(data[offset + 3])) << 24);
}

class StringViewInputStream : public tsl::io::InputStreamInterface {
 public:
  explicit StringViewInputStream(absl::string_view data) : data_(data) {}

  absl::Status ReadNBytes(int64_t bytes_to_read,
                          tsl::tstring* result) override {
    if (bytes_to_read < 0) {
      return absl::InvalidArgumentError("Negative bytes_to_read");
    }
    size_t available = data_.size() - pos_;
    size_t to_read = std::min(static_cast<size_t>(bytes_to_read), available);
    result->assign(data_.data() + pos_, to_read);
    pos_ += to_read;
    if (to_read < static_cast<size_t>(bytes_to_read)) {
      return absl::OutOfRangeError("EOF");
    }
    return absl::OkStatus();
  }

  int64_t Tell() const override { return static_cast<int64_t>(pos_); }

  absl::Status Reset() override {
    pos_ = 0;
    return absl::OkStatus();
  }

 private:
  absl::string_view data_;
  size_t pos_ = 0;
};

absl::StatusOr<std::string> RawInflate(absl::string_view compressed,
                                       size_t uncompressed_size) {
  StringViewInputStream raw_stream(compressed);
  auto opts = tsl::io::ZlibCompressionOptions::RAW();
  tsl::io::ZlibInputStream zlib_stream(&raw_stream, opts.input_buffer_size,
                                       opts.output_buffer_size, opts);
  tsl::tstring output;
  absl::Status status =
      zlib_stream.ReadNBytes(static_cast<int64_t>(uncompressed_size), &output);
  if (!status.ok()) {
    return status;
  }
  tsl::tstring trailing;
  if (!absl::IsOutOfRange(zlib_stream.ReadNBytes(1, &trailing)) ||
      !trailing.empty()) {
    return absl::DataLossError("Unexpected trailing bytes in DEFLATE stream");
  }
  return std::string(output);
}

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ExtractZipappEntries(absl::string_view zipapp) {
  constexpr absl::string_view kShebang = "#!/usr/bin/env python3\n";
  if (!absl::StartsWith(zipapp, kShebang)) {
    return absl::InvalidArgumentError("Missing python3 shebang prefix");
  }
  constexpr uint32_t kLocalFileHeaderSig = 0x04034b50;
  constexpr uint32_t kCentralDirHeaderSig = 0x02014b50;
  constexpr uint32_t kEndOfCentralDirSig = 0x06054b50;
  constexpr uint16_t kMethodDeflate = 8;

  absl::flat_hash_map<std::string, std::string> entries;
  size_t pos = kShebang.size();
  while (pos + 4 <= zipapp.size() &&
         ReadLe32(zipapp, pos) == kLocalFileHeaderSig) {
    if (pos + 30 > zipapp.size()) {
      return absl::InvalidArgumentError("Truncated local file header");
    }
    uint16_t method = ReadLe16(zipapp, pos + 8);
    uint32_t compressed_size = ReadLe32(zipapp, pos + 18);
    uint32_t uncompressed_size = ReadLe32(zipapp, pos + 22);
    uint16_t name_len = ReadLe16(zipapp, pos + 26);
    uint16_t extra_len = ReadLe16(zipapp, pos + 28);
    size_t data_offset = pos + 30 + name_len + extra_len;
    if (data_offset + compressed_size > zipapp.size()) {
      return absl::InvalidArgumentError("Truncated compressed payload");
    }
    std::string name(zipapp.substr(pos + 30, name_len));
    absl::string_view payload = zipapp.substr(data_offset, compressed_size);
    std::string content;
    if (method == 0) {
      content = std::string(payload);
    } else if (method == kMethodDeflate) {
      absl::StatusOr<std::string> inflated =
          RawInflate(payload, uncompressed_size);
      if (!inflated.ok()) {
        return inflated.status();
      }
      content = *std::move(inflated);
    } else {
      return absl::InvalidArgumentError("Unsupported ZIP compression method");
    }
    entries.emplace(std::move(name), std::move(content));
    pos = data_offset + compressed_size;
  }

  size_t cd_entries = 0;
  while (pos + 4 <= zipapp.size() &&
         ReadLe32(zipapp, pos) == kCentralDirHeaderSig) {
    if (pos + 46 > zipapp.size()) {
      return absl::InvalidArgumentError("Truncated central directory header");
    }
    uint16_t name_len = ReadLe16(zipapp, pos + 28);
    uint16_t extra_len = ReadLe16(zipapp, pos + 30);
    uint16_t comment_len = ReadLe16(zipapp, pos + 32);
    pos += 46 + name_len + extra_len + comment_len;
    ++cd_entries;
  }
  if (cd_entries != entries.size()) {
    return absl::DataLossError("Central directory entry count mismatch");
  }
  if (pos + 22 != zipapp.size() ||
      ReadLe32(zipapp, pos) != kEndOfCentralDirSig) {
    return absl::InvalidArgumentError("Missing or invalid EOCD record");
  }
  return entries;
}

TEST_F(HloGraphDumperTest, WrapFusionExplorerEmptyReturnsError) {
  const char* hlo_string = R"(
    HloModule empty_fusion_state

    ENTRY main {
      ROOT p0 = f32[4] parameter(0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_THAT(WrapFusionExplorer(*module->entry_computation()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("Empty")));
}

TEST_F(HloGraphDumperTest, WrapFusionExplorerPackagesGraphsAndEscapedMetadata) {
  const char* hlo_string = R"(
    HloModule fusion_vis

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      add.0 = f32[8] add(p0, p1)
      ROOT mul.0 = f32[8] multiply(add.0, p1)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* entry = module->entry_computation();
  const HloInstruction* add = entry->GetInstructionWithName("add.0");
  const HloInstruction* mul = entry->GetInstructionWithName("mul.0");
  ASSERT_NE(add, nullptr);
  ASSERT_NE(mul, nullptr);

  // Register two consecutive states with the same rendered DOT graph (to test
  // graph deduplication across frames) followed by a distinct consumer state.
  std::string special_label("Step \"1\" \\ line1\nline2\r\t\x01");
  RegisterFusionState(*entry, special_label, *mul, /*producer=*/add);
  RegisterFusionState(*entry, "Step 2 same graph", *mul, /*producer=*/nullptr);
  RegisterFusionState(*entry, "Step 3 add neighborhood", *add,
                      /*producer=*/nullptr);

  ASSERT_OK_AND_ASSIGN(std::string zipapp, WrapFusionExplorer(*entry));
  ASSERT_OK_AND_ASSIGN(auto entries, ExtractZipappEntries(zipapp));

  EXPECT_THAT(entries,
              UnorderedElementsAre(Key("__main__.py"), Key("viewer.html"),
                                   Key("metadata.json"), Key("graphs/0.dot"),
                                   Key("graphs/1.dot")));
  EXPECT_THAT(entries, Not(Contains(Key("fusions.db"))));
  EXPECT_THAT(entries["__main__.py"], HasSubstr("class FusionExplorerHandler"));
  EXPECT_THAT(entries["viewer.html"], HasSubstr("Fusion Explorer"));

  const std::string& metadata = entries["metadata.json"];
  EXPECT_THAT(metadata, HasSubstr("\"title\":\"fusion_vis_main\""));
  EXPECT_THAT(
      metadata,
      HasSubstr("\"id\":0,\"label\":\"Step \\\"1\\\" \\\\ line1\\nline2\\r\\t"
                "\\u0001\""));
  EXPECT_THAT(metadata, HasSubstr("\"graph_id\":0}"));
  EXPECT_THAT(
      metadata,
      HasSubstr(
          "{\"id\":1,\"label\":\"Step 2 same graph\",\"to_highlight\":\"\","
          "\"graph_id\":0}"));
  EXPECT_THAT(metadata,
              HasSubstr("{\"id\":2,\"label\":\"Step 3 add neighborhood\","
                        "\"to_highlight\":\"\",\"graph_id\":1}"));

  EXPECT_THAT(entries["graphs/0.dot"], HasSubstr("digraph"));
  EXPECT_THAT(entries["graphs/0.dot"], HasSubstr("mul.0"));
  EXPECT_THAT(entries["graphs/1.dot"], HasSubstr("add.0"));
}

TEST_F(HloGraphDumperTest, RenderAllComputationsToHtmlProducesValidZipapp) {
  const char* hlo_string = R"(
    HloModule render_all

    ENTRY main {
      p0 = f32[4,8] parameter(0)
      t0 = f32[8,4] transpose(p0), dimensions={1,0}
      ROOT neg0 = f32[8,4] negate(t0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(std::string zipapp,
                       RenderAllComputationsToHtml(*module));
  ASSERT_OK_AND_ASSIGN(auto entries, ExtractZipappEntries(zipapp));

  EXPECT_THAT(entries, Contains(Key("__main__.py")));
  EXPECT_THAT(entries, Contains(Key("viewer.html")));
  EXPECT_THAT(entries, Contains(Key("metadata.json")));
  EXPECT_THAT(entries, Contains(Key("graphs/0.dot")));
  EXPECT_THAT(entries["metadata.json"], HasSubstr("\"title\":\"render_all\""));
  EXPECT_THAT(entries["metadata.json"], HasSubstr("\"label\":\"neg0\""));
}

}  // anonymous namespace
}  // namespace xla
