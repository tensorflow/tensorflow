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

#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"

#include <sys/types.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"

namespace tooling {
namespace visualization_client {
namespace {

using HloGraphAdapterTest = xla::HloHardwareIndependentTestBase;

const char hlo_string_single_computation[] = R"(
HloModule axpy_module

ENTRY axpy.v5 {
  alpha = f32[] parameter(0)
  broadcast = f32[2,4]{1,0} broadcast(alpha), dimensions={}
  x = f32[2,4]{1,0} parameter(1)
  multiply = f32[2,4]{1,0} multiply(broadcast, x)
  y = f32[2,4]{1,0} parameter(2)
  ROOT add = f32[2,4]{1,0} add(multiply, y)
}
)";

const char hlo_string_multiple_computations[] = R"(
HloModule axpy_module

calculate_alpha {
c.0 = f32[] constant(1)
c.1 = f32[] constant(2)
ROOT ret.0 = f32[] multiply(c.0, c.1)
}

calculate_y {
c.2 = f32[] constant(2)
c.3 = f32[] constant(3)
ROOT ret.1 = f32[] add(c.2, c.3)
}

ENTRY axpy_computation {
alpha = f32[] call(), to_apply=calculate_alpha
y = f32[] call(), to_apply=calculate_y
add.0 = f32[] add(alpha, y)
p.0 = f32[] parameter(0)
ROOT add.1 = f32[] add(add.0, p.0)
})";

const char hlo_string_nested_fusion[] = R"(
HloModule SimpleLoop

%fused_computation.inner (param_0.51117: s8[56,4096,4096], param_1:
s32[]) -> s8[1,4096,4096] {
%param_0.51117 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
p1 = s32[]{:T(128)} parameter(1)
%constant.85694 = s32[]{:T(128)} constant(0)

ROOT %dynamic-slice.22040 = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)}
dynamic-slice(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} %param_0.51117,
s32[]{:T(128)} p1, s32[]{:T(128)} %constant.85694, s32[]{:T(128)}
%constant.85694), dynamic_slice_sizes={1,4096,4096}
}

%fused_computation (param_0.51118: s8[56,4096,4096], param_1:
s32[]) -> s8[4096,4096] {
%param_0.51118 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
p2 = s32[]{:T(128)} parameter(1)

%inner.fusion = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} fusion(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} %param_0.51118, s32[]{:T(128)} p2), kind=kLoop, calls=%fused_computation.inner

ROOT %bitcast = s8[4096,4096]{1,0:T(8,128)(4,1)} bitcast(s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} %inner.fusion)
}

ENTRY main {
p0 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
c = s32[]{:T(128)} constant(10)
fusion.0 = s8[4096,4096]{1,0:T(8,128)(4,1)} fusion(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} p0, s32[]{:T(128)} c), kind=kLoop, calls=%fused_computation
ROOT out = (s8[4096,4096]) tuple(fusion.0)
})";

const char hlo_string_with_constant[] = R"(
HloModule test_module
ENTRY test {
  c0 = f32[] constant(1)
  c1 = f32[] constant(2)
  ROOT add = f32[] add(c0, c1)
}
)";

const char hlo_string_fusion_broadcast_constant[] = R"(
HloModule test_module
fused_computation {
  param = f32[2]{0} parameter(0)
  constant = f32[] constant(2)
  broadcast = f32[2]{0} broadcast(constant), dimensions={}
  ROOT add = f32[2]{0} add(param, broadcast)
}
ENTRY test {
  p0 = f32[2]{0} parameter(0)
  ROOT fusion = f32[2]{0} fusion(p0), kind=kLoop, calls=fused_computation
}
)";

const char hlo_string_get_tuple_element[] = R"(
HloModule axpy_module

ENTRY test {
  p0 = f32[] parameter(0)
  tuple = (f32[], f32[]) tuple(p0, p0)
  p1 = f32[] parameter(1)
  gte.0 = f32[] get-tuple-element(tuple), index=0
  gte.1 = f32[] get-tuple-element(tuple), index=1
  ROOT add.0 = f32[] add(gte.0, p1)
}
)";

const char hlo_string_get_tuple_element_multiple[] = R"(
HloModule multiple_gte_operands
ENTRY test {
  p0 = f32[] parameter(0)
  tuple = (f32[], f32[]) tuple(p0, p0)
  gte.0 = f32[] get-tuple-element(tuple), index=0
  gte.1 = f32[] get-tuple-element(tuple), index=1
  ROOT add.0 = f32[] add(gte.0, gte.1)
}
)";

const char hlo_string_fusion_parameter_gte[] = R"(
HloModule fusion_parameter_gte
fused_computation {
  fp0 = f32[] parameter(0)
  ROOT add = f32[] add(fp0, fp0)
}
ENTRY test {
  p0 = f32[] parameter(0)
  tuple = (f32[], f32[]) tuple(p0, p0)
  gte0 = f32[] get-tuple-element(tuple), index=0
  fusion = f32[] fusion(gte0), kind=kLoop, calls=fused_computation
  ROOT root = f32[] add(fusion, fusion)
}
)";

// Max users to render is 16.
const char hlo_string_with_too_many_users[] = R"(
HloModule axpy_module

ENTRY axpy.v5 {
  alpha = f32[] parameter(0)
  broadcast = f32[2,4]{1,0} broadcast(alpha), dimensions={}
  x = f32[2,4]{1,0} parameter(1)
  sqrt.0 = f32[2,4]{1,0} sqrt(x)
  multiply.0 = f32[2,4]{1,0} multiply(broadcast, sqrt.0)
  multiply.1 = f32[2,4]{1,0} multiply(multiply.0, sqrt.0)
  multiply.2 = f32[2,4]{1,0} multiply(multiply.1, sqrt.0)
  multiply.3 = f32[2,4]{1,0} multiply(multiply.2, sqrt.0)
  multiply.4 = f32[2,4]{1,0} multiply(multiply.3, sqrt.0)
  multiply.5 = f32[2,4]{1,0} multiply(multiply.4, sqrt.0)
  multiply.6 = f32[2,4]{1,0} multiply(multiply.5, sqrt.0)
  multiply.7 = f32[2,4]{1,0} multiply(multiply.6, sqrt.0)
  multiply.8 = f32[2,4]{1,0} multiply(multiply.7, sqrt.0)
  multiply.9 = f32[2,4]{1,0} multiply(multiply.8, sqrt.0)
  multiply.10 = f32[2,4]{1,0} multiply(multiply.9, sqrt.0)
  multiply.11 = f32[2,4]{1,0} multiply(multiply.10, sqrt.0)
  multiply.12 = f32[2,4]{1,0} multiply(multiply.11, sqrt.0)
  multiply.13 = f32[2,4]{1,0} multiply(multiply.12, sqrt.0)
  multiply.14 = f32[2,4]{1,0} multiply(multiply.13, sqrt.0)
  multiply.15 = f32[2,4]{1,0} multiply(multiply.14, sqrt.0)
  multiply.16 = f32[2,4]{1,0} multiply(multiply.15, sqrt.0)
  multiply.17 = f32[2,4]{1,0} multiply(multiply.16, sqrt.0)
  multiply.18 = f32[2,4]{1,0} multiply(multiply.17, sqrt.0)
  y = f32[2,4]{1,0} parameter(2)
  ROOT add = f32[2,4]{1,0} add(multiply.18, y)
}
)";

const char hlo_string_long_shape[] = R"(
HloModule long_shape_module

ENTRY test {
  p0 = bf16[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0} parameter(0)
  ROOT add = bf16[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0} add(p0, p0)
}
)";

llvm::json::Value ParseJson(absl::string_view json) {
  auto val = llvm::json::parse(llvm::StringRef(json.data(), json.size()));
  if (!val) {
    ADD_FAILURE() << "Failed to parse JSON: "
                  << llvm::toString(val.takeError());
    return llvm::json::Value(nullptr);
  }
  return std::move(*val);
}

const llvm::json::Object* GetFirstSubgraph(
    const llvm::json::Value& parsed_json) {
  const llvm::json::Array* graphs = parsed_json.getAsArray();
  if (!graphs || graphs->empty()) {
    return nullptr;
  }
  const llvm::json::Object* graph = (*graphs)[0].getAsObject();
  if (!graph) {
    return nullptr;
  }
  const llvm::json::Array* subgraphs = graph->getArray("subgraphs");
  if (!subgraphs || subgraphs->empty()) {
    return nullptr;
  }
  return (*subgraphs)[0].getAsObject();
}

int GetNumberOfNodes(absl::string_view json) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return 0;
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  return nodes ? nodes->size() : 0;
}

int GetNumberOfSubgraphs(absl::string_view json) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Array* graphs = parsed_json.getAsArray();
  if (!graphs || graphs->empty()) {
    return 0;
  }
  const llvm::json::Object* graph = (*graphs)[0].getAsObject();
  if (!graph) {
    return 0;
  }
  const llvm::json::Array* subgraphs = graph->getArray("subgraphs");
  return subgraphs ? subgraphs->size() : 0;
}

int GetNumberOfIncomingEdges(absl::string_view json,
                             absl::string_view node_id) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return 0;
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  if (!nodes) {
    return 0;
  }
  for (const auto& node_val : *nodes) {
    const llvm::json::Object* node = node_val.getAsObject();
    if (!node) {
      continue;
    }
    std::optional<llvm::StringRef> id = node->getString("id");
    if (id && *id == llvm::StringRef(node_id.data(), node_id.size())) {
      const llvm::json::Array* edges = node->getArray("incomingEdges");
      return edges ? edges->size() : 0;
    }
  }
  return 0;
}

int GetNumberOfOutputsMetadata(absl::string_view json,
                               absl::string_view node_id) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return 0;
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  if (!nodes) {
    return 0;
  }
  for (const auto& node_val : *nodes) {
    const llvm::json::Object* node = node_val.getAsObject();
    if (!node) {
      continue;
    }
    std::optional<llvm::StringRef> id = node->getString("id");
    if (id && *id == llvm::StringRef(node_id.data(), node_id.size())) {
      const llvm::json::Array* outputs = node->getArray("outputsMetadata");
      return outputs ? outputs->size() : 0;
    }
  }
  return 0;
}

int GetNumberOfNamespaces(absl::string_view json) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return 0;
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  if (!nodes) {
    return 0;
  }
  absl::flat_hash_set<std::string> namespaces;
  for (const auto& node_val : *nodes) {
    const llvm::json::Object* node = node_val.getAsObject();
    if (!node) {
      continue;
    }
    std::optional<llvm::StringRef> ns = node->getString("namespace");
    if (ns) {
      namespaces.insert(std::string(*ns));
    }
  }
  return namespaces.size();
}

int GetNumberOfNodesOfNamespace(absl::string_view json,
                                absl::string_view namespace_name) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return 0;
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  if (!nodes) {
    return 0;
  }
  int count = 0;
  for (const auto& node_val : *nodes) {
    const llvm::json::Object* node = node_val.getAsObject();
    if (!node) {
      continue;
    }
    std::optional<llvm::StringRef> ns = node->getString("namespace");
    if (ns &&
        *ns == llvm::StringRef(namespace_name.data(), namespace_name.size())) {
      count++;
    }
  }
  return count;
}

std::string GetAttribute(absl::string_view json, absl::string_view node_id,
                         absl::string_view attr_name) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return "";
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  if (!nodes) {
    return "";
  }
  for (const auto& node_val : *nodes) {
    const llvm::json::Object* node = node_val.getAsObject();
    if (!node) {
      continue;
    }
    std::optional<llvm::StringRef> id = node->getString("id");
    if (id && *id == llvm::StringRef(node_id.data(), node_id.size())) {
      const llvm::json::Array* attrs = node->getArray("attrs");
      if (!attrs) {
        return "";
      }
      for (const auto& attr_val : *attrs) {
        const llvm::json::Object* attr = attr_val.getAsObject();
        if (!attr) {
          continue;
        }
        std::optional<llvm::StringRef> key = attr->getString("key");
        if (key &&
            *key == llvm::StringRef(attr_name.data(), attr_name.size())) {
          std::optional<llvm::StringRef> val = attr->getString("value");
          return val ? std::string(*val) : "";
        }
      }
    }
  }
  return "";
}

std::string GetOpcode(absl::string_view json, absl::string_view node_id) {
  return GetAttribute(json, node_id, "opcode");
}

std::string GetGetTupleElementIndexAttribute(absl::string_view json,
                                             absl::string_view node_id) {
  return GetAttribute(json, node_id, "get_tuple_element_index");
}

std::string GetInlinedOperandsAttribute(absl::string_view json,
                                        absl::string_view node_id) {
  return GetAttribute(json, node_id, "inlined_operands");
}

std::string GetLiteral(absl::string_view json, absl::string_view node_id) {
  return GetAttribute(json, node_id, "literal");
}

std::string GetSourceStack(absl::string_view json, absl::string_view node_id) {
  return GetAttribute(json, node_id, "source_stack");
}

std::string GetOutputsMetadataAttribute(absl::string_view json,
                                        absl::string_view node_id,
                                        absl::string_view output_id_str,
                                        absl::string_view attr_key) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return "";
  }
  const llvm::json::Array* nodes = subgraph->getArray("nodes");
  if (!nodes) {
    return "";
  }
  for (const auto& node_val : *nodes) {
    const llvm::json::Object* node = node_val.getAsObject();
    if (!node) {
      continue;
    }
    std::optional<llvm::StringRef> id = node->getString("id");
    if (id && *id == llvm::StringRef(node_id.data(), node_id.size())) {
      const llvm::json::Array* outputs_metadata =
          node->getArray("outputsMetadata");
      if (!outputs_metadata) {
        return "";
      }
      for (const auto& output_meta_val : *outputs_metadata) {
        const llvm::json::Object* output_meta = output_meta_val.getAsObject();
        if (!output_meta) {
          continue;
        }
        std::optional<llvm::StringRef> out_id = output_meta->getString("id");
        if (out_id && *out_id == llvm::StringRef(output_id_str.data(),
                                                 output_id_str.size())) {
          const llvm::json::Array* attrs = output_meta->getArray("attrs");
          if (!attrs) {
            return "";
          }
          for (const auto& attr_val : *attrs) {
            const llvm::json::Object* attr = attr_val.getAsObject();
            if (!attr) {
              continue;
            }
            std::optional<llvm::StringRef> key = attr->getString("key");
            if (key &&
                *key == llvm::StringRef(attr_key.data(), attr_key.size())) {
              std::optional<llvm::StringRef> val = attr->getString("value");
              return val ? std::string(*val) : "";
            }
          }
        }
      }
    }
  }
  return "";
}

std::string GetGroupNodeAttribute(absl::string_view json,
                                  absl::string_view group_name,
                                  absl::string_view attr_key) {
  llvm::json::Value parsed_json = ParseJson(json);
  const llvm::json::Object* subgraph = GetFirstSubgraph(parsed_json);
  if (!subgraph) {
    return "";
  }
  const llvm::json::Object* group_node_attributes =
      subgraph->getObject("groupNodeAttributes");
  if (!group_node_attributes) {
    return "";
  }
  const llvm::json::Object* group =
      group_node_attributes->getObject(group_name);
  if (!group) {
    return "";
  }
  std::optional<llvm::StringRef> val = group->getString(attr_key);
  return val ? std::string(*val) : "";
}

int countStringLines(const std::string& str) {
  int numLines = 1;
  for (char c : str) {
    if (c == '\n') {
      numLines++;
    }
  }
  return numLines;
}

int GetNumberOfOperands(std::string json, absl::string_view node_id) {
  std::string val = GetAttribute(json, node_id, "operands");
  return val.empty() ? 0 : countStringLines(val);
}

int GetNumberOfUsers(std::string json, absl::string_view node_id) {
  std::string val = GetAttribute(json, node_id, "users");
  return val.empty() ? 0 : countStringLines(val);
}

TEST_F(HloGraphAdapterTest, SingleComputation) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_single_computation));

  auto entry_computation = hlo_module->entry_computation();

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));

  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 7);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 1);

  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "broadcast"), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "multiply"), 2);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add"), 2);

  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "alpha"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "broadcast"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "x"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "multiply"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "y"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "add"), 0);
}

TEST_F(HloGraphAdapterTest, SingleComputationNeighborhood) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_single_computation));

  auto root_instruction = hlo_module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*root_instruction, 2));

  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 6);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add"), 2);
}

TEST_F(HloGraphAdapterTest, MultipleComputations) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_multiple_computations));

  auto entry_computation = hlo_module->entry_computation();

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));

  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 10);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 3);

  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "calculate_alpha"), 0);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "calculate_y"), 0);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "axpy_computation"), 6);

  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "ret.1"), 0);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "alpha"), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "y"), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add.1"), 2);

  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "c.0"), 0);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "c.1"), 0);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "ret.0"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "alpha"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "add.0"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "add.1"), 0);
}

TEST_F(HloGraphAdapterTest, MultipleComputationsNeighborhood) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_multiple_computations));

  auto alpha_instruction = FindInstruction(hlo_module.get(), "alpha");

  // Root as alpha, radius as 1.
  ASSERT_OK_AND_ASSIGN(std::string json_output_0,
                       HloGraphAdapter(*alpha_instruction, 1));
  EXPECT_EQ(GetNumberOfSubgraphs(json_output_0), 1);
  EXPECT_EQ(GetNumberOfNamespaces(json_output_0), 2);
  EXPECT_EQ(GetNumberOfNodes(json_output_0), 5);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output_0, "calculate_alpha"), 0);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output_0, "axpy_computation"), 3);

  // Root as alpha, radius as 2.
  ASSERT_OK_AND_ASSIGN(std::string json_output_1,
                       HloGraphAdapter(*alpha_instruction, 2));
  EXPECT_EQ(GetNumberOfSubgraphs(json_output_1), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output_1), 9);
  EXPECT_EQ(GetNumberOfNamespaces(json_output_1), 3);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output_1, "calculate_alpha"), 0);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output_1, "calculate_y"), 0);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output_1, "axpy_computation"), 5);
}

TEST_F(HloGraphAdapterTest, NestedFusionComputation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto entry_computation = hlo_module->entry_computation();

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));

  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 11);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 3);

  // Testing fusion node deduplication.
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "main"), 3);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "main/fusion.0"), 4);
  EXPECT_EQ(
      GetNumberOfNodesOfNamespace(json_output, "main/fusion.0/inner.fusion"),
      4);

  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "bitcast"), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "out"), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "dynamic-slice.22040"), 2);
}

TEST_F(HloGraphAdapterTest, NestedFusionInstruction) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto fusion_instruction = FindInstruction(hlo_module.get(), "fusion.0");

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*fusion_instruction, 0));

  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 9);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 3);

  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "main/fusion.0"), 4);
  EXPECT_EQ(
      GetNumberOfNodesOfNamespace(json_output, "main/fusion.0/inner.fusion"),
      4);

  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "bitcast"), 1);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "dynamic-slice.22040"), 2);

  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "p1"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "param_0.51117"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "p2"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "param_0.51118"), 1);
  EXPECT_EQ(GetNumberOfOutputsMetadata(json_output, "dynamic-slice.22040"), 1);
}

TEST_F(HloGraphAdapterTest, Opcode) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto entry_computation = hlo_module->entry_computation();
  auto root_instruction = entry_computation->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));
  EXPECT_EQ(GetOpcode(json_output, root_instruction->name()), "tuple");
}

TEST_F(HloGraphAdapterTest, SourceStack) {
  const std::string text = R"(
    HloModule a_module

    ENTRY main {
      %c = s32[] constant(1)
      ROOT %result = s32[] parameter(0)
    }
    )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));

  auto module_proto = module->ToProto();
  auto index = module_proto.mutable_stack_frame_index();
  index->add_file_names("main.py");
  index->add_function_names("func1");
  auto location = index->add_file_locations();
  location->set_file_name_id(1);
  location->set_function_name_id(1);
  location->set_line(10);
  location->set_column(5);

  auto frame = index->add_stack_frames();
  frame->set_file_location_id(1);

  // Set the stack frame id of the root instruction.
  for (auto& computation : *module_proto.mutable_computations()) {
    if (computation.id() == module_proto.entry_computation_id()) {
      for (auto& instruction : *computation.mutable_instructions()) {
        if (instruction.id() == computation.root_id()) {
          instruction.mutable_metadata()->set_stack_frame_id(1);
        }
      }
    }
  }

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> module_with_stack_frames,
      xla::HloModule::CreateFromProto(module_proto, module->config()));
  auto entry_computation = module_with_stack_frames->entry_computation();
  auto root_instruction = entry_computation->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));
  EXPECT_EQ(GetSourceStack(json_output, root_instruction->name()),
            "main.py:10:5\n");
}

TEST_F(HloGraphAdapterTest, BackendConfigSimple) {
  const std::string hlo_string = R"(
    HloModule backend_config_module
    ENTRY entry {
      p0 = f32[] parameter(0)
      ROOT result = f32[] sqrt(p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto* instruction = module->entry_computation()->root_instruction();
  const std::string backend_config = R"json({"config":"test"})json";
  instruction->set_raw_backend_config_string(backend_config);
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*module->entry_computation()));
  EXPECT_EQ(GetAttribute(json_output, "result", "backend_config"),
            R"json({"config":"test"})json");
}

TEST_F(HloGraphAdapterTest, BackendConfigProcessing) {
  const std::string hlo_string = R"(
    HloModule backend_config_module
    ENTRY entry {
      p0 = f32[] parameter(0)
      ROOT result = f32[] sqrt(p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto* instruction = module->entry_computation()->root_instruction();
  const std::string backend_config = R"json({
    "barrier_config": {
      "barrier_type": "CUSTOM",
      "id": 8
    },
    "flag_configs": [],
    "custom_call_config": "some_binary"
  })json";
  instruction->set_raw_backend_config_string(backend_config);
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*module->entry_computation()));
  EXPECT_EQ(GetAttribute(json_output, "result", "backend_config"),
            R"json({"barrier_config":{"barrier_type":"CUSTOM","id":8}})json");
}

TEST_F(HloGraphAdapterTest, BackendConfigEmptyAfterProcessing) {
  const std::string hlo_string = R"(
    HloModule backend_config_module
    ENTRY entry {
      p0 = f32[] parameter(0)
      ROOT result = f32[] sqrt(p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto* instruction = module->entry_computation()->root_instruction();
  const std::string backend_config = R"json({
    "custom_call_config":"some_binary",
    "empty_list":[]
  })json";
  instruction->set_raw_backend_config_string(backend_config);
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*module->entry_computation()));
  EXPECT_EQ(GetAttribute(json_output, "result", "backend_config"), "");
}

TEST_F(HloGraphAdapterTest, BackendConfigInvalidJson) {
  const std::string hlo_string = R"(
    HloModule backend_config_module
    ENTRY entry {
      p0 = f32[] parameter(0)
      ROOT result = f32[] sqrt(p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto* instruction = module->entry_computation()->root_instruction();
  instruction->set_raw_backend_config_string("invalid-json");
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*module->entry_computation()));
  EXPECT_EQ(GetAttribute(json_output, "result", "backend_config"),
            "invalid-json");
}

TEST_F(HloGraphAdapterTest, BackendConfigNotJsonObject) {
  const std::string hlo_string = R"(
    HloModule backend_config_module
    ENTRY entry {
      p0 = f32[] parameter(0)
      ROOT result = f32[] sqrt(p0)
    })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  auto* instruction = module->entry_computation()->root_instruction();
  instruction->set_raw_backend_config_string("[1,2]");
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*module->entry_computation()));
  EXPECT_EQ(GetAttribute(json_output, "result", "backend_config"), "[1,2]");
}

TEST_F(HloGraphAdapterTest, InstructionRadiusIncludeParentComputation) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_multiple_computations));

  auto root_instruction = hlo_module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*root_instruction, 1));
  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 4);
}

TEST_F(HloGraphAdapterTest, InstructionRadiusIncludeFusionComputation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto fusion_instruction = FindInstruction(hlo_module.get(), "fusion.0");
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*fusion_instruction, 1));
  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  // Nested fusion is not included.
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 3);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "fusion.0"), 0);
  EXPECT_EQ(GetNumberOfNodesOfNamespace(json_output, "main"), 3);
}

TEST_F(HloGraphAdapterTest, GetTupleElementFolding) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_get_tuple_element));

  HloAdapterOption options = {.get_tuple_element_folding = true};
  ASSERT_OK_AND_ASSIGN(
      std::string json_output,
      HloGraphAdapter(*hlo_module->entry_computation(), options));

  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 5);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add.0"), 2);
  EXPECT_EQ(GetGetTupleElementIndexAttribute(json_output, "add.0"),
            "operand 0: tuple-element 0 of tuple f32[]");
}

TEST_F(HloGraphAdapterTest, GetTupleElementFoldingMultiple) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_get_tuple_element_multiple));

  HloAdapterOption options = {.get_tuple_element_folding = true};
  ASSERT_OK_AND_ASSIGN(
      std::string json_output,
      HloGraphAdapter(*hlo_module->entry_computation(), options));

  EXPECT_EQ(GetGetTupleElementIndexAttribute(json_output, "add.0"),
            "operand 0: tuple-element 0 of tuple f32[]\noperand 1: "
            "tuple-element 1 of tuple f32[]");
}

TEST_F(HloGraphAdapterTest, GetTupleElementFoldingFusionParameter) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_fusion_parameter_gte));

  HloAdapterOption options = {.get_tuple_element_folding = true};
  ASSERT_OK_AND_ASSIGN(
      std::string json_output,
      HloGraphAdapter(*hlo_module->entry_computation(), options));
  EXPECT_EQ(GetGetTupleElementIndexAttribute(json_output, "fp0"),
            "tuple-element 0 of tuple f32[]");
}

TEST_F(HloGraphAdapterTest, ConstantFolding) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_with_constant));

  HloAdapterOption options = {.constant_folding = true};
  ASSERT_OK_AND_ASSIGN(
      std::string json_output,
      HloGraphAdapter(*hlo_module->entry_computation(), options));

  EXPECT_EQ(GetNumberOfNodes(json_output), 2);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add"), 0);
  EXPECT_EQ(GetInlinedOperandsAttribute(json_output, "add"),
            "operand 0 = f32[] 1\noperand 1 = f32[] 2");
}

TEST_F(HloGraphAdapterTest, ConstantFoldingDisabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_with_constant));

  HloAdapterOption options = {.constant_folding = false};
  ASSERT_OK_AND_ASSIGN(
      std::string json_output,
      HloGraphAdapter(*hlo_module->entry_computation(), options));

  EXPECT_EQ(GetNumberOfNodes(json_output), 4);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add"), 2);
  EXPECT_EQ(GetLiteral(json_output, "c0"), "f32[] 1");
  EXPECT_EQ(GetLiteral(json_output, "c1"), "f32[] 2");
  EXPECT_EQ(GetInlinedOperandsAttribute(json_output, "add"), "");
}

TEST_F(HloGraphAdapterTest, ConstantFoldingFusedBroadcast) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_fusion_broadcast_constant));

  HloAdapterOption options = {.constant_folding = true};
  ASSERT_OK_AND_ASSIGN(
      std::string json_output,
      HloGraphAdapter(*hlo_module->entry_computation(), options));
  EXPECT_EQ(GetNumberOfNodes(json_output), 5);
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "add"), 1);
  EXPECT_EQ(GetInlinedOperandsAttribute(json_output, "add"),
            "operand 1 = f32[2]{0} 2");
}

TEST_F(HloGraphAdapterTest, ComputationPinNode) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_single_computation));

  auto entry_computation = hlo_module->entry_computation();

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, "Computation: axpy.v5"), 0);
}

TEST_F(HloGraphAdapterTest, InstructionID) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_single_computation));

  auto root_instruction = hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(GetInstructionId(root_instruction), "add");

  auto multiply_instruction = FindInstruction(hlo_module.get(), "multiply");
  EXPECT_EQ(GetInstructionId(multiply_instruction), "multiply");

  auto x_instruction = FindInstruction(hlo_module.get(), "x");
  EXPECT_EQ(GetInstructionId(x_instruction), "x");

  auto y_instruction = FindInstruction(hlo_module.get(), "y");
  EXPECT_EQ(GetInstructionId(y_instruction), "y");
}

TEST_F(HloGraphAdapterTest, ComputationID) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_single_computation));

  auto entry_computation = hlo_module->entry_computation();
  EXPECT_EQ(GetComputationId(entry_computation), "axpy.v5");
}

TEST_F(HloGraphAdapterTest, NonRootTupleOperands) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_get_tuple_element));
  // Will include the get tuple element node.
  HloAdapterOption options = {.get_tuple_element_folding = false};
  // Should omit P0 which is operands of non root tuple.
  auto instruction = FindInstruction(hlo_module.get(), "gte.0");
  auto non_root_tuple_instruction = FindInstruction(hlo_module.get(), "tuple");
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*instruction, 2, options));
  EXPECT_EQ(GetNumberOfNodes(json_output), 6);
  EXPECT_EQ(
      GetNumberOfOperands(json_output, non_root_tuple_instruction->name()), 2);
}

TEST_F(HloGraphAdapterTest, TooManyUsers) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      ParseAndReturnVerifiedModule(hlo_string_with_too_many_users));
  auto instruction1 = FindInstruction(hlo_module.get(), "sqrt.0");
  ASSERT_OK_AND_ASSIGN(std::string json_output_1,
                       HloGraphAdapter(*instruction1, 5));
  // None of the users are included for sqrt.0 since there are too many.
  EXPECT_EQ(GetNumberOfNodes(json_output_1), 3);
  EXPECT_EQ(GetNumberOfUsers(json_output_1, instruction1->name()), 19);

  auto instruction2 = FindInstruction(hlo_module.get(), "multiply.18");
  ASSERT_OK_AND_ASSIGN(std::string json_output_2,
                       HloGraphAdapter(*instruction2, 5));
  // Operands (inputs) within radius are always included for multiply.18.
  EXPECT_EQ(GetNumberOfNodes(json_output_2), 11);
}

TEST_F(HloGraphAdapterTest, NodeAttributesForFusionComputationPinnedNode) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto entry_computation = hlo_module->entry_computation();
  auto fusion_instruction = FindInstruction(hlo_module.get(), "fusion.0");
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));
  EXPECT_EQ(GetOpcode(json_output, fusion_instruction->name()), "fusion:kLoop");
  EXPECT_EQ(GetNumberOfIncomingEdges(json_output, fusion_instruction->name()),
            0);
}

TEST_F(HloGraphAdapterTest, NestedFusionComputationNotExpanded) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto entry_computation = hlo_module->entry_computation();

  ASSERT_OK_AND_ASSIGN(
      GraphCollection graph,
      HloToGraph(
          *entry_computation,
          [](const xla::HloInstruction* instruction) { return true; },
          [](const xla::HloInstruction* instruction,
             const xla::HloComputation* computation) {
            return computation->name() != "fused_computation.inner";
          }));
  std::string json_output =
      llvm::formatv("{0:2}", llvm::json::Value(graph.Json())).str();
  EXPECT_EQ(GetNumberOfSubgraphs(json_output), 1);
  EXPECT_EQ(GetNumberOfNodes(json_output), 8);  // inner has 4 nodes.
  EXPECT_EQ(GetNumberOfNamespaces(json_output), 2);
}

TEST_F(HloGraphAdapterTest, ShapeStringNoTruncation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_long_shape));

  auto entry_computation = hlo_module->entry_computation();

  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));

  const std::string expected_shape =
      "bf16[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]{15,14,13,12,11,10,9,8,7,6,5,4,3,2,"
      "1,0}";

  std::string shape_with_layout_p0 =
      GetAttribute(json_output, "p0", "shape_with_layout");
  EXPECT_EQ(shape_with_layout_p0, expected_shape);

  std::string shape_with_layout_add =
      GetAttribute(json_output, "add", "shape_with_layout");
  EXPECT_EQ(shape_with_layout_add, expected_shape);

  EXPECT_EQ(
      GetOutputsMetadataAttribute(json_output, "p0", "0", "shape_with_layout"),
      shape_with_layout_p0);
}

TEST_F(HloGraphAdapterTest, GroupNodeAttributes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_string_nested_fusion));

  auto entry_computation = hlo_module->entry_computation();
  ASSERT_OK_AND_ASSIGN(std::string json_output,
                       HloGraphAdapter(*entry_computation));

  // fusion.0 is in main.
  EXPECT_EQ(GetGroupNodeAttribute(json_output, "main/fusion.0", "opcode"),
            "fusion:kLoop");
  // inner.fusion is in main/fusion.0.
  EXPECT_EQ(GetGroupNodeAttribute(json_output, "main/fusion.0/inner.fusion",
                                  "opcode"),
            "fusion:kLoop");
}

}  // namespace
}  // namespace visualization_client
}  // namespace tooling
