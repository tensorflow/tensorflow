/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace gtl = ::tensorflow::gtl;

namespace {
using Analysis = IndexedArrayAnalysis;
using UnknownArray = Analysis::UnknownArray;
using ConstantArray = Analysis::ConstantArray;
using ScalarIndexedArray = Analysis::ScalarIndexedArray;
}  // namespace

string IndexedArrayAnalysis::ToString(Array* root) {
  switch (root->kind()) {
    case Array::kUnknown: {
      auto* unknown_tensor = root->as<UnknownArray>();
      return tensorflow::strings::StrCat("%",
                                         unknown_tensor->instruction().name());
    }

    case Array::kConstant: {
      return tensorflow::strings::StrCat(
          "(constant ", ShapeUtil::HumanString(root->shape()), ")");
    }

    case Array::kScalarIndexedConstant:
    case Array::kScalarIndexed: {
      auto* indexed_array = root->as<ScalarIndexedArray>();
      string name = root->kind() == Array::kScalarIndexedConstant
                        ? "scalar-indexed-const"
                        : "scalar-indexed";
      return tensorflow::strings::StrCat(
          "(", name, " ", ToString(indexed_array->source()), " ",
          ToString(indexed_array->indices()), " ", indexed_array->source_dim(),
          "->[", tensorflow::str_util::Join(indexed_array->output_dims(), ","),
          "])");
    }
  }
}

Analysis::Array* IndexedArrayAnalysis::GetArrayFor(
    const HloInstruction* instr) {
  auto it = cache_.find(instr);
  if (it != cache_.end()) {
    return it->second;
  }

  TraverseAndPopulateCache(instr);
  return FindOrDie(cache_, instr);
}

void IndexedArrayAnalysis::TraverseAndPopulateCache(
    const HloInstruction* root) {
  // Depth first search over the DAG, invoking ComputeArrayFor in post order.
  // The HLO instructions already in the cache are considered leaves.

  gtl::InlinedVector<const HloInstruction*, 4> stack;

  enum DfsState { kDiscovered, kVisited };
  gtl::FlatMap<const HloInstruction*, DfsState> dfs_state_map;

  stack.push_back(root);
  InsertOrDie(&dfs_state_map, root, kDiscovered);

  do {
    const HloInstruction* instr = stack.back();
    if (cache_.count(instr)) {
      stack.pop_back();
      continue;
    }

    switch (FindOrDie(dfs_state_map, instr)) {
      case kDiscovered: {
        for (const HloInstruction* operand : instr->operands()) {
          if (!cache_.count(operand)) {
            stack.push_back(operand);
            CHECK(!dfs_state_map.count(operand) ||
                  dfs_state_map[operand] == kDiscovered);
            dfs_state_map[operand] = kDiscovered;
          }
        }
        dfs_state_map[instr] = kVisited;
        break;
      }

      case kVisited:
        stack.pop_back();
        InsertOrDie(&cache_, instr, ComputeArrayFor(instr));
        break;
    }
  } while (!stack.empty());
}

Analysis::Array* IndexedArrayAnalysis::ComputeArrayFor(
    const HloInstruction* instr) {
  Array* computed_array;
  switch (instr->opcode()) {
    default:
      computed_array = nullptr;
      break;
    case HloOpcode::kConstant:
      computed_array = ComputeArrayForConstant(instr->literal());
      break;
    case HloOpcode::kGather:
      computed_array = ComputeArrayForGather(
          instr->shape(), instr->gather_dimension_numbers(),
          instr->gather_window_bounds(), FindOrDie(cache_, instr->operand(0)),
          FindOrDie(cache_, instr->operand(1)));
      break;
  }

  if (!computed_array) {
    computed_array = Construct<UnknownArray>(instr);
  }

  return computed_array;
}

Analysis::Array* IndexedArrayAnalysis::ComputeArrayForConstant(
    const Literal& literal) {
  return Construct<ConstantArray>(&literal);
}

ScalarIndexedArray* IndexedArrayAnalysis::FoldGatherOfGather(
    ScalarIndexedArray* source, Array* indices, int64 source_dim,
    tensorflow::gtl::ArraySlice<int64> output_dims, Shape shape) {
  // We want to transform Gather(Gather(A, X), Y) => Gather(A, Gather(X, Y)).
  // `source` is the inner Gather(A, X).

  Array* a = source->source();
  Array* x = source->indices();
  Array* y = indices;

  // This bit is slightly tricky, so we do a naive "simulation" of the two
  // consecutive gather operations to infer what the composed gather should look
  // like.

  enum class IndexComponent { Ungathered, GatheredFirst, GatheredSecond };

  std::vector<IndexComponent> simulated_index(a->shape().dimensions_size(),
                                              IndexComponent::Ungathered);

  // Simulate the first gather.
  simulated_index.erase(simulated_index.begin() + source->source_dim());
  for (int64 gather_dim : source->output_dims()) {
    simulated_index.insert(simulated_index.begin() + gather_dim,
                           IndexComponent::GatheredFirst);
  }

  // Simulate the second gather.
  simulated_index.erase(simulated_index.begin() + source_dim);
  for (int64 output_dim : output_dims) {
    simulated_index.insert(simulated_index.begin() + output_dim,
                           IndexComponent::GatheredSecond);
  }

  int64 source_dim_for_index_array =
      FindIndex(source->output_dims(), source_dim);
  CHECK_NE(source_dim_for_index_array, source->output_dims().size());

  std::vector<int64> output_dims_for_index_array;
  int64 gathered_index_components_seen = 0;
  for (IndexComponent simulation_dim : simulated_index) {
    if (simulation_dim == IndexComponent::GatheredSecond) {
      output_dims_for_index_array.push_back(gathered_index_components_seen);
    }
    if (simulation_dim != IndexComponent::Ungathered) {
      gathered_index_components_seen++;
    }
  }

  std::vector<int64> dim_sizes_for_composed_index;
  std::vector<int64> output_dims_for_new_gather;
  for (int64 i = 0, e = simulated_index.size(); i < e; i++) {
    if (simulated_index[i] != IndexComponent::Ungathered) {
      dim_sizes_for_composed_index.push_back(shape.dimensions(i));
      output_dims_for_new_gather.push_back(i);
    }
  }

  Array* inner_indices = ConstructScalarIndexedArray(
      x, y, source_dim_for_index_array, output_dims_for_index_array,
      ShapeUtil::MakeShape(x->shape().element_type(),
                           dim_sizes_for_composed_index));
  return ConstructScalarIndexedArray(a, inner_indices, source->source_dim(),
                                     output_dims_for_new_gather,
                                     std::move(shape));
}

Analysis::Array* IndexedArrayAnalysis::ComputeArrayForGather(
    const Shape& shape, const GatherDimensionNumbers& dim_numbers,
    tensorflow::gtl::ArraySlice<int64> window_bounds, Array* source,
    Array* indices) {
  if (dim_numbers.index_vector_dim() != indices->shape().dimensions_size()) {
    return nullptr;
  }

  CHECK_EQ(dim_numbers.gather_dims_to_operand_dims_size(), 1);
  if (!c_binary_search(dim_numbers.elided_window_dims(),
                       dim_numbers.gather_dims_to_operand_dims(0))) {
    return nullptr;
  }

  int64 source_dim = dim_numbers.gather_dims_to_operand_dims(0);
  std::vector<int64> output_dims;
  for (int64 i = 0, e = shape.dimensions_size(); i < e; i++) {
    if (!c_binary_search(dim_numbers.output_window_dims(), i)) {
      output_dims.push_back(i);
    }
  }

  if (auto* indexed = dynamic_cast<ScalarIndexedArray*>(source)) {
    auto it = c_find(indexed->output_dims(), source_dim);
    if (it != indexed->output_dims().end()) {
      return FoldGatherOfGather(indexed, indices, source_dim, output_dims,
                                shape);
    }
  } else if (auto* constant = dynamic_cast<ConstantArray*>(source)) {
    return Construct<ScalarIndexedConstantArray>(constant, indices, source_dim,
                                                 output_dims, shape);
  }

  return Construct<ScalarIndexedArray>(source, indices, source_dim, output_dims,
                                       shape);
}

tensorflow::StringPiece IndexedArrayAnalysisPrinterPass::name() const {
  return "indexed-array-analysis-printer-pass";
}

StatusOr<bool> IndexedArrayAnalysisPrinterPass::Run(HloModule* module) {
  if (!VLOG_IS_ON(2)) {
    return false;
  }

  IndexedArrayAnalysis analysis;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      auto* t = analysis.GetArrayFor(instr);
      if (!dynamic_cast<UnknownArray*>(t) && !dynamic_cast<ConstantArray*>(t)) {
        VLOG(2) << instr->ToString() << "   ->   " << analysis.ToString(t);
      }
    }
  }

  return false;
}

}  // namespace xla
