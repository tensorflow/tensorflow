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

#include "tensorflow/dtensor/mlir/expansions/einsum_spmd_expander.h"

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

bool Equal(const ShardingSpec& a, const ShardingSpec& b) {
  return a.sharding_spec() == b.sharding_spec();
}

}  // namespace

// Einsum, like reductions, is implemented as a local operation followed by
// an all-reduce over dimensions that have been reduced.
StatusOr<mlir::Operation*> EinsumSPMDExpander::ExpandOp(mlir::Operation* op) {
  std::vector<Layout> input_layouts(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    TF_ASSIGN_OR_RETURN(auto layout,
                        ExtractLayoutFromOperand(op->getOperand(i)));
    if (!layout) return errors::InvalidArgument("missing layout for input ", i);
    input_layouts[i] = layout.value();
  }
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  if (!output_layout)
    return errors::InvalidArgument("is missing output layout.");

  std::vector<mlir::Value> new_inputs;
  Layout layout_after_einsum;
  absl::flat_hash_set<std::string> reduce_dims;
  TF_RETURN_IF_ERROR(MaybeRelayoutInputs(input_layouts, op,
                                         output_layout.value(), reduce_dims,
                                         layout_after_einsum, new_inputs));

  mlir::OpBuilder builder(op);
  mlir::BlockAndValueMapping mapping;
  for (int i = 0; i < op->getNumOperands(); ++i)
    mapping.map(op->getOperand(i), new_inputs[i]);
  mlir::Operation* new_op = builder.clone(*op, mapping);
  // Note that the output shape of new_op is cloned from op, so we need to
  // update to the local shape.
  new_op = InferSPMDExpandedLocalShape(new_op);

  if (!reduce_dims.empty()) {
    TF_ASSIGN_OR_RETURN(
        new_op, EmitAllReduce(builder, layout_after_einsum, reduce_dims, new_op,
                              kReduceOpAdd));
  }

  TF_ASSIGN_OR_RETURN(auto final_output,
                      EmitRelayout(new_op->getOpResult(0), layout_after_einsum,
                                   output_layout.value()));

  op->getOpResult(0).replaceAllUsesWith(final_output);
  op->erase();

  return final_output.getDefiningOp();
}

// TODO(power) -- we use a simplified equation parser here. consider
// refactoring einsum_spmd_expander and reusing the TF parser.
//
// Given the input equation, this has 3 outputs:
// reduced_dims: The set of mesh dimesions that we need to all reduce over.
// input_mappings: for each equation input, the map from the equation labels
//   to the tensor dimension of that label.
// output_mapping: as above, but for the equation output.
Status ExtractEquationRelations(
    absl::string_view equation, absl::flat_hash_set<char>& reduced_dims,
    std::vector<absl::flat_hash_map<char, std::vector<int>>>& input_mappings,
    absl::flat_hash_map<char, std::vector<int>>& output_mapping) {
  std::pair<std::string, std::string> parts = absl::StrSplit(equation, "->");
  absl::flat_hash_set<char> non_reduced_dims;

  // Mark kept dimensions from the output.
  for (const auto& char_and_index : llvm::enumerate(parts.second)) {
    // TODO(b/172691887): Support Broadcasting for einsum.
    if (char_and_index.value() == '.')
      return errors::Unimplemented(
          "Broadcasting is unimplemented for einsum. Received equation ",
          equation);
    non_reduced_dims.insert(char_and_index.value());

    // Construct the output mapping, note that output is not allowed to have
    // duplicate labels. This would mean that the datatype of output_mapping
    // should really be absl::flat_hash_map<char, int>, but having the same
    // type as the input_mapping keeps GetSpecsFromLabelsAndMap simpler.
    if (output_mapping.contains(char_and_index.value()))
      return errors::InvalidArgument("received label ", char_and_index.value(),
                                     " multiple times in the "
                                     "output of einsum equation ",
                                     equation);

    output_mapping[char_and_index.value()].emplace_back(char_and_index.index());
  }

  std::vector<std::string> inputs = absl::StrSplit(parts.first, ',');
  // Note that the TF einsum op only supports at most 2 inputs. This is slightly
  // confusing as the tf.einsum interface actually supports > 2 inputs.
  if (inputs.size() > 2)
    return errors::InvalidArgument(
        "einsum only supports at most 2 inputs received equation ", equation,
        " which has ", inputs.size(), " inputs");

  input_mappings.resize(inputs.size());

  // Compute the input mappings and keep track of labels which are reduced.
  for (int i = 0; i < inputs.size(); ++i) {
    for (const auto& char_and_index : llvm::enumerate(inputs[i])) {
      input_mappings[i][char_and_index.value()].emplace_back(
          char_and_index.index());
      if (!non_reduced_dims.contains(char_and_index.value()))
        reduced_dims.insert(char_and_index.value());
    }
  }

  return Status::OK();
}

// For a set of layouts and mappings from labels to offsets in the layouts,
// return a mappings of labels to ShardingSpecs.
// If the label appears multiples with different mesh dimensions in the
// sharding specs we raise an error if replicate_incompatible_dimensions is
// false. Otherwise we treat the dimension as if it were unsharded.
// Labels with unsharded dimensions are not recorded in the output.
StatusOr<absl::flat_hash_map<char, ShardingSpec>> GetLabelToShardingSpec(
    bool replicate_incompatible_dimensions, const std::vector<Layout>& layouts,
    const std::vector<absl::flat_hash_map<char, std::vector<int>>>& mappings) {
  absl::flat_hash_map<char, ShardingSpec> label_to_sharding_spec;
  absl::flat_hash_set<char> incompatible_labels;

  // For each mapping, identify the mesh dimension and whether it has been
  // reduced away.
  for (int index = 0; index < layouts.size(); ++index) {
    for (const auto& mapping : mappings[index]) {
      for (int offset : mapping.second) {
        if (offset >= layouts[index].rank())
          return errors::InvalidArgument(
              llvm::formatv(
                  "specified einsum equation for operand {0} tried to "
                  "read layout at offset {1}, but layout is {2} with rank "
                  "{3}",
                  index, offset, layouts[index].ToString(),
                  layouts[index].rank())
                  .str());

        const ShardingSpec& sharding_spec = layouts[index].dim(offset);

        if (label_to_sharding_spec.contains(mapping.first)) {
          if (Layout::IsShardedSpec(sharding_spec) &&
              !Equal(label_to_sharding_spec[mapping.first], sharding_spec)) {
            if (!replicate_incompatible_dimensions)
              return errors::InvalidArgument(
                  llvm::formatv(
                      "incompatible mesh dimensions in equation, label '{0}' "
                      "is mapped to mesh dimension '{1}' and '{2}'",
                      mapping.first, sharding_spec.sharding_spec(),
                      label_to_sharding_spec[mapping.first].sharding_spec())
                      .str());
            else
              incompatible_labels.insert(mapping.first);
          }
        } else if (Layout::IsShardedSpec(sharding_spec)) {
          label_to_sharding_spec[mapping.first] = sharding_spec;
        }
      }
    }
  }

  // For labels that had incompatible dimensions, treat them as replicated.
  // We would need to insert some all to all in the SPMD expansion for these.
  for (char label : incompatible_labels) label_to_sharding_spec.erase(label);

  return label_to_sharding_spec;
}

// The layout we generated may be invalid as the same dimension may be used
// multiple times. E.g. ab,bc->ac (i.e. matmul) with a and c sharded over the
// same dim. In this case we mark all such dimensions as replicated.
StatusOr<Layout> VerifyOrFixLayout(
    std::pair<std::vector<ShardingSpec>, absl::flat_hash_map<std::string, int>>
        pair,
    const Mesh& mesh) {
  std::vector<ShardingSpec> sharding_specs = pair.first;
  absl::flat_hash_map<std::string, int> dimension_use_count = pair.second;
  for (int i = 0; i < sharding_specs.size(); ++i)
    if (Layout::IsShardedSpec(sharding_specs[i]) &&
        dimension_use_count[sharding_specs[i].sharding_spec()] > 1)
      sharding_specs[i].set_sharding_spec(Layout::kUnshardedDim);
  return Layout::GetLayout(sharding_specs, mesh);
}

// Construct a layout on a given mesh from the label to tensor dimension map
// and the label to mesh_dimension map.
std::pair<std::vector<ShardingSpec>, absl::flat_hash_map<std::string, int>>
GetSpecsFromLabelsAndMap(
    const absl::flat_hash_map<char, std::vector<int>>& label_to_index,
    const absl::flat_hash_map<char, ShardingSpec>& label_to_sharding_spec) {
  int layout_rank = 0;
  for (const auto& label_and_indices : label_to_index)
    layout_rank += label_and_indices.second.size();

  std::vector<ShardingSpec> sharding_specs(layout_rank);
  absl::flat_hash_map<std::string, int> dimension_use_count;
  absl::flat_hash_set<std::string> dimension_use_set;
  for (const auto& label_and_indices : label_to_index) {
    const auto& loc = label_to_sharding_spec.find(label_and_indices.first);
    if (loc != label_to_sharding_spec.end()) {
      const ShardingSpec& sharding_spec = loc->second;
      for (int index : label_and_indices.second)
        sharding_specs[index] = sharding_spec;
      dimension_use_count[sharding_spec.sharding_spec()] +=
          label_and_indices.second.size();
    } else {
      for (int index : label_and_indices.second)
        sharding_specs[index].set_sharding_spec(Layout::kUnshardedDim);
    }
  }
  return std::make_pair(sharding_specs, dimension_use_count);
}

StatusOr<llvm::DenseMap<int, Layout>> EinsumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.empty()) return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Need the mapping of input and output labels from the equation.
  auto einsum_op = mlir::cast<mlir::TF::EinsumOp>(op);
  size_t num_inputs = einsum_op.getNumOperands();
  std::string equation = einsum_op.equation().str();
  absl::flat_hash_set<char> reduced_dim_labels;
  std::vector<absl::flat_hash_map<char, std::vector<int>>> input_mappings;
  absl::flat_hash_map<char, std::vector<int>> output_mapping;

  TF_RETURN_IF_ERROR(ExtractEquationRelations(equation, reduced_dim_labels,
                                              input_mappings, output_mapping));
  if (input_mappings.size() != num_inputs)
    return errors::InvalidArgument(
        "Einsum equation ", equation, " has ", input_mappings.size(),
        " inputs but this op has ", num_inputs, " inputs.");

  // GetLabelToShardingSpec requires two inputs if the einsum equation needs
  // two inputs. We may only have one layout, so make other replicated. This
  // will have the same effect as only using the defined layout and using
  // replicated for all the missing dimensions.
  std::vector<Layout> layouts;
  for (int k = 0; k < num_inputs; ++k) {
    if (input_layouts.find(k) != input_layouts.end()) {
      layouts.emplace_back(input_layouts.lookup(k));
    } else {
      int rank = ValueRank(op->getOperand(k));
      if (rank < 0) return errors::InvalidArgument("No rank for input ", k);
      // This case can only happen when there are two inputs. Input 1 - k
      // is the other input. In this case of the if, input k is missing, so
      // this means that input 1 - k must be there.
      layouts.emplace_back(Layout::ReplicatedOnMesh(mesh, rank));
    }
  }

  // For each input, identify the mesh dimension
  TF_ASSIGN_OR_RETURN(
      auto input_label_to_sharding_spec,
      GetLabelToShardingSpec(
          /*replicate_incompatible_dimensions=*/true, layouts, input_mappings));
  // Compute output layout based on retained mesh dimensions
  TF_ASSIGN_OR_RETURN(
      const auto& output_layout,
      VerifyOrFixLayout(GetSpecsFromLabelsAndMap(output_mapping,
                                                 input_label_to_sharding_spec),
                        mesh));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> EinsumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Need the mapping of input and output labels from the equation.
  auto einsum_op = mlir::cast<mlir::TF::EinsumOp>(op);
  size_t num_inputs = einsum_op.getNumOperands();
  std::string equation = einsum_op.equation().str();
  absl::flat_hash_set<char> reduced_dim_labels;
  std::vector<absl::flat_hash_map<char, std::vector<int>>> input_mappings;
  absl::flat_hash_map<char, std::vector<int>> output_mapping;

  TF_RETURN_IF_ERROR(ExtractEquationRelations(equation, reduced_dim_labels,
                                              input_mappings, output_mapping));
  if (input_mappings.size() != num_inputs)
    return errors::InvalidArgument(
        "Einsum equation ", equation, " has ", input_mappings.size(),
        " inputs but this op has ", num_inputs, " inputs.");

  // Using the output mapping, construct an equation label to mesh dimension
  // mapping.
  TF_ASSIGN_OR_RETURN(auto output_label_to_sharding_spec,
                      GetLabelToShardingSpec(
                          /*replicate_incompatible_dimensions=*/false,
                          {output_layout}, {output_mapping}));

  // Defines a set of labels that could be set to Any. The conditions for an
  // operand label to be set to any are that 1) is not present in the output
  // and 2) is not repeated in any operand.
  absl::flat_hash_set<char> labels_for_any;
  for (const auto& operand_mapping : input_mappings)
    for (const auto& label_to_indices : operand_mapping)
      labels_for_any.insert(label_to_indices.first);

  // Filter repeated labels.
  for (const auto& operand_mapping : input_mappings)
    for (const auto& label_to_indices : operand_mapping)
      if (label_to_indices.second.size() > 1)
        labels_for_any.erase(label_to_indices.first);

  // Filter labels in output.
  for (const auto& label_to_indices : output_mapping)
    labels_for_any.erase(label_to_indices.first);

  llvm::DenseMap<int, Layout> input_layouts(num_inputs);

  // Derive operand sharding specs from output's sharding specs.
  for (size_t i = 0; i < num_inputs; ++i) {
    absl::flat_hash_map<char, std::vector<int>> labels_to_indices =
        input_mappings[i];
    std::pair<std::vector<ShardingSpec>, absl::flat_hash_map<std::string, int>>
        sharding_specs_and_dim_count = GetSpecsFromLabelsAndMap(
            labels_to_indices, output_label_to_sharding_spec);

    std::vector<ShardingSpec> sharding_specs =
        sharding_specs_and_dim_count.first;
    absl::flat_hash_map<std::string, int> dim_count =
        sharding_specs_and_dim_count.second;

    // Flip "unsharded" specs to "any" if they are present in the set.
    for (const auto& label_to_indices : labels_to_indices) {
      char label = label_to_indices.first;
      if (labels_for_any.contains(label)) {
        int index = label_to_indices.second[0];
        sharding_specs[index].set_sharding_spec(Layout::kAny);
      }
    }
    TF_ASSIGN_OR_RETURN(
        const auto& layout,
        VerifyOrFixLayout(std::make_pair(sharding_specs, dim_count), mesh));
    input_layouts[i] = layout;
  }

  return input_layouts;
}

// A few things we don't support, but could:
// * "xx->" or "xx->x": Trace like operation where at least one input dimension
//   for x is sharded. If both are sharded, we can compute the einsum on the
//   diagonal machines in the mesh and 0s on the off diagonals and then all
//   the much smaller matrix.
Status EinsumSPMDExpander::MaybeRelayoutInputs(
    const std::vector<Layout>& input_layouts, mlir::Operation* op,
    const Layout& output_layout, absl::flat_hash_set<std::string>& reduce_dims,
    Layout& einsum_layout, std::vector<mlir::Value>& new_inputs) {
  if (!mlir::isa<mlir::TF::EinsumOp>(op))
    return errors::InvalidArgument(
        "called einsum spmd expander but op is not Einsum.");

  mlir::TF::EinsumOp einsum = mlir::cast<mlir::TF::EinsumOp>(op);
  std::vector<absl::flat_hash_map<char, std::vector<int>>> input_mappings;
  absl::flat_hash_map<char, std::vector<int>> output_mapping;
  absl::flat_hash_set<char> contracting_labels;
  absl::flat_hash_set<char> all_labels;
  TF_RETURN_IF_ERROR(ExtractEquationRelations(einsum.equation().str(),
                                              contracting_labels,
                                              input_mappings, output_mapping));

  for (const auto& input_mapping : input_mappings)
    for (const auto& char_and_positions : input_mapping)
      all_labels.emplace(char_and_positions.first);

  // We will update this array throughout this function with the following rules
  // 1. The sharding of a label which is not in the map is unknown.
  // 2. Once the sharding of label becomes known and is unsharded, we
  //    won't change that.
  TF_ASSIGN_OR_RETURN(auto input_label_to_sharding_spec,
                      GetLabelToShardingSpec(
                          /*replicate_incompatible_dimensions=*/false,
                          input_layouts, input_mappings));

  TF_ASSIGN_OR_RETURN(const auto output_label_to_sharding_spec,
                      GetLabelToShardingSpec(
                          /*replicate_incompatible_dimensions=*/false,
                          {output_layout}, {output_mapping}));

  for (const char label : all_labels) {
    if (input_label_to_sharding_spec.contains(label) &&
        output_label_to_sharding_spec.contains(label) &&
        !Equal(input_label_to_sharding_spec[label],
               output_label_to_sharding_spec.find(label)->second))
      return errors::InvalidArgument(
          "for label ", label, " input and output layouts are sharded on ",
          " non-equal dimensions ",
          input_label_to_sharding_spec[label].sharding_spec(), " and ",
          output_label_to_sharding_spec.find(label)->second.sharding_spec(),
          "respectively");
  }

  // First priority is to ensure that labels which occur at least twice on one
  // side never get sharded, as we cannot deal with that. This corresponds to
  // taking a trace on that input, which will require us to be unsharded.
  for (const auto& input_mapping : input_mappings)
    for (const auto& char_and_positions : input_mapping)
      if (char_and_positions.second.size() > 1)
        input_label_to_sharding_spec[char_and_positions.first]
            .set_sharding_spec(Layout::kUnshardedDim);

  absl::flat_hash_map<std::string, absl::flat_hash_set<char>>
      sharding_dim_to_non_contracting_labels;
  absl::flat_hash_map<std::string, absl::flat_hash_set<char>>
      sharding_dim_to_contracting_labels;
  for (const auto& label_and_spec : input_label_to_sharding_spec) {
    if (Layout::IsShardedSpec(label_and_spec.second)) {
      if (contracting_labels.contains(label_and_spec.first))
        sharding_dim_to_contracting_labels[label_and_spec.second
                                               .sharding_spec()]
            .insert(label_and_spec.first);
      else
        sharding_dim_to_non_contracting_labels[label_and_spec.second
                                                   .sharding_spec()]
            .insert(label_and_spec.first);
    }
  }

  // If a non-contracting dimension is sharded in the output and non-sharded
  // in the input and no other label is sharded on that dimension, then shard
  // it.
  // This handles the *,x . x,* -> *,y case and also if batch dimensions are
  // sharded on the output but not the input.
  for (const char label : all_labels) {
    // Note that only sharded labels are in output_label_to_sharding_spec, so
    // there is no need to check that the spec is sharded.
    if (!contracting_labels.contains(label) &&
        output_label_to_sharding_spec.contains(label) &&
        !input_label_to_sharding_spec.contains(label)) {
      const ShardingSpec& sharding_spec =
          output_label_to_sharding_spec.find(label)->second;
      const std::string& string_spec = sharding_spec.sharding_spec();
      if (!sharding_dim_to_non_contracting_labels.contains(string_spec) &&
          !sharding_dim_to_contracting_labels.contains(string_spec)) {
        input_label_to_sharding_spec[label] = sharding_spec;
        sharding_dim_to_non_contracting_labels[string_spec].insert(label);
      }
    }
  }

  // Handle the case when two non-contracting dimensions are have the same
  // sharding spec.
  // Note that the case of three non-contracting dimensions having the same
  // sharding spec is impossible. Since there are at most two inputs, at least
  // one input would have two dimensions with the same sharing spec.
  // This handles the y,x . x,y -> *,y case.
  absl::flat_hash_set<std::string> dims_with_multiple_labels;
  for (const auto& spec_and_labels : sharding_dim_to_non_contracting_labels) {
    if (spec_and_labels.second.size() > 1) {
      assert(spec_and_labels.second.size() == 2);
      dims_with_multiple_labels.insert(spec_and_labels.first);
    }
  }
  for (const auto& dim : dims_with_multiple_labels) {
    // TODO(bfontain): Update this to pick default label to keep based on shape.
    char label_to_keep = 0xFF;
    // Note that all these conditions evaluated in the loop below are mutually
    // as exclusive as no two labels in the output have the same sharding
    // spec.
    // If the no label is found we choose the lexicographically least label to
    // keep this stable with respect to ordering.
    for (const char label : sharding_dim_to_non_contracting_labels[dim]) {
      if (output_label_to_sharding_spec.contains(label) &&
          output_label_to_sharding_spec.find(label)->second.sharding_spec() ==
              dim) {
        label_to_keep = label;
        break;
      } else if (label < label_to_keep) {
        label_to_keep = label;
      }
    }
    for (const char label : sharding_dim_to_non_contracting_labels[dim])
      if (label != label_to_keep)
        input_label_to_sharding_spec[label].set_sharding_spec(
            Layout::kUnshardedDim);
    sharding_dim_to_non_contracting_labels[dim].clear();
    sharding_dim_to_non_contracting_labels[dim].insert(label_to_keep);
  }

  // Handle the case where a non-contracting and contracting dim have the same
  // sharding spec. For now we always unshard the contracting axis. Note that
  // this is safe.
  // This handles the case x,y . *,y -> x,y
  for (const auto& spec_and_labels : sharding_dim_to_contracting_labels) {
    if (!spec_and_labels.second.empty() &&
        !sharding_dim_to_non_contracting_labels[spec_and_labels.first]
             .empty()) {
      assert(spec_and_labels.second.size() == 1);
      assert(sharding_dim_to_non_contracting_labels[spec_and_labels.first]
                 .size() == 1);
      input_label_to_sharding_spec[*spec_and_labels.second.begin()]
          .set_sharding_spec(Layout::kUnshardedDim);
    }
  }

  // Relayout the inputs
  mlir::OpBuilder builder(op);
  new_inputs.resize(input_mappings.size());
  for (int i = 0; i < input_mappings.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        const Layout new_input_layout,
        VerifyOrFixLayout(GetSpecsFromLabelsAndMap(
                              input_mappings[i], input_label_to_sharding_spec),
                          output_layout.mesh()));

    TF_ASSIGN_OR_RETURN(
        new_inputs[i],
        EmitRelayout(op->getOperand(i), input_layouts[i], new_input_layout));
  }

  TF_ASSIGN_OR_RETURN(
      einsum_layout,
      VerifyOrFixLayout(GetSpecsFromLabelsAndMap(output_mapping,
                                                 input_label_to_sharding_spec),
                        output_layout.mesh()));

  for (const auto& contracting : contracting_labels)
    reduce_dims.emplace(
        input_label_to_sharding_spec[contracting].sharding_spec());

  return Status::OK();
}

}  // namespace dtensor
}  // namespace tensorflow
