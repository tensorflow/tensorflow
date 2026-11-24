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

#include "xla/backends/gpu/hlo/transforms/cudnn_gemm_dimensions.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/hlo/transforms/cudnn_fusion_utils.h"
#include "xla/hlo/analysis/shape_tracker.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

using DotPort = CuDnnGemmDimensions::Port;

// Instructions that change the logical shape and are tracked by ShapeTracker.
bool IsShapeOp(const HloInstruction& hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kReshape,
                          HloOpcode::kTranspose, HloOpcode::kBroadcast>(&hlo);
}

// Instructions that keep the logical shape.
bool IsElementwiseOrCopy(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCopy || hlo.IsElementwise();
}

// Instructions of the computation that are not part of the cuDNN graph.
bool IsOutsideCuDnnGraph(const HloInstruction& hlo) {
  return hlo.IsCustomCall(kWorkspaceAllocationCustomCallTarget) ||
         (hlo.opcode() == HloOpcode::kTuple && hlo.IsRoot());
}

// Returns the single output of the computation, ignoring workspace allocation.
absl::StatusOr<const HloInstruction*> GetOutput(
    const HloComputation& computation) {
  const HloInstruction* root = computation.root_instruction();
  if (root->opcode() != HloOpcode::kTuple) {
    return root;
  }
  std::vector<const HloInstruction*> outputs;
  for (const HloInstruction* operand : root->operands()) {
    if (!operand->IsCustomCall(kWorkspaceAllocationCustomCallTarget)) {
      outputs.push_back(operand);
    }
  }
  if (outputs.size() != 1) {
    return absl::UnimplementedError(
        "Only GEMM fusions with a single output are supported.");
  }
  return outputs.front();
}

// For each of the three canonical cuDNN dimensions, the dimensions of a port of
// the dot that it consists of.
using CanonicalDims = std::array<std::vector<int64_t>, 3>;
using PortToCanonicalDims = absl::flat_hash_map<DotPort, CanonicalDims>;

absl::StatusOr<PortToCanonicalDims> GetCanonicalDimensions(
    const HloInstruction& dot) {
  using Cat = DotOperandDims::Category;
  PortToCanonicalDims result;
  for (int64_t i = 0; i < dot.operand_count(); ++i) {
    const DotPort port = static_cast<DotPort>(i);
    CanonicalDims& port_dims = result[port];
    ABSL_ASSIGN_OR_RETURN(DotOperandDims dims,
                     DotOperandDims::FromDotOperand(&dot, i));
    // cuDNN multiplies [batch, M, K] by [batch, K, N].
    const auto order =
        (port == DotPort::kLhs || port == DotPort::kLhsScale)
            ? std::array{Cat::kBatch, Cat::kNonContracting, Cat::kContracting}
            : std::array{Cat::kBatch, Cat::kContracting, Cat::kNonContracting};
    for (int position = 0; position < order.size(); ++position) {
      absl::Span<const int64_t> indices = dims.Indices(order[position]);
      port_dims[position].assign(indices.begin(), indices.end());
    }
  }
  // The dot is [batch..., M..., N...].
  int64_t next_dim = 0;
  auto range = [&next_dim](int64_t count) {
    std::vector<int64_t> dims(count);
    absl::c_iota(dims, next_dim);
    next_dim += count;
    return dims;
  };
  const CanonicalDims& lhs = result[DotPort::kLhs];
  const CanonicalDims& rhs = result[DotPort::kRhs];
  result[DotPort::kOutput] = {range(lhs[0].size()), range(lhs[1].size()),
                              range(rhs[2].size())};
  return result;
}

// Ranges of the dimensions of a port of the dot that a parameter is sliced to.
struct SliceRanges {
  std::vector<int64_t> starts;
  std::vector<int64_t> sizes;

  bool operator==(const SliceRanges& other) const = default;
};

// How the shape of an instruction gets to a port of the dot.
struct PortPath {
  DotPort port;
  // Maps the shape of the instruction to the shape of `port` before `slices`.
  ShapeTracker to_port;
  // Slices between the instruction and `port`, if any.
  std::optional<SliceRanges> slices;

  bool operator==(const PortPath& other) const = default;
};

// How each instruction relates to the dot.
using PortPaths = absl::flat_hash_map<const HloInstruction*, PortPath>;

// Returns the shape transformations from the dot to `output`.
absl::StatusOr<ShapeTracker> TrackFromDot(const HloInstruction& dot,
                                          const HloInstruction& output) {
  ShapeTracker tracker(dot.shape());
  for (const HloInstruction* hlo = &dot; hlo != &output;) {
    if (hlo->users().empty()) {
      return absl::UnimplementedError(
          absl::StrCat("Not computed from the dot: ", output.ToString()));
    }
    // Pick any path to the output - instructions with different ShapeTrackers
    // from the dot will be rejected later.
    hlo = hlo->users().front();
    if (IsShapeOp(*hlo)) {
      ABSL_RETURN_IF_ERROR(tracker.AppendInstruction(hlo));
    } else if (!IsElementwiseOrCopy(*hlo)) {
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported instruction in a cuDNN GEMM fusion: ", hlo->ToString()));
    } else if (hlo->shape().dimensions() !=
               tracker.output_shape().dimensions()) {
      // E.g. clamp with a scalar dot as one of its bounds.
      return absl::UnimplementedError(
          absl::StrCat("Implicit broadcasts of the dot are not supported: ",
                       hlo->ToString()));
    }
    tracker.SetElementType(hlo->shape().element_type());
  }
  return tracker;
}

// The `starts` and `sizes` of a slice of a shape with dimensions `unsliced`.
struct SlicedShape {
  std::vector<int64_t> unsliced;
  std::vector<int64_t> starts;
  std::vector<int64_t> sizes;
};

// Returns slice dimensions to represent slice(reshape(..)) rather than
// reshape(slice(..)). Fails if the slice cannot be represented by a single
// start/size pair per dimension.
absl::StatusOr<SlicedShape> ReshapeSlicedDimensions(
    const SlicedShape& slice, absl::Span<const int64_t> target_dims) {
  SlicedShape result = {
      .unsliced = SpanToVector(target_dims),
      .starts = std::vector<int64_t>(target_dims.size(), 0),
      .sizes = SpanToVector(target_dims),
  };
  // Groups of source and target dimensions with the same number of elements.
  const absl::InlinedVector<std::pair<int64_t, int64_t>, 8> groups =
      CommonFactors(slice.sizes, target_dims);
  int64_t src_begin = 0;
  for (int64_t group = 0; group < groups.size() - 1; ++group) {
    const int64_t dst_begin = groups[group].second;
    const auto [src_end, dst_end] = groups[group + 1];
    // Removed size-1 dimensions join the next group, e.g. [1,K,N] -> [K,N].
    if (dst_begin == dst_end && group < groups.size() - 2) {
      continue;
    }
    absl::Span<const int64_t> src_sizes =
        absl::MakeConstSpan(slice.sizes)
            .subspan(src_begin, src_end - src_begin);
    absl::Span<const int64_t> dst_sizes =
        target_dims.subspan(dst_begin, dst_end - dst_begin);
    for (int64_t i = 0; i < src_sizes.size(); ++i) {
      const int64_t src = src_begin + i;
      if (slice.unsliced[src] == slice.sizes[src]) {
        continue;
      }
      // Only the major-most dimension of a group can be sliced.
      if (Product(src_sizes.first(i)) != 1 || dst_sizes.empty()) {
        return absl::UnimplementedError(
            "The slice is not a range of a dimension after a reshape.");
      }
      // The major-most target dimension of the group that is not of size 1.
      const auto it =
          absl::c_find_if(dst_sizes, [](int64_t size) { return size != 1; });
      const int64_t major_dst_offset =
          it == dst_sizes.end() ? 0 : it - dst_sizes.begin();
      const int64_t dst = dst_begin + major_dst_offset;
      const int64_t minor_src = Product(src_sizes.subspan(i + 1));
      const int64_t minor_dst =
          Product(dst_sizes.subspan(major_dst_offset + 1));
      if (result.unsliced[dst] != target_dims[dst] ||
          (slice.starts[src] * minor_src) % minor_dst != 0 ||
          (slice.unsliced[src] * minor_src) % minor_dst != 0) {
        return absl::UnimplementedError(
            "The slice is not a range of a dimension after a reshape.");
      }
      result.unsliced[dst] = slice.unsliced[src] * minor_src / minor_dst;
      result.starts[dst] = slice.starts[src] * minor_src / minor_dst;
    }
    src_begin = src_end;
  }
  return result;
}

// Returns a new PortPath for `slice`. Recreates the ShapeTracker for the
// unsliced shape and transforms the slice dimensions to represent a slice of
// the dot operand: turns dot(bitcast(slice(p))) into dot(slice(bitcast(p))).
absl::StatusOr<PortPath> PropagateSlice(const HloInstruction& slice,
                                        const PortPath& path) {
  if (!absl::c_all_of(slice.slice_strides(),
                      [](int64_t stride) { return stride == 1; })) {
    return absl::UnimplementedError("Only unit-stride slices are supported.");
  }
  if (path.port == DotPort::kOutput) {
    // cuDNN builds plans for these graphs but fails to execute them.
    return absl::UnimplementedError("Slices are not supported after the dot.");
  }
  const Shape& operand_shape = slice.operand(0)->shape();
  SlicedShape sliced = {
      .unsliced = SpanToVector(operand_shape.dimensions()),
      .starts = slice.slice_starts(),
      .sizes = SpanToVector(slice.shape().dimensions()),
  };
  ShapeTracker unsliced_to_port(operand_shape);
  for (const ShapeTracker::Step& step : path.to_port.GetSteps()) {
    if (const auto* transpose = std::get_if<ShapeTracker::Transpose>(&step)) {
      sliced = {
          .unsliced = Permute(sliced.unsliced, transpose->permutation),
          .starts = Permute(sliced.starts, transpose->permutation),
          .sizes = Permute(sliced.sizes, transpose->permutation),
      };
      ABSL_RETURN_IF_ERROR(unsliced_to_port.AppendTranspose(transpose->permutation));
    } else if (const auto* reshape =
                   std::get_if<ShapeTracker::Reshape>(&step)) {
      ABSL_ASSIGN_OR_RETURN(sliced,
                       ReshapeSlicedDimensions(sliced, reshape->dimensions));
      ABSL_RETURN_IF_ERROR(unsliced_to_port.AppendReshape(sliced.unsliced));
    } else if (const auto* broadcast =
                   std::get_if<ShapeTracker::Broadcast>(&step)) {
      // Broadcasted dimensions are not sliced.
      SlicedShape broadcasted = {
          .unsliced = broadcast->dimensions,
          .starts = std::vector<int64_t>(broadcast->dimensions.size(), 0),
          .sizes = broadcast->dimensions,
      };
      for (int64_t i = 0; i < broadcast->broadcast_dimensions.size(); ++i) {
        broadcasted.unsliced[broadcast->broadcast_dimensions[i]] =
            sliced.unsliced[i];
        broadcasted.starts[broadcast->broadcast_dimensions[i]] =
            sliced.starts[i];
      }
      sliced = std::move(broadcasted);
      ABSL_RETURN_IF_ERROR(unsliced_to_port.AppendBroadcast(
          sliced.unsliced, broadcast->broadcast_dimensions));
    } else {
      return absl::UnimplementedError("Cannot slice a reduced tensor.");
    }
  }
  PortPath result = {
      .port = path.port,
      .to_port = std::move(unsliced_to_port),
      .slices = SliceRanges{.starts = std::move(sliced.starts),
                            .sizes = std::move(sliced.sizes)},
  };
  if (path.slices.has_value()) {
    // Slicing an already sliced port adds offsets.
    for (int64_t dim = 0; dim < result.slices->starts.size(); ++dim) {
      result.slices->starts[dim] += path.slices->starts[dim];
    }
    result.slices->sizes = path.slices->sizes;
  }
  return result;
}

// Records `path` for `hlo` and walks towards the parameters, stopping at
// the dot. Instructions that cannot be walked through have to be parameters or
// constants.
absl::Status TraceToPort(const HloInstruction& hlo, PortPath path,
                         const HloInstruction& dot, PortPaths& paths) {
  path.to_port.SetElementType(hlo.shape().element_type());
  auto [it, inserted] = paths.try_emplace(&hlo, path);
  if (!inserted) {
    // cuDNN has one tensor per instruction, so non-scalars must relate to the
    // dot in a single way. This rejects e.g. `d + transpose(d)`.
    if (it->second == path || ShapeUtil::ElementsIn(hlo.shape()) == 1) {
      return absl::OkStatus();
    }
    return absl::UnimplementedError(absl::StrCat(
        "Instruction relates to the dot in several ways: ", hlo.ToString()));
  }
  if (&hlo == &dot) {
    return absl::OkStatus();
  }
  if (IsShapeOp(hlo)) {
    ABSL_RETURN_IF_ERROR(path.to_port.PrependInstruction(&hlo));
    return TraceToPort(*hlo.operand(0), std::move(path), dot, paths);
  }
  if (hlo.opcode() == HloOpcode::kSlice) {
    ABSL_ASSIGN_OR_RETURN(PortPath operand_path, PropagateSlice(hlo, path),
                     _ << "Unsupported slice: " << hlo.ToString());
    return TraceToPort(*hlo.operand(0), std::move(operand_path), dot, paths);
  }
  if (IsElementwiseOrCopy(hlo)) {
    for (const HloInstruction* operand : hlo.operands()) {
      PortPath operand_path = path;
      // Scalar operands of clamp are implicitly broadcasted.
      if (ShapeUtil::IsScalar(operand->shape()) &&
          !ShapeUtil::IsScalar(hlo.shape())) {
        ABSL_RETURN_IF_ERROR(operand_path.to_port.PrependBroadcast({}));
      }
      ABSL_RETURN_IF_ERROR(
          TraceToPort(*operand, std::move(operand_path), dot, paths));
    }
    return absl::OkStatus();
  }
  if (!HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant>(&hlo)) {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported instruction in a cuDNN GEMM fusion: ", hlo.ToString()));
  }
  return absl::OkStatus();
}

absl::StatusOr<PortPaths> TraceDotPortsToFusionEdges(
    const HloInstruction& dot, const HloInstruction& output) {
  ABSL_ASSIGN_OR_RETURN(ShapeTracker dot_to_output, TrackFromDot(dot, output));
  ABSL_ASSIGN_OR_RETURN(ShapeTracker output_to_dot, dot_to_output.GetInverted());

  PortPaths paths;
  // The output and the instructions computed between the dot and the output.
  ABSL_RETURN_IF_ERROR(TraceToPort(
      output, {DotPort::kOutput, std::move(output_to_dot)}, dot, paths));
  // The instructions computing the operands of the dot.
  for (int64_t i = 0; i < dot.operand_count(); ++i) {
    const HloInstruction& operand = *dot.operand(i);
    ABSL_RETURN_IF_ERROR(TraceToPort(
        operand, {static_cast<DotPort>(i), ShapeTracker(operand.shape())}, dot,
        paths));
  }
  return paths;
}

// Ensure all instructions connect to the dot.
absl::Status ValidateComputation(const HloComputation& computation,
                                 const PortPaths& paths) {
  for (const HloInstruction* hlo : computation.instructions()) {
    if (!IsOutsideCuDnnGraph(*hlo) && !paths.contains(hlo)) {
      return absl::UnimplementedError(absl::StrCat(
          "Instruction does not contribute to the output: ", hlo->ToString()));
    }
  }
  return absl::OkStatus();
}

// Makes `to_port`, which starts at the logical shape of a tensor, start at the
// physical order of the tensor instead.
absl::Status PrependPhysicalLayout(const Shape& shape, ShapeTracker& to_port) {
  if (!shape.has_layout()) {
    return absl::OkStatus();
  }
  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  std::vector<int64_t> major_to_minor(minor_to_major.rbegin(),
                                      minor_to_major.rend());
  return to_port.PrependTranspose(InversePermutation(major_to_minor));
}

// Adjusts the strides of degenerate dimensions of `dims`, which do not affect
// the elements that are read, to the values that cuDNN expects.
void NormalizeStrides(bool has_batch_dimensions, CudnnTensorLayout& dims) {
  // 0 is always the batch dimension; 1 and 2 are the non-batch ones. cuDNN
  // relies on strides to determine layouts and gets confused when both
  // strides of non-batch dimensions are equal to 1 - this is the case for
  // tensors with 1-sized dimension like [A,1]. The stride of the 1-sized
  // dimension does not matter for correctness because there is no iteration
  // along this dimension, but setting it to A and representing the tensor as
  // its equivalent [1,A] helps cuDNN.
  if (dims.strides[1] == 1 && dims.strides[2] == 1) {
    const int one_sized_dim_idx = (dims.sizes[1] == 1) ? 1 : 2;
    dims.strides[one_sized_dim_idx] = dims.sizes[1] * dims.sizes[2];
  }
  // For tensors with an implicit batch dimension, set the batch stride to the
  // total packed size of the non-batch dimensions as required by cuDNN for
  // operations like block-scale dequantization.
  if (!has_batch_dimensions && dims.sizes[0] == 1) {
    dims.strides[0] = std::max(dims.sizes[1] * dims.strides[1],
                               dims.sizes[2] * dims.strides[2]);
  }
}

// Returns the sizes of the three canonical dimensions of a port with dimensions
// `port_dims`. If `virtual_batch_size` is not 1, M is split into
// [virtual_batch_size, M / virtual_batch_size] and the outer part becomes the
// batch dimension.
std::vector<int64_t> CanonicalPortSizes(absl::Span<const int64_t> port_dims,
                                        const CanonicalDims& canonical,
                                        int64_t virtual_batch_size) {
  std::vector<int64_t> sizes(canonical.size(), 1);
  for (int position = 0; position < canonical.size(); ++position) {
    for (int64_t dim : canonical[position]) {
      sizes[position] *= port_dims[dim];
    }
  }
  if (virtual_batch_size != 1) {
    sizes[0] = virtual_batch_size;
    sizes[1] /= virtual_batch_size;
  }
  return sizes;
}

// Returns how each of the three canonical dimensions of the port that `to_port`
// leads to iterates over the buffer of a tensor of shape `shape`.
absl::StatusOr<std::vector<ShapeTracker::BufferView>> CanonicalViews(
    const Shape& shape, ShapeTracker to_port, const CanonicalDims& canonical,
    int64_t virtual_batch_size) {
  ABSL_RETURN_IF_ERROR(PrependPhysicalLayout(shape, to_port));
  // Transpose and reshape the port into its canonical rank-3 shape.
  const std::vector<int64_t> sizes = CanonicalPortSizes(
      to_port.output_shape().dimensions(), canonical, virtual_batch_size);
  std::vector<int64_t> permutation;
  for (const std::vector<int64_t>& dims : canonical) {
    permutation.insert(permutation.end(), dims.begin(), dims.end());
  }
  ABSL_RETURN_IF_ERROR(to_port.AppendTranspose(permutation));
  ABSL_RETURN_IF_ERROR(to_port.AppendReshape(sizes));
  std::optional<std::vector<ShapeTracker::BufferView>> views =
      to_port.TryGetStridedView();
  if (!views.has_value()) {
    return absl::UnimplementedError(
        absl::StrCat("Not a strided view: ", to_port.DebugString()));
  }
  return *std::move(views);
}

// Returns the size of the outer part of M of the LHS if it can be used as a
// virtual batch dimension and 1 otherwise.
absl::StatusOr<int64_t> FindVirtualBatchSize(
    const HloComputation& computation, const PortPaths& paths,
    const CanonicalDims& lhs_canonical) {
  if (!lhs_canonical[0].empty()) {
    return 1;
  }
  for (const HloInstruction* parameter : computation.parameter_instructions()) {
    const PortPath& path = paths.at(parameter);
    if (path.port != DotPort::kLhs) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(
        std::vector<ShapeTracker::BufferView> views,
        CanonicalViews(parameter->shape(), path.to_port, lhs_canonical,
                       /*virtual_batch_size=*/1));
    // Split only if M consists of exactly two strided segments, neither of
    // which is broadcasted, and the outer one is also outer in memory.
    absl::Span<const int64_t> m_strides = views[1].strides();
    if (m_strides.size() == 2 && !absl::c_linear_search(m_strides, 0) &&
        m_strides[0] > m_strides[1]) {
      return views[1].extents()[0];
    }
  }
  return 1;
}

// Computes the layout of a tensor of shape `shape` that relates to a port of
// the dot via `path`.
absl::StatusOr<CudnnTensorLayout> ComputeTensorLayout(
    const Shape& shape, const PortPath& path, const CanonicalDims& canonical,
    int64_t virtual_batch_size) {
  ABSL_ASSIGN_OR_RETURN(
      std::vector<ShapeTracker::BufferView> views,
      CanonicalViews(shape, path.to_port, canonical, virtual_batch_size));
  CudnnTensorLayout layout;
  for (const ShapeTracker::BufferView& view : views) {
    if (absl::c_all_of(view.strides(), [](int64_t s) { return s == 0; })) {
      // Broadcasted and degenerate dimensions have size 1 and are broadcasted
      // by cuDNN.
      layout.strides.push_back(layout.strides.empty() ? 1
                                                      : layout.strides.back());
      layout.sizes.push_back(1);
    } else if (view.strides().size() == 1) {
      layout.strides.push_back(view.strides()[0]);
      layout.sizes.push_back(view.extents()[0]);
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Canonical dimension ", layout.sizes.size(),
          " is not a single strided dimension: ", path.to_port.DebugString()));
    }
  }
  if (path.slices.has_value()) {
    absl::Span<const int64_t> unsliced =
        path.to_port.output_shape().dimensions();
    std::vector<std::pair<int64_t, int64_t>> ranges;
    ranges.reserve(canonical.size());
    bool is_sliced = false;
    for (int position = 0; position < canonical.size(); ++position) {
      ranges.push_back({0, layout.sizes[position]});
      for (int64_t dim : canonical[position]) {
        const int64_t start = path.slices->starts[dim];
        const int64_t size = path.slices->sizes[dim];
        // Unsliced and broadcasted dimensions do not restrict the tensor.
        if (layout.sizes[position] == 1 || size == unsliced[dim]) {
          continue;
        }
        if (unsliced[dim] != layout.sizes[position]) {
          return absl::UnimplementedError(
              "Cannot slice a dimension that is not a whole canonical "
              "dimension.");
        }
        ranges.back() = {start, start + size};
        is_sliced = true;
      }
    }
    if (is_sliced) {
      layout.slices = std::move(ranges);
    }
  }
  NormalizeStrides(/*has_batch_dimensions=*/!canonical[0].empty(), layout);
  return layout;
}

}  // namespace

absl::StatusOr<CuDnnGemmDimensions> CuDnnGemmDimensions::Create(
    const HloInstruction& dot) {
  const HloComputation& computation = *dot.parent();
  ABSL_ASSIGN_OR_RETURN(PortToCanonicalDims canonical, GetCanonicalDimensions(dot));
  ABSL_ASSIGN_OR_RETURN(const HloInstruction* output, GetOutput(computation));

  // Relate every instruction to a port of the dot.
  ABSL_ASSIGN_OR_RETURN(PortPaths paths, TraceDotPortsToFusionEdges(dot, *output));
  ABSL_RETURN_IF_ERROR(ValidateComputation(computation, paths));

  // If the LHS NC dimension has been split, we can treat the outer part as a
  // batch dimension if there isn't one already. Scaled dots are not split.
  int64_t virtual_batch_size = 1;
  if (!canonical.contains(DotPort::kLhsScale)) {
    ABSL_ASSIGN_OR_RETURN(
        virtual_batch_size,
        FindVirtualBatchSize(computation, paths, canonical.at(DotPort::kLhs)));
  }

  // Compute sizes and strides.
  CuDnnGemmDimensions result;
  for (const HloInstruction* hlo : computation.instructions()) {
    auto it = paths.find(hlo);
    if (it == paths.end()) {
      continue;  // Outside of the cuDNN graph.
    }
    const PortPath& path = it->second;
    const CanonicalDims& port_canonical = canonical.at(path.port);
    // Virtual batch size is only used for the ports that share M with the LHS.
    const int64_t port_virtual_batch_size =
        path.port == DotPort::kLhs || path.port == DotPort::kOutput
            ? virtual_batch_size
            : 1;
    result.ports_[hlo] = path.port;
    // The compiler sets the sizes of ops whose operands are broadcasts, which
    // cuDNN cannot infer because broadcasts are not part of the cuDNN graph.
    auto [sizes, inserted] = result.canonical_sizes_.try_emplace(path.port);
    if (inserted) {
      const Shape& port_shape =
          path.port == DotPort::kOutput
              ? dot.shape()
              : dot.operand(static_cast<int>(path.port))->shape();
      sizes->second = CanonicalPortSizes(
          port_shape.dimensions(), port_canonical, port_virtual_batch_size);
    }
    if (hlo == output || hlo->opcode() == HloOpcode::kParameter) {
      ABSL_ASSIGN_OR_RETURN(result.tensor_layouts_[hlo],
                       ComputeTensorLayout(hlo->shape(), path, port_canonical,
                                           port_virtual_batch_size));
    }
  }
  return result;
}

const CudnnTensorLayout* absl_nullable CuDnnGemmDimensions::TensorLayout(
    const HloInstruction& hlo) const {
  auto it = tensor_layouts_.find(&hlo);
  return it == tensor_layouts_.end() ? nullptr : &it->second;
}

const std::vector<int64_t>* absl_nullable CuDnnGemmDimensions::CanonicalSizes(
    const HloInstruction& hlo) const {
  auto it = ports_.find(&hlo);
  return it == ports_.end() ? nullptr : &canonical_sizes_.at(it->second);
}

}  // namespace xla::gpu
