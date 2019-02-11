/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/custom_ops/custom_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/stream_executor/lib/status.h"

#include "absl/strings/str_cat.h"

#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplin/Norms.hpp>
#include <poputil/TileMapping.hpp>

#include <functional>
#include <numeric>

using ::absl::StrCat;
using ::tensorflow::str_util::Join;

namespace xla {
namespace poplarplugin {
namespace {
using TensorVector = std::vector<std::pair<TensorKey, poplar::Tensor>>;

bool InstructionSharded(const HloInstruction* a) {
  if (a->has_sharding()) {
    const auto& a_sharding = a->sharding();
    if (a_sharding.HasUniqueDevice()) {
      return true;
    }
  }

  return false;
}

TensorVector GetAllTensorsInMap(const TensorMap& map,
                                const HloInstruction* inst) {
  auto lower = std::make_pair(inst->name(), 0);
  auto upper = std::make_pair(inst->name(), std::numeric_limits<int64>::max());
  TensorVector outputs;
  for (auto it = map.lower_bound(lower); it != map.upper_bound(upper); it++) {
    outputs.push_back(*it);
  }
  return outputs;
}

ArgVector GetTensorsMaybeExpand(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    poplar::program::Sequence& seq, const bool expand_constants,
    absl::optional<int64> opt_tensors_start = absl::nullopt,
    absl::optional<int64> opt_tensors_end = absl::nullopt) {
  TensorVector tensor_vector = GetAllTensorsInMap(map, inst);
  int64 tensors_start = opt_tensors_start ? *opt_tensors_start : 0;
  int64 tensors_end = opt_tensors_end ? *opt_tensors_end : tensor_vector.size();
  auto& graph = GetGraph(res, inst);
  ArgVector outputs;
  for (int64 i = tensors_start; i < tensors_end; i++) {
    const auto key = tensor_vector[i].first;
    poplar::Tensor tensor = tensor_vector[i].second;
    // Check if we need to expand the constant tensor.
    if (tensor.containsConstant() && expand_constants) {
      const auto& mapping = graph.getTileMapping(tensor);
      // We only expand the constant tensor if it's mapped to 1 tile and it is
      // not a tensor of scalar shape.
      uint64 tiles_used = 0;
      for (size_t tile_idx = 0; tile_idx < mapping.size(); tile_idx++) {
        const auto& tile = mapping[tile_idx];
        tiles_used += tile.size() > 0 ? 1 : 0;
      }
      const auto& tensor_shape = tensor.shape();
      const auto num_elements =
          std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1,
                          std::multiplies<std::size_t>());

      if (tiles_used == 1 && num_elements > 1) {
        poplar::Tensor expanded_tensor = graph.addVariable(
            tensor.elementType(), tensor_shape, "wide_constant");
        poputil::mapTensorLinearly(graph, expanded_tensor);
        seq.add(poplar::program::Copy(tensor, expanded_tensor));
        tensor = expanded_tensor;
      }
    }
    map[key] = tensor;
    outputs.push_back(tensor);
  }
  return outputs;
}

}  // namespace

StatusOr<poplar::Type> PoplarDataType(const xla::PrimitiveType& element_type) {
  switch (element_type) {
    case PRED:
      return poplar::BOOL;
    case S8:
    case U8:
      return poplar::CHAR;
    case S16:
    case U16:
      return poplar::SHORT;
    case S32:
      return poplar::INT;
    case U32:
      return poplar::UNSIGNED_INT;
    case S64:
    case U64:
      return poplar::INT;
    case F16:
      return poplar::HALF;
    case F32:
      return poplar::FLOAT;
    default:
      return xla::FailedPrecondition("unsupported primitive type in poplar %s",
                                     PrimitiveType_Name(element_type));
  }
}

StatusOr<poplar::Type> PoplarDataType(const xla::Shape& shape) {
  return PoplarDataType(shape.element_type());
}

std::vector<size_t> PoplarShapeFromXlaShape(const xla::Shape& xla_shape) {
  std::vector<size_t> shape;
  for (auto d : xla_shape.dimensions()) {
    shape.push_back(d);
  }
  return shape;
}

xla::Shape XlaShapeFromPoplarShape(PrimitiveType element_type,
                                   const std::vector<size_t>& poplar_shape) {
  xla::Shape shape;
  shape.set_element_type(element_type);
  for (int64 dimension : poplar_shape) {
    shape.add_dimensions(dimension);
  }
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape;
}

poplar::Tensor ConvertToDeviceLayout(const Shape& shape,
                                     const poplar::Tensor& tensor) {
  // Reshape then dimshuffle
  poplar::Tensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<std::size_t> dim(rank);
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i = 0; i < rank; i++) {
      shuffle[shape.layout().minor_to_major(i)] = rank - i - 1;
      dim[rank - i - 1] = tensor.dim(shape.layout().minor_to_major(i));
    }

    out = out.reshape(dim);
    out = out.dimShuffle(shuffle);
  }
  return out;
}

poplar::Tensor ConvertFromDeviceLayout(const Shape& shape,
                                       const poplar::Tensor& tensor) {
  // Dimshuffle then reshape
  poplar::Tensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i = 0; i < rank; i++) {
      shuffle[rank - i - 1] = shape.layout().minor_to_major(i);
    }
    out = out.dimShuffle(shuffle);
    out = out.reshape(tensor.shape());
  }
  return out;
}

StatusOr<poplar::Tensor> AddPlainTensor(poplar::Graph& graph,
                                        const std::string& debug_name,
                                        const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addVariable(poplar_type, dim, debug_name);
  poputil::mapTensorLinearly(graph, out);
  return out;
}

StatusOr<poplar::Tensor> AddRnnSequence(poplar::Graph& graph,
                                        const std::string& debug_name,
                                        const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addVariable(poplar_type, dim, debug_name);

  for (auto i = 0; i != dim[0]; ++i) {
    poputil::mapTensorLinearly(graph, out[i]);
  }

  return out;
}

template <typename IIter1, typename IIter2, typename OIter, typename Zipper>
static void zip(IIter1 ibegin1, IIter1 iend1, IIter2 ibegin2, OIter obegin,
                Zipper zipper) {
  for (; ibegin1 != iend1; ++ibegin1, ++ibegin2, ++obegin) {
    *obegin = zipper(*ibegin1, *ibegin2);
  }
}

// Find a value for G s.t. D / G <= T, and G | D.
static StatusOr<std::size_t> FindG(const std::size_t D, const std::size_t T) {
  for (std::size_t g = (D + T - 1) / T; g <= D; ++g) {
    if (D % g == 0) {
      return g;
    }
  }

  return tensorflow::errors::FailedPrecondition(
      "Cannot find a value of G that is both a factor of D and satisfies D / G "
      "<= T");
}

// Find the sequence dimension, if there is one
static StatusOr<std::size_t> FindSeqDim(const xla::Shape& shape_xla,
                                        const xla::Shape& slice_shape_xla) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);
  const auto slice_shape = PoplarShapeFromXlaShape(slice_shape_xla);
  const auto volume =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto slice_volume =
      std::accumulate(slice_shape.begin(), slice_shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // If the desired shape is 1D, then no special work is required.
  // If the slice shape is the same as the input shape, this is just a copy
  if (shape_xla.rank() > 1 && shape != slice_shape && volume > 1 &&
      slice_volume > 1) {
    // Calculate the element-wise ratio between the slice the input rank
    std::vector<float> dimension_ratios(shape.size());
    zip(slice_shape.begin(), slice_shape.end(), shape.begin(),
        dimension_ratios.begin(), std::divides<float>());

    // Assumes the sequence dimension is the dimension with the smallest ratio
    // between the input and the slice.
    return std::distance(
        dimension_ratios.begin(),
        std::min_element(dimension_ratios.begin(), dimension_ratios.end()));
  }

  return tensorflow::errors::FailedPrecondition(
      "Cannot compute slice sequence dimension");
}

StatusOr<poplar::Tensor> AddDynamicSliceTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla) {
  poplar::Tensor unused;
  return AddDynamicSliceTensor(graph, debug_name, shape_xla, slice_shape_xla,
                               unused);
}

StatusOr<poplar::Tensor> AddDynamicSliceTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla,
    poplar::Tensor& physical_layout) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);
  const auto volume =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // If we are able to compute the sequence_dimension
  const auto sequence_dimension_status = FindSeqDim(shape_xla, slice_shape_xla);
  if (!sequence_dimension_status.ok()) {
    TF_ASSIGN_OR_RETURN(physical_layout,
                        AddPlainTensor(graph, debug_name, shape_xla));
    return physical_layout;
  }

  const auto sequence_dimension = sequence_dimension_status.ValueOrDie();

  // Create a tensor of the form [D/G, S, G] where D is the product of the N-1
  // dimensions that are not the sequence dimension, S is the size of the
  // sequence dimension, and G is a factor of D chosen to ensure that
  // D/G <= T, where T is the number of tiles.
  const auto T = graph.getTarget().getNumTiles();
  const auto D = volume / shape[sequence_dimension];
  const auto S = shape[sequence_dimension];
  const auto G_status = FindG(D, T);
  if (!G_status.ok()) {
    TF_ASSIGN_OR_RETURN(physical_layout,
                        AddPlainTensor(graph, debug_name, shape_xla));
    return physical_layout;
  }

  const auto G = G_status.ValueOrDie();
  if (D == G) {
    TF_ASSIGN_OR_RETURN(physical_layout,
                        AddPlainTensor(graph, debug_name, shape_xla));
    return physical_layout;
  }

  // If a value for G was found
  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(shape_xla));

  poplar::Tensor out =
      graph.addVariable(poplar_type, {D / G, S, G}, debug_name);
  physical_layout = out;

  // Map the sequence dimension across the tiles
  for (std::size_t i = 0; i < out.dim(0); ++i) {
    graph.setTileMapping(out[i], i);
  }

  // Reshape, with the sequence dimension being the last dimension
  auto shape_tmp = shape;
  std::swap(shape_tmp[sequence_dimension], shape_tmp.back());
  out = out.reshape(shape_tmp);

  // Shuffle the dimensions back into the desired order
  // out.dimSwap(sequence_dimension, shape.size() - 1)
  std::vector<unsigned> permutation(shape.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[sequence_dimension], permutation.back());
  out = out.dimShuffle(permutation);

  return out;
}

static StatusOr<poplar::Tensor> AddConvolutionInput(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* target, CompilerResources& resources) {
  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(target, 0, 1));

  auto name = StrCat(debug_name, "_input");
  poplar::OptionFlags opts;
  poplar::Tensor out = poplin::createInput(graph, params, name, opts,
                                           &resources.convolution_cache);
  return ShuffleConvolutionInputToTensorflow(target, out);
}

static StatusOr<poplar::Tensor> AddConvolutionWeights(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* target, CompilerResources& resources) {
  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(target, 0, 1));

  auto name = StrCat(debug_name, "_weights");
  poplar::OptionFlags opts;
  poplar::Tensor out = poplin::createWeights(graph, params, name, opts,
                                             &resources.convolution_cache);

  out = RemoveGroupsDimensionFromWeights(params, out, false);

  return ShuffleConvolutionWeightsToTensorflow(target, out);
}

static StatusOr<poplar::Tensor> AddConvAddBiasTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* layout, int64 layout_output_idx,
    const TensorMap& tensor_map) {
  OutVector outputs = FindInstructionOutputs(tensor_map, layout);

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition("Convolution %s output not found for %s",
                                   layout->name(), debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];

  acts = ShuffleConvolutionOutputToPoplar(layout, acts);

  return poplin::createBiases(graph, acts, debug_name);
}

static StatusOr<poplar::Tensor> AddMatMulAddBiasTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* layout, int64 layout_output_idx,
    const TensorMap& tensor_map) {
  OutVector outputs = FindInstructionOutputs(tensor_map, layout);

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition("Matmul %s output not found for %s",
                                   layout->name(), debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];

  return poplin::createBiases(graph, acts, debug_name);
}

static StatusOr<poplar::Tensor> AddLeftMatMul(poplar::Graph& graph,
                                              const std::string& debug_name,
                                              const xla::Shape& shape,
                                              const HloInstruction* target,
                                              CompilerResources& resources) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(shape));
  const auto& aShape = PoplarShapeFromXlaShape(target->operand(0)->shape());
  const auto& bShape = PoplarShapeFromXlaShape(target->operand(1)->shape());
  auto name = StrCat(debug_name, "_lhs");
  poplar::OptionFlags opts;
  return poplin::createMatMulInputLHS(graph, type, aShape, bShape, name, opts,
                                      &resources.dot_cache);
}

static StatusOr<poplar::Tensor> AddRightMatMul(poplar::Graph& graph,
                                               const std::string& debug_name,
                                               const xla::Shape& shape,
                                               const HloInstruction* target,
                                               CompilerResources& resources) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(shape));
  const auto& aShape = PoplarShapeFromXlaShape(target->operand(0)->shape());
  const auto& bShape = PoplarShapeFromXlaShape(target->operand(1)->shape());
  auto name = StrCat(debug_name, "_rhs");
  poplar::OptionFlags opts;
  return poplin::createMatMulInputRHS(graph, type, aShape, bShape, name, opts,
                                      &resources.dot_cache);
}

StatusOr<poplar::Tensor> AddNormScaleTensor(poplar::Graph& graph,
                                            const std::string& debug_name,
                                            const HloInstruction* layout,
                                            int64 layout_output_idx,
                                            const unsigned feature_dimension,
                                            const TensorMap& tensor_map) {
  OutVector outputs = FindInstructionOutputs(tensor_map, layout);

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Batch Norm %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];
  auto pair = ShuffleNormInputToPoplar(acts, feature_dimension);

  return poplin::createNormGamma(graph, pair.first);
}

StatusOr<poplar::Tensor> AddNormOffsetTensor(poplar::Graph& graph,
                                             const std::string& debug_name,
                                             const HloInstruction* layout,
                                             int64 layout_output_idx,
                                             const unsigned feature_dimension,
                                             const TensorMap& tensor_map) {
  OutVector outputs = FindInstructionOutputs(tensor_map, layout);

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Batch Norm %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor acts = outputs[layout_output_idx];
  auto pair = ShuffleNormInputToPoplar(acts, feature_dimension);

  return poplin::createNormBeta(graph, pair.first);
}

static StatusOr<poplar::Tensor> AddElementwiseBinary(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* layout, int64 layout_output_idx,
    const TensorMap& tensor_map) {
  OutVector outputs = FindInstructionOutputs(tensor_map, layout);

  if (layout_output_idx < 0 || outputs.size() <= layout_output_idx) {
    return xla::FailedPrecondition(
        "Elementwise %s layout input not found for %s", layout->name(),
        debug_name);
  }

  poplar::Tensor other_side = outputs[layout_output_idx];
  return graph.clone(other_side, debug_name);
}

static StatusOr<poplar::Tensor> PathTransform(
    poplar::Graph& graph, poplar::Tensor in,
    const std::vector<const HloInstruction*>& path) {
  // Now apply any transformations required by the path from the source to
  // the target

  for (auto i = path.rbegin(); i != path.rend(); ++i) {
    auto& inst = *i;
    switch (inst->opcode()) {
      case HloOpcode::kTranspose: {
        auto optional_permutation =
            convert_array<std::vector<unsigned>>(inst->dimensions());
        if (!optional_permutation) {
          return xla::FailedPrecondition(
              "PathTransform - cannot cast permutation.");
        }
        std::vector<unsigned> permutation = *optional_permutation;
        std::vector<unsigned> shuffle(permutation.size());
        for (int d = 0; d < permutation.size(); d++) {
          shuffle[permutation[d]] = d;
        }
        in = in.dimShuffle(shuffle);
        break;
      }
      case HloOpcode::kReshape: {
        std::vector<size_t> dims(
            PoplarShapeFromXlaShape(inst->operand(0)->shape()));
        in = in.reshape(dims);
        break;
      }
      default: {
        // All other instructions in the path do not modify the shape
        break;
      }
    }
  }

  return in;
}

StatusOr<poplar::Tensor> AddTensor(poplar::Graph& graph,
                                   const TensorSource& src,
                                   const xla::Shape& shape,
                                   CompilerResources& resources,
                                   const TensorMap& tensor_map) {
  const auto& name = GetDebugName(src.first);
  poplar::Tensor out;

  auto target = resources.annotations.tensor_allocation_map.find(src);
  if (target != resources.annotations.tensor_allocation_map.end()) {
    const auto* tgt = target->second.tgt;
    auto tshape = tgt->operand(target->second.input_index)->shape();
    const auto optional_layout = target->second.layout;
    const auto optional_layout_output_idx = target->second.layout_output_idx;

    if (IsPopOpsElementwiseBinary(tgt) && !IsPopOpsBiasAdd(tgt)) {
      TF_ASSIGN_OR_RETURN(
          out, AddElementwiseBinary(graph, name, *optional_layout,
                                    *optional_layout_output_idx, tensor_map));
    } else {
      switch (tgt->opcode()) {
        case HloOpcode::kBatchNormInference:
        case HloOpcode::kBatchNormTraining: {
          const unsigned feature_dimension =
              Cast<HloBatchNormInstruction>(tgt)->feature_index();
          switch (target->second.input_index) {
            case 1: {
              TF_ASSIGN_OR_RETURN(
                  out, AddNormScaleTensor(graph, name, *optional_layout,
                                          *optional_layout_output_idx,
                                          feature_dimension, tensor_map));
              break;
            }
            case 2: {
              TF_ASSIGN_OR_RETURN(
                  out, AddNormOffsetTensor(graph, name, *optional_layout,
                                           *optional_layout_output_idx,
                                           feature_dimension, tensor_map));
              break;
            }
            default:
              return xla::FailedPrecondition(
                  "invalid operand for tensor allocation on %s",
                  src.first->name().c_str());
          }
          break;
        }
        case HloOpcode::kConvolution: {
          switch (target->second.input_index) {
            case 0: {
              TF_ASSIGN_OR_RETURN(
                  out, AddConvolutionInput(graph, name, tgt, resources));
              break;
            }
            case 1: {
              TF_ASSIGN_OR_RETURN(
                  out, AddConvolutionWeights(graph, name, tgt, resources));
              break;
            }
            default:
              return xla::FailedPrecondition(
                  "invalid operand for tensor allocation on %s",
                  src.first->name().c_str());
          }
          break;
        }
        case HloOpcode::kDot: {
          switch (target->second.input_index) {
            case 0: {
              TF_ASSIGN_OR_RETURN(
                  out, AddLeftMatMul(graph, name, tshape, tgt, resources));
              break;
            }
            case 1: {
              TF_ASSIGN_OR_RETURN(
                  out, AddRightMatMul(graph, name, tshape, tgt, resources));
              break;
            }
            default:
              return xla::FailedPrecondition(
                  "invalid operand for tensor allocation on %s",
                  src.first->name().c_str());
          }
          break;
        }
        case HloOpcode::kDynamicSlice: {
          if (target->second.input_index == 0) {
            TF_ASSIGN_OR_RETURN(
                out, AddDynamicSliceTensor(graph, name, tshape,
                                           target->second.tgt->shape()));
          } else {
            TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, tshape));
          }
          break;
        }
        case HloOpcode::kDynamicUpdateSlice: {
          if (target->second.input_index == 0) {
            TF_ASSIGN_OR_RETURN(
                out, AddDynamicSliceTensor(graph, name, tshape,
                                           target->second.tgt->shape()));
          } else {
            TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, tshape));
          }
          break;
        }
        case HloOpcode::kFusion: {
          const HloComputation* comp = tgt->fused_instructions_computation();
          if (IsPopOpsFusion(comp)) {
            if (IsPopOpsFusion(comp, "depthwise_conv")) {
              switch (target->second.input_index) {
                case 0: {
                  TF_ASSIGN_OR_RETURN(
                      out, AddConvolutionInput(graph, name, tgt, resources));
                  break;
                }
                case 1: {
                  TF_ASSIGN_OR_RETURN(
                      out, AddConvolutionWeights(graph, name, tgt, resources));
                  break;
                }
                default:
                  return xla::FailedPrecondition(
                      "invalid operand for tensor allocation on %s",
                      src.first->name().c_str());
              }
            } else if (IsPopOpsFusion(comp, "conv_biasadd")) {
              TF_ASSIGN_OR_RETURN(
                  out, AddConvAddBiasTensor(graph, name, *optional_layout,
                                            *optional_layout_output_idx,
                                            tensor_map));
            } else if (IsPopOpsFusion(comp, "matmul_biasadd")) {
              TF_ASSIGN_OR_RETURN(
                  out, AddMatMulAddBiasTensor(graph, name, *optional_layout,
                                              *optional_layout_output_idx,
                                              tensor_map));
            } else {
              return xla::FailedPrecondition(
                  "Unknown poplibs fusion for tensor %s: %s",
                  src.first->name().c_str(), name.c_str());
            }
          } else {
            TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, tshape));
          }
          break;
        }
        case HloOpcode::kCustomCall: {
          if (IsPoplibsCustomOp(tgt)) {
            TF_ASSIGN_OR_RETURN(
                out, AllocatePoplibsOpTensor(
                         graph, resources, name, tgt,
                         target->second.input_index, optional_layout,
                         optional_layout_output_idx, shape, tensor_map));
          } else {
            LOG(FATAL) << "Unsupported custom call " << tgt->name();
          }
          break;
        }
        default:
          return xla::FailedPrecondition("Unknown tensor target for %s: %s",
                                         src.first->name().c_str(),
                                         tgt->name().c_str());
      }
    }

    TF_ASSIGN_OR_RETURN(
        out, PathTransform(graph, out, target->second.backward_path));

  } else {
    TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, shape));
  }
  return out;
}

template <typename TYPE>
static void AddConstantTensor(poplar::Graph& graph, const xla::Literal& literal,
                              const xla::Shape& shape, const poplar::Type& type,
                              poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));

  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (TYPE)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data[0]);
  } else {
    tensor = graph.addConstant(type, dim, data);
  }
  graph.setTileMapping(tensor, 0);

  tensor = ConvertToDeviceLayout(shape, tensor);
}

static void AddFp16ConstantTensor(poplar::Graph& graph,
                                  const xla::Literal& literal,
                                  const xla::Shape& shape,
                                  const poplar::Type& type,
                                  poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const uint16_t* data(static_cast<const uint16_t*>(literal.untyped_data()));

  if (num_elements == 0) {
    tensor = graph.addConstantHalf(type, {0}, (uint16_t)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstantHalf(type, dim, data[0]);
  } else {
    tensor = graph.addConstantHalf(type, dim, (uint16_t*)data);
  }
  graph.setTileMapping(tensor, 0);

  tensor = ConvertToDeviceLayout(shape, tensor);
}

static void Add64BitConstantTensor(poplar::Graph& graph,
                                   const xla::Literal& literal,
                                   const xla::Shape& shape,
                                   const poplar::Type& type,
                                   poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const void* data(static_cast<const void*>(literal.untyped_data()));

  std::vector<char> converted =
      ConvInt64ToInt32(data, num_elements * sizeof(int64), 0);

  const int32* data32 = reinterpret_cast<const int32*>(converted.data());

  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (int32)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data32[0]);
  } else {
    tensor = graph.addConstant(type, dim, data32);
  }
  graph.setTileMapping(tensor, 0);
}

template <typename TYPE>
static void SetInitialTensorValue(poplar::Graph& graph, poplar::Tensor& tensor,
                                  const xla::Literal& literal) {
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));
  size_t element_count = literal.element_count();
  poplar::ArrayRef<TYPE> array(data, element_count);
  graph.setInitialValue<TYPE>(tensor, array);
}

static void SetFp16InitialTensorValue(poplar::Graph& graph,
                                      poplar::Tensor& tensor,
                                      const xla::Literal& literal) {
  const uint16_t* data(static_cast<const uint16_t*>(literal.untyped_data()));
  size_t element_count = literal.element_count();
  poplar::ArrayRef<uint16_t> array(data, element_count);
  graph.setInitialValueHalf(tensor, array);
}

static void Set64BitInitialTensorValue(poplar::Graph& graph,
                                       poplar::Tensor& tensor,
                                       const xla::Literal& literal) {
  size_t element_count = literal.element_count();
  const void* data(static_cast<const void*>(literal.untyped_data()));
  std::vector<char> converted =
      ConvInt64ToInt32(data, element_count * sizeof(int64), 0);

  int32* data32 = reinterpret_cast<int32*>(converted.data());
  poplar::ArrayRef<int32> array(data32, element_count);
  graph.setInitialValue<int>(tensor, array);
}

StatusOr<poplar::Tensor> AddConstantTensor(poplar::Graph& graph,
                                           const TensorSource& src,
                                           const xla::Shape& shape,
                                           const xla::Literal& literal,
                                           CompilerResources& resources,
                                           const TensorMap& tensor_map) {
  poplar::Tensor tensor;

  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(literal.shape()));

  if (ShapeUtil::ElementsIn(literal.shape()) > 32) {
    TF_ASSIGN_OR_RETURN(tensor,
                        AddTensor(graph, src, shape, resources, tensor_map));
    switch (literal.shape().element_type()) {
      case PRED:
        SetInitialTensorValue<bool>(graph, tensor, literal);
        break;
      case S32:
        SetInitialTensorValue<int>(graph, tensor, literal);
        break;
      case U32:
        SetInitialTensorValue<unsigned>(graph, tensor, literal);
        break;
      case U64:
      case S64:
        Set64BitInitialTensorValue(graph, tensor, literal);
        break;
      case F16:
        SetFp16InitialTensorValue(graph, tensor, literal);
        break;
      case F32:
        SetInitialTensorValue<float>(graph, tensor, literal);
        break;
      default:
        // The unsupported cases were caught in the call to PoplarDataType above
        break;
    }
    return ConvertToDeviceLayout(shape, tensor);
  } else {
    switch (literal.shape().element_type()) {
      case PRED:
        AddConstantTensor<bool>(graph, literal, shape, type, tensor);
        break;
      case S32:
        AddConstantTensor<int>(graph, literal, shape, type, tensor);
        break;
      case U32:
        AddConstantTensor<unsigned>(graph, literal, shape, type, tensor);
        break;
      case U64:
      case S64:
        Add64BitConstantTensor(graph, literal, shape, type, tensor);
        break;
      case F16:
        AddFp16ConstantTensor(graph, literal, shape, type, tensor);
        break;
      case F32:
        AddConstantTensor<float>(graph, literal, shape, type, tensor);
        break;
      default:
        // The unsupported cases were caught in the call to PoplarDataType above
        break;
    }

    std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
    return tensor.reshape(dim);
  }
}

template <typename TYPE>
static Literal GetIotaLiteral(int64 len) {
  std::vector<TYPE> data(len);
  std::iota(data.begin(), data.end(), 0);
  return LiteralUtil::CreateR1<TYPE>(data);
}

StatusOr<poplar::Tensor> AddIotaTensor(poplar::Graph& graph,
                                       const TensorSource& src,
                                       const xla::Shape& shape,
                                       int64 iota_dimension,
                                       CompilerResources& resources,
                                       const TensorMap& tensor_map) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(shape));

  int64 len = shape.dimensions(iota_dimension);
  Literal literal;

  switch (shape.element_type()) {
    case S32: {
      literal = GetIotaLiteral<int>(len);
      break;
    }
    case U32: {
      literal = GetIotaLiteral<unsigned>(len);
      break;
    }
    case F32: {
      literal = GetIotaLiteral<float>(len);
      break;
    }
    default:
      return xla::FailedPrecondition("unsupported primitive type for iota: %s",
                                     PrimitiveType_Name(shape.element_type()));
  }
  poplar::Tensor t;
  auto iota_shape = ShapeUtil::MakeShape(shape.element_type(),
                                         {shape.dimensions(iota_dimension)});
  TF_ASSIGN_OR_RETURN(t, AddConstantTensor(graph, src, iota_shape, literal,
                                           resources, tensor_map));
  return BroadcastTensor(t, shape, {iota_dimension});
}

template <typename T>
poplar::Tensor TileTensor(const T& multiples, const poplar::Tensor& in) {
  poplar::Tensor out = in;
  for (unsigned d = 0; d < multiples.size(); d++) {
    int m = multiples[d];
    out = out.broadcast(m, d);
  }
  return out;
}

template poplar::Tensor TileTensor<tensorflow::BCast::Vec>(
    const tensorflow::BCast::Vec&, const poplar::Tensor&);

template poplar::Tensor TileTensor<std::vector<std::size_t>>(
    const std::vector<std::size_t>&, const poplar::Tensor&);

StatusOr<poplar::Tensor> PadTensor(const PaddingConfig& cfg,
                                   const poplar::Tensor& in,
                                   const poplar::Tensor& pad) {
  if (pad.numElements() != 1) {
    return xla::FailedPrecondition(
        "PadTensor: pad tensor is not single valued");
  }

  poplar::Tensor p(pad.reshape(std::vector<std::size_t>(in.rank(), 1)));

  poplar::Tensor out = in;
  for (unsigned d = 0; d < in.rank(); d++) {
    std::vector<std::size_t> shape(out.shape());

    if (cfg.dimensions(d).interior_padding() > 0 && shape[d] > 0) {
      shape[d] = cfg.dimensions(d).interior_padding();
      poplar::Tensor padded = TileTensor(shape, p);
      poplar::Tensor interleaved = out.slice(0, 1, d);
      for (unsigned int slice = 1; slice < out.dim(d); slice++) {
        interleaved = poplar::concat(interleaved, padded, d);
        interleaved =
            poplar::concat(interleaved, out.slice(slice, slice + 1, d), d);
      }
      out = interleaved;
    }

    if (cfg.dimensions(d).edge_padding_low() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_low();
      poplar::Tensor padded = TileTensor(shape, p);
      out = poplar::concat(padded, out, d);
    }

    if (cfg.dimensions(d).edge_padding_high() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_high();
      poplar::Tensor padded = TileTensor(shape, p);
      out = poplar::concat(out, padded, d);
    }
  }

  return out;
}

StatusOr<poplar::Tensor> ReverseTensor(const poplar::Tensor& in,
                                       const std::vector<int64>& dimensions) {
  poplar::Tensor out = in;
  if (in.numElements() > 0) {
    for (int64 d : dimensions) {
      out = out.reverse(d);
    }
  }
  return out;
}

StatusOr<poplar::Tensor> BroadcastTensor(const poplar::Tensor& in,
                                         const xla::Shape& out,
                                         const std::vector<int64>& dimensions) {
  if (PoplarShapeMatchesXLAShape(in, out)) {
    return in;
  }

  auto optional_bcast_shape =
      convert_array<tensorflow::BCast::Vec>(out.dimensions());
  if (!optional_bcast_shape) {
    return xla::FailedPrecondition(
        "BroadcastTensor - cannot cast output shape.");
  }
  tensorflow::BCast::Vec bcast_shape = *optional_bcast_shape;

  tensorflow::BCast::Vec tensor_shape(out.rank(), 1);
  if (dimensions.size() > 0) {
    for (size_t d = 0; d < dimensions.size(); d++) {
      tensor_shape[dimensions[d]] = in.dim(d);
    }
  } else {
    for (size_t d = 0; d < in.rank(); d++) {
      tensor_shape[d] = in.dim(d);
    }
  }

  tensorflow::BCast bcast(tensor_shape, bcast_shape);
  if (!bcast.IsValid()) {
    return xla::FailedPrecondition("Incompatible broadcast from (%s) to (%s)",
                                   Join(tensor_shape, ",").c_str(),
                                   Join(bcast_shape, ",").c_str());
  }

  poplar::Tensor o = in;
  auto optional_bcast_x_shape =
      convert_array<std::vector<size_t>>(bcast.x_reshape());
  if (!optional_bcast_x_shape) {
    return xla::FailedPrecondition(
        "BroadcastTensor - cannot cast broadcast shape.");
  }
  std::vector<size_t> bcast_x_shape = *optional_bcast_x_shape;
  o = in.reshape(bcast_x_shape);
  o = TileTensor(bcast.x_bcast(), o);
  return o.reshape(PoplarShapeFromXlaShape(out));
}

bool PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                                const xla::Shape& shape) {
  if (tensor.rank() != shape.rank()) return false;
  for (size_t d = 0; d < tensor.rank(); d++) {
    if (tensor.dim(d) != (unsigned)shape.dimensions(d)) return false;
  }

  return true;
}

std::pair<int64, int64> FindTupleInputIndices(const HloInstruction* tuple,
                                              int64 n) {
  int64 start = 0;
  for (int64 i = 0; i < n; i++) {
    start += CountShapes(tuple->operand(i)->shape());
  }
  int64 end = start + CountShapes(tuple->operand(n)->shape());
  return std::make_pair(start, end);
}

ArgVector FindTupleInInstructionInput(TensorMap& map, CompilerResources& res,
                                      const HloInstruction* inst, int64 input,
                                      int64 n, poplar::program::Sequence& seq,
                                      const bool expand_constants) {
  const HloInstruction* operand = inst->operand(input);
  const Shape& shape = operand->shape();
  int64 start = 0;
  for (int64 i = 0; i < n; i++) {
    start += CountShapes(ShapeUtil::GetTupleElementShape(shape, i));
  }
  int64 end = start + CountShapes(ShapeUtil::GetTupleElementShape(shape, n));
  ArgVector inputs = GetTensorsMaybeExpand(map, res, operand, seq,
                                           expand_constants, start, end);
  return inputs;
}

StatusOr<poplar::Tensor> FindInstructionInput(
    TensorMap& map, CompilerResources& res, const HloInstruction* inst,
    int64 input, poplar::program::Sequence& seq, const bool expand_constants) {
  const HloInstruction* operand = inst->operand(input);
  ArgVector inputs =
      GetTensorsMaybeExpand(map, res, operand, seq, expand_constants, 0, 1);

  if (inputs.size() == 0) {
    return tensorflow::errors::Unknown(
        StrCat("[Poplar] Couldn't find input ", input, " for ", inst->name()));
  }

  return inputs[0];
}

ArgVector FindInstructionInputs(TensorMap& map, CompilerResources& res,
                                const HloInstruction* inst, int64 input,
                                poplar::program::Sequence& seq,
                                const bool expand_constants) {
  const HloInstruction* operand = inst->operand(input);
  return GetTensorsMaybeExpand(map, res, operand, seq, expand_constants);
}

OutVector FindInstructionOutputs(const TensorMap& map,
                                 const HloInstruction* inst) {
  TensorVector tensor_vector = GetAllTensorsInMap(map, inst);
  OutVector outputs;
  std::transform(tensor_vector.begin(), tensor_vector.end(),
                 std::back_inserter(outputs),
                 [](const std::pair<TensorKey, poplar::Tensor>& value) {
                   return std::get<1>(value);
                 });
  return outputs;
}

OutVector FindExpandedInstructionOutputs(TensorMap& map, CompilerResources& res,
                                         const HloInstruction* inst,
                                         poplar::program::Sequence& seq) {
  OutVector outputs = GetTensorsMaybeExpand(map, res, inst, seq, true);
  return outputs;
}

StatusOr<ArgVectors> GetInplaceOutputTensors(TensorMap& map,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             poplar::program::Sequence& seq,
                                             const bool expand_constants) {
  // Check that the instruction description is for an inplace operation.
  auto inst_description = InplaceUtil::GetHloInstructionDescription(inst);
  if (!inst_description->IsInPlaceType(inst)) {
    LOG(FATAL) << "Trying to execute " << inst->name()
               << " as an inplace operation, but it is not.";
  }
  auto& inplace_description =
      *static_cast<InplaceUtil::InplaceHloInstructionDescription*>(
          inst_description.get());

  const bool is_still_inplace =
      res.annotations.inplace_instructions.count(inst);

  // Get all the input tensors for all the inplace operands
  auto inplace_indexes = inplace_description.GetInplaceOperandIndexes();
  ArgVectors tensors(inplace_indexes.size());
  // Specialise for GTEs
  if (inst->opcode() == HloOpcode::kGetTupleElement) {
    // This should *always* be true
    CHECK_EQ(inplace_indexes.size(), 1);
    CHECK_EQ(inplace_indexes[0], 0);
    tensors[0] = FindTupleInInstructionInput(
        map, res, inst, 0, inst->tuple_index(), seq, expand_constants);
  } else {
    for (uint64 i = 0; i < inplace_indexes.size(); i++) {
      tensors[i] = FindInstructionInputs(map, res, inst, inplace_indexes[i],
                                         seq, expand_constants);
    }
  }

  // Go through all the inplace tensors and check if we need to add copies.
  auto& graph = GetGraph(res, inst);
  for (uint64 i = 0; i < inplace_indexes.size(); i++) {
    for (uint64 j = 0; j < tensors[i].size(); j++) {
      poplar::Tensor t = tensors[i][j];
      // We need to add a copy before an inplace op if:
      // 1. t is not ParallelWriteable,
      // 2. inst is not marked as inplace.
      bool requires_copy_inplace =
          !t.isParallelWriteable() || !is_still_inplace;
      if (requires_copy_inplace) {
        VLOG(1) << "Adding a copy for inplace op " << inst->name();
        poplar::Tensor copy = graph.clone(t, GetDebugName(inst) + ".clone");
        seq.add(poplar::program::Copy(t, copy));
        t = copy;
      }
      tensors[i][j] = t;
    }
  }
  return tensors;
}

Status AddOutputTensor(TensorMap& map, const HloInstruction* inst, int64 n,
                       const poplar::Tensor& tensor) {
  auto p = std::make_pair(inst->name(), n);
  auto it = map.find(p);
  if (it != map.end()) {
    return tensorflow::errors::Unknown(StrCat(
        "[Poplar] Ouptut Tensor for ", GetDebugName(inst), " already exists"));
  }
  map[p] = tensor;
  return Status::OK();
}

std::string GetTensorMappingJson(const poplar::Graph& graph,
                                 const TensorMaps& tensor_maps) {
  Json::Value mappings;

  for (auto tm : tensor_maps) {
    mappings[tm.first] = Json::Value(Json::arrayValue);

    for (auto pair : tm.second) {
      const auto& pop_tensor = pair.second;

      Json::Value tensor;
      tensor["inst_name"] = Json::Value(pair.first.first);
      tensor["output_index"] = Json::Value::UInt64(pair.first.second);
      tensor["constant"] = Json::Value::UInt64(pop_tensor.containsConstant());
      tensor["tiles"] = Json::Value(Json::arrayValue);

      const auto& mapping = graph.getTileMapping(pop_tensor);
      unsigned tiles_used = 0;
      size_t total_elements = 0;

      for (size_t tile_idx = 0; tile_idx < mapping.size(); tile_idx++) {
        const auto& tile = mapping[tile_idx];
        if (tile.size() != 0) {
          tiles_used++;
          size_t tile_element_count = 0;
          for (const auto& interval : tile) {
            tile_element_count += interval.size();
          }

          Json::Value tile;
          tile["tile_id"] = Json::Value::UInt64(tile_idx);
          tile["num_intervals"] = Json::Value::UInt64(tile.size());
          tile["num_elements"] = Json::Value::UInt64(tile_element_count);
          tile["element_type"] =
              Json::Value(pop_tensor.elementType().toString());
          tensor["tiles"].append(tile);

          total_elements += tile_element_count;
        }
      }

      tensor["tiles_used"] = Json::Value::UInt64(tiles_used);
      tensor["total_elements"] = Json::Value::UInt64(total_elements);

      mappings[tm.first].append(tensor);
    }
  }

  Json::Value root;
  root["mappings"] = mappings;

  Json::StreamWriterBuilder json_builder;
  std::string json_msg = Json::writeString(json_builder, root);

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "[Poplar] Dumping tensor mapping";
    VLOG(2) << json_msg;
  }

  return json_msg;
}

}  // namespace poplarplugin
}  // namespace xla
