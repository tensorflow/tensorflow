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
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Engine.hpp>
#include <popstd/TileMapping.hpp>
#include <popstd/Pad.hpp>

namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

port::StatusOr<poplar::Type>
PoplarDataType(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case PRED:
      return poplar::BOOL;
    case S8:
      return poplar::CHAR;
    case S16:
      return poplar::SHORT;
    case S32:
      return poplar::INT;
    case S64:
      return poplar::INT;
    case U8:
      return poplar::UNSIGNED_CHAR;
    case U16:
      return poplar::UNSIGNED_SHORT;
    case U32:
      return poplar::UNSIGNED_INT;
    case U64:
      return poplar::UNSIGNED_INT;
    case F16:
      return poplar::HALF;
    case F32:
      return poplar::FLOAT;
    default:
      return tensorflow::errors::FailedPrecondition(
              port::StrCat("unsupported primitive type in poplar ",
                           shape.element_type()));
  }
}

std::vector<size_t>
PoplarShapeFromXlaShape(const xla::Shape &xla_shape) {
  std::vector <size_t> shape;
  for (auto d : xla_shape.dimensions()) {
    shape.push_back(d);
  }
  return shape;
}

xla::Shape
XlaShapeFromPoplarShape(PrimitiveType element_type,
                        const std::vector<size_t> &poplar_shape) {
  xla::Shape shape;
  shape.set_element_type(element_type);
  for (int64 dimension : poplar_shape) {
    shape.add_dimensions(dimension);
  }
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape;
}


poplar::Tensor
ConvertToDeviceLayout(const Shape& shape, const poplar::Tensor& tensor) {
  // Reshape then dimshuffle
  poplar::Tensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector <std::size_t> dim(rank);
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i=0; i<rank; i++) {
      shuffle[shape.layout().minor_to_major(i)] = rank - i - 1;
      dim[rank - i - 1] = tensor.dim(shape.layout().minor_to_major(i));
    }

    out = out.reshape(dim);
    out = out.dimShuffle(shuffle);
  }
  return out;
}

poplar::Tensor
ConvertFromDeviceLayout(const Shape& shape, const poplar::Tensor& tensor) {
  // Dimshuffle then reshape
  poplar::Tensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i=0; i<rank; i++) {
      shuffle[rank - i - 1] = shape.layout().minor_to_major(i);
    }
    out = out.dimShuffle(shuffle);
    out = out.reshape(tensor.shape());
  }
  return out;
}

port::StatusOr<poplar::Tensor>
AddPlainTensor(poplar::Graph& graph,
               const HloInstruction* inst,
               const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addVariable(poplar_type, dim, inst->name());
  popstd::mapTensorLinearly(graph, out);
  return out;
}

port::StatusOr<poplar::Tensor>
AddRnnSequence(poplar::Graph& graph,
               const HloInstruction* inst,
               const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addVariable(poplar_type, dim, inst->name());

  for (auto i = 0; i != dim[0]; ++i) {
    popstd::mapTensorLinearly(graph, out[i]);
  }

  return out;
}

static port::StatusOr<poplar::Tensor>
AddConvolutionInput(poplar::Graph& graph,
                    const HloInstruction* inst,
                    const HloInstruction* target,
                    CompilerResources& resources) {
  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(target));

  popconv::ConvOptions opts;
  opts.cache = &resources.convolution_cache;

  auto name = port::StrCat(inst->name(), "_input");
  poplar::Tensor out = popconv::createInput(graph, params, name, opts);
  return ShuffleConvolutionInputToTensorflow(target, out);
}

static port::StatusOr<poplar::Tensor>
AddConvolutionWeights(poplar::Graph& graph,
                      const HloInstruction* inst,
                      const HloInstruction* target,
                      CompilerResources& resources) {
  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(target));

  popconv::ConvOptions opts;
  opts.cache = &resources.convolution_cache;

  auto name = port::StrCat(inst->name(), "_weights");
  poplar::Tensor out = popconv::createWeights(graph, params, name, opts);

  out = RemoveGroupsDimensionFromWeights(out);
  return ShuffleConvolutionWeightsToTensorflow(target, out);
}

static port::StatusOr<poplar::Tensor>
AddLeftMatMul(poplar::Graph& graph,
              const HloInstruction* inst,
              const HloInstruction* target,
              CompilerResources& resources) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(inst->shape()));
  poplin::MatMulOptions opts;
  opts.cache = &resources.dot_cache;
  const auto& aShape = PoplarShapeFromXlaShape(target->operand(0)->shape());
  const auto& bShape = PoplarShapeFromXlaShape(target->operand(1)->shape());
  auto name = port::StrCat(inst->name(), "_lhs");
  return poplin::createMatMulInputLHS(graph,type,aShape,bShape, name, opts);
}

static port::StatusOr<poplar::Tensor>
AddRightMatMul(poplar::Graph& graph,
              const HloInstruction* inst,
              const HloInstruction* target,
              CompilerResources& resources) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(inst->shape()));
  poplin::MatMulOptions opts;
  opts.cache = &resources.dot_cache;
  const auto& aShape = PoplarShapeFromXlaShape(target->operand(0)->shape());
  const auto& bShape = PoplarShapeFromXlaShape(target->operand(1)->shape());
  auto name = port::StrCat(inst->name(), "_rhs");
  return poplin::createMatMulInputRHS(graph,type,aShape,bShape, name, opts);
}


port::StatusOr<poplar::Tensor>
AddTensor(poplar::Graph& graph,
          const TensorSource& src,
          const xla::Shape& shape,
          CompilerResources& resources) {
  poplar::Tensor out;

  auto target = resources.tensor_allocation_map.find(src);
  if (target != resources.tensor_allocation_map.end()) {
    switch (target->second.first->opcode()) {
      case HloOpcode::kConvolution:
      {
        switch (target->second.second) {
          case 0:
          {
            TF_ASSIGN_OR_RETURN(out, AddConvolutionInput(graph, src.first,
                                                         target->second.first,
                                                         resources));
            break;
          }
          case 1:
          {
            TF_ASSIGN_OR_RETURN(out, AddConvolutionWeights(graph, src.first,
                                                           target->second.first,
                                                           resources));
            break;
          }
          default:
            return tensorflow::errors::FailedPrecondition(
                    port::StrCat("invalid operand for tensor allocation on ",
                                 src.first->name()));
        }
        break;
      }
      case HloOpcode::kDot:
      {
        switch (target->second.second) {
          case 0:
          {
            TF_ASSIGN_OR_RETURN(out, AddLeftMatMul(graph, src.first,
                                                   target->second.first,
                                                   resources));
            break;
          }
          case 1:
          {
            TF_ASSIGN_OR_RETURN(out, AddRightMatMul(graph, src.first,
                                                    target->second.first,
                                                    resources));
            break;
          }
          default:
            return tensorflow::errors::FailedPrecondition(
                    port::StrCat("invalid operand for tensor allocation on ",
                                 src.first->name()));
        }
        break;
      }
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      {
        if (ShapeUtil::Rank(shape) == 3) {
          TF_ASSIGN_OR_RETURN(out, AddRnnSequence(graph, src.first, shape));
        } else {
          TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, src.first, shape));
        }
        break;
      }
      default:
        return tensorflow::errors::FailedPrecondition(
                port::StrCat("unknown special tensor target on ",
                             src.first->name()));
    }
  } else {
    TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, src.first, shape));
  }
  return out;
}

template<typename TYPE>
static void
AddConstantTensor(poplar::Graph& graph,
                  const xla::Literal& literal,
                  const xla::Shape& shape,
                  const poplar::Type& type,
                  poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));

  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (TYPE)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data[0]);
  } else {
    tensor = graph.addConstant(type, dim, data);
  }

  tensor = ConvertToDeviceLayout(shape, tensor);
}

static void
Add64BitConstantTensor(poplar::Graph&graph,
                  const xla::Literal &literal,
                  const xla::Shape &shape,
                  const poplar::Type &type,
                  poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const void* data(static_cast<const void*>(literal.untyped_data()));

  std::vector<char> converted =
          sep::ConvInt64ToInt32(data, num_elements * sizeof(int64), 0);

  const int32* data32 = reinterpret_cast<const int32*>(converted.data());

  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (int32)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data32[0]);
  } else {
    tensor = graph.addConstant(type, dim, data32);
  }
}

template<typename TYPE>
static void
SetInitialTensorValue(poplar::Graph &graph,
                      poplar::Tensor &tensor,
                      const xla::Literal &literal) {
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));
  graph.setInitialValue<TYPE>(tensor, data);
}

static void
Set64BitInitialTensorValue(poplar::Graph &graph,
                           poplar::Tensor &tensor,
                           const xla::Literal &literal) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  const void* data(static_cast<const void*>(literal.untyped_data()));
  std::vector<char> converted =
          sep::ConvInt64ToInt32(data, num_elements * sizeof(int64), 0);

  const int32* data32 = reinterpret_cast<const int32*>(converted.data());
  graph.setInitialValue<int>(tensor, static_cast<const int*>(data32));
}

port::StatusOr<poplar::Tensor>
AddConstantTensor(poplar::Graph& graph,
                  const TensorSource& src,
                  const xla::Shape& shape,
                  const xla::Literal& literal,
                  CompilerResources& resources) {
  poplar::Tensor tensor;

  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(literal.shape()));

  if (ShapeUtil::ElementsIn(literal.shape()) > 32) {
    TF_ASSIGN_OR_RETURN(tensor, AddTensor(graph, src, shape, resources));
    switch (literal.shape().element_type()) {
      case PRED:
        SetInitialTensorValue<bool>(graph, tensor, literal);
        break;
      case S32:
      case U32:
        SetInitialTensorValue<int>(graph, tensor, literal);
        break;
      case U64:
      case S64:
        Set64BitInitialTensorValue(graph, tensor, literal);
        break;
      case F16:
        SetInitialTensorValue<poplar::IeeeHalf>(graph, tensor, literal);
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
      case U32:
        AddConstantTensor<int>(graph, literal, shape, type, tensor);
        break;
      case U64:
      case S64:
        Add64BitConstantTensor(graph, literal, shape, type, tensor);
        break;
      case F16:
        AddConstantTensor<poplar::IeeeHalf>(graph, literal, shape, type, tensor);
        break;
      case F32:
        AddConstantTensor<float>(graph, literal, shape, type, tensor);
        break;
      default:
        // The unsupported cases were caught in the call to PoplarDataType above
        break;
    }

    std::vector <std::size_t> dim = PoplarShapeFromXlaShape(shape);
    return tensor.reshape(dim);
  }
}

template<typename T>
poplar::Tensor
TileTensor(const T &multiples, const poplar::Tensor &in) {
  poplar::Tensor out = in;
  for (unsigned d = 0; d < multiples.size(); d++) {
    int m = multiples[d];
    out = out.broadcast(m, d);
  }
  return out;
}

template poplar::Tensor
TileTensor<tensorflow::BCast::Vec>(const tensorflow::BCast::Vec &,
                                   const poplar::Tensor &);

template poplar::Tensor
TileTensor<std::vector<std::size_t>>(const std::vector<std::size_t> &,
                                     const poplar::Tensor &);

port::StatusOr<poplar::Tensor>
PadTensor(const PaddingConfig& cfg,
          const poplar::Tensor &in,
          const poplar::Tensor& pad) {
  if (pad.numElements() != 1) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "PadTensor: pad tensor is not single valued");
  }

  poplar::Tensor p(pad.reshape(std::vector<std::size_t>(in.rank(), 1)));

  poplar::Tensor out = in;
  for (unsigned d = 0; d < in.rank(); d++) {
    std::vector<std::size_t> shape(out.shape());

    if (cfg.dimensions(d).interior_padding() > 0 && shape[d] > 0) {
      shape[d] = cfg.dimensions(d).interior_padding();
      poplar::Tensor padded = TileTensor(shape, p);
      poplar::Tensor interleaved = out.slice(0,1,d);
      for (unsigned int slice=1; slice<out.dim(d); slice++) {
        interleaved = poplar::concat(interleaved, padded, d);
        interleaved = poplar::concat(interleaved, out.slice(slice, slice+1, d), d);
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

port::StatusOr<poplar::Tensor>
PadWithConstantZero(poplar::Graph& graph,
                    const PaddingConfig& cfg,
                    const poplar::Tensor &in) {
  std::vector<std::ptrdiff_t> paddingLower;
  std::vector<std::ptrdiff_t> paddingUpper;
  for (auto& d : cfg.dimensions()) {
    paddingLower.push_back(d.edge_padding_low());
    paddingUpper.push_back(d.edge_padding_high());
  }
  return popstd::pad(graph, in, paddingLower, paddingUpper);
}

port::StatusOr<poplar::Tensor>
ReverseTensor(const poplar::Tensor &in,
              const std::vector<int64>& dimensions) {
  poplar::Tensor out = in;
  if (in.numElements() > 0) {
    for (int64 d : dimensions) {
      out = out.reverse(d);
    }
  }
  return out;
}

port::StatusOr<poplar::Tensor>
BroadcastTensor(const poplar::Tensor &in,
                const xla::Shape& out,
                const std::vector<int64>& dimensions) {
  if (PoplarShapeMatchesXLAShape(in, out)) {
    return in;
  }

  tensorflow::BCast::Vec bcast_shape =
          convert_array<tensorflow::BCast::Vec>(out.dimensions());

  tensorflow::BCast::Vec tensor_shape(ShapeUtil::Rank(out), 1);
  if (dimensions.size() > 0) {
    for (size_t d=0; d<dimensions.size(); d++) {
      tensor_shape[dimensions[d]] = in.dim(d);
    }
  } else {
    for (size_t d=0; d<in.rank(); d++) {
      tensor_shape[d] = in.dim(d);
    }
  }

  tensorflow::BCast bcast(tensor_shape, bcast_shape);
  if (!bcast.IsValid()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Incompatible broadcast");
  }

  poplar::Tensor o = in;
  o = in.reshape(convert_array<std::vector<size_t>>(bcast.x_reshape()));
  o = TileTensor(bcast.x_bcast(), o);
  return o.reshape(PoplarShapeFromXlaShape(out));
}

bool
PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                           const xla::Shape& shape) {
  if (tensor.rank() != ShapeUtil::Rank(shape)) return false;
  for (size_t d=0; d<tensor.rank(); d++) {
    if (tensor.dim(d) != (unsigned)shape.dimensions(d)) return false;
  }

  return true;
}

}
}
