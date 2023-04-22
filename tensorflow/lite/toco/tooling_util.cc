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
#include "tensorflow/lite/toco/tooling_util.h"

#include <functional>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "re2/re2.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/dump_graphviz.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_graphviz_dump_options.h"

namespace toco {

// Find the longest common prefix of two strings.
absl::string_view FindLongestCommonPrefix(absl::string_view a,
                                          absl::string_view b) {
  if (a.empty() || b.empty()) return absl::string_view();

  const char* pa = a.data();
  const char* pb = b.data();
  size_t count = 0;
  const size_t limit = std::min(a.size(), b.size());
  while (count < limit && *pa == *pb) {
    ++pa;
    ++pb;
    ++count;
  }

  return absl::string_view(a.data(), count);
}

std::string LogName(const Operator& op) {
  const std::string& opname = HelpfulOperatorTypeName(op);
  if (op.outputs.empty()) {
    return toco::port::StringF("{%s operator}", opname);
  } else {
    return toco::port::StringF("{%s operator with output %s}", opname,
                               op.outputs[0]);
  }
}

std::string ArrayDataTypeName(ArrayDataType data_type) {
  switch (data_type) {
    case ArrayDataType::kFloat:
      return "float";
    case ArrayDataType::kInt8:
      return "int8";
    case ArrayDataType::kUint8:
      return "uint8";
    case ArrayDataType::kInt16:
      return "int16";
    case ArrayDataType::kUint16:
      return "uint16";
    case ArrayDataType::kInt32:
      return "int32";
    case ArrayDataType::kUint32:
      return "uint32";
    case ArrayDataType::kInt64:
      return "int64";
    case ArrayDataType::kUint64:
      return "uint64";
    case ArrayDataType::kString:
      return "string";
    case ArrayDataType::kBool:
      return "bool";
    case ArrayDataType::kComplex64:
      return "complex64";
    case ArrayDataType::kNone:
      return "None";
    default:
      LOG(FATAL) << "Unhandled array data type " << static_cast<int>(data_type);
  }
}

bool IsInputArray(const Model& model, const std::string& array_name) {
  for (const auto& input_array : model.flags.input_arrays()) {
    if (array_name == input_array.name()) {
      return true;
    }
  }
  return false;
}

bool IsOutputArray(const Model& model, const std::string& array_name) {
  for (const auto& output_array : model.flags.output_arrays()) {
    if (array_name == output_array) {
      return true;
    }
  }
  return false;
}

bool IsArrayConsumed(const Model& model, const std::string& name) {
  if (GetOpWithInput(model, name)) {
    return true;
  }
  if (IsOutputArray(model, name)) {
    return true;
  }
  for (const auto& rnn_state : model.flags.rnn_states()) {
    if (rnn_state.back_edge_source_array() == name) {
      return true;
    }
  }
  return false;
}

int CountTrueOutputs(const Model& model, const Operator& op) {
  int count = 0;
  for (const std::string& output : op.outputs) {
    if (IsArrayConsumed(model, output)) {
      ++count;
    }
  }
  return count;
}

int CountOpsWithInput(const Model& model, const std::string& array_name) {
  int count = 0;
  for (const auto& op : model.operators) {
    for (auto& input : op->inputs) {
      if (input == array_name) {
        count++;
        // Breaking here is important: some graphs have ops that use the
        // same array as more than one of their inputs, and in that case
        // we want it counted only once.
        break;
      }
    }
  }
  return count;
}

bool DeleteArrayIfUnused(const std::string& array_name, Model* model) {
  if (IsDiscardableArray(*model, array_name) &&
      CountOpsWithInput(*model, array_name) == 0 &&
      GetOpWithOutput(*model, array_name) == nullptr) {
    model->EraseArray(array_name);
    return true;
  }
  return false;
}

bool DeleteArrayIfUnusedOutsideOfOp(const std::string& array_name,
                                    const Operator* op, Model* model) {
  if (!IsDiscardableArray(*model, array_name)) {
    return false;
  }
  if (CountOpsWithInput(*model, array_name) > 1) {
    return false;
  }
  const Operator* op_having_this_as_input = GetOpWithInput(*model, array_name);
  if (op_having_this_as_input && op_having_this_as_input != op) {
    return false;
  }
  const Operator* op_having_this_as_output =
      GetOpWithOutput(*model, array_name);
  if (op_having_this_as_output && op_having_this_as_output != op) {
    return false;
  }
  model->EraseArray(array_name);
  return true;
}

void DeleteOpAndArrays(Model* model, const Operator* op) {
  for (const std::string& array_name : op->inputs) {
    DeleteArrayIfUnusedOutsideOfOp(array_name, op, model);
  }
  for (const std::string& array_name : op->outputs) {
    DeleteArrayIfUnusedOutsideOfOp(array_name, op, model);
  }
  auto op_it = FindOp(*model, op);
  CHECK(op_it != model->operators.end());
  model->operators.erase(op_it);
}

std::vector<std::unique_ptr<Operator>>::const_iterator FindOpWithOutput(
    const Model& model, const std::string& array_name) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    for (auto& output : it->get()->outputs) {
      if (output == array_name) {
        return it;
      }
    }
  }
  return model.operators.end();
}

std::vector<std::unique_ptr<Operator>>::iterator FindOpWithOutput(
    Model& model, const std::string& array_name) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    for (auto& output : it->get()->outputs) {
      if (output == array_name) {
        return it;
      }
    }
  }
  return model.operators.end();
}

Operator* GetOpWithOutput(const Model& model, const std::string& array_name) {
  auto it = FindOpWithOutput(model, array_name);
  return it == model.operators.end() ? nullptr : it->get();
}

// GetFirstOpWithInput assumes that this finds the first op.
std::vector<std::unique_ptr<Operator>>::const_iterator FindOpWithInput(
    const Model& model, const std::string& array_name) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    for (auto& input : it->get()->inputs) {
      if (input == array_name) {
        return it;
      }
    }
  }
  return model.operators.end();
}

std::vector<std::unique_ptr<Operator>>::iterator FindOpWithInput(
    Model& model, const std::string& array_name) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    for (auto& input : it->get()->inputs) {
      if (input == array_name) {
        return it;
      }
    }
  }
  return model.operators.end();
}

std::vector<std::unique_ptr<Operator>>::const_iterator FindOp(
    const Model& model, const Operator* op) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    if (it->get() == op) {
      return it;
    }
  }
  return model.operators.end();
}

std::vector<std::unique_ptr<Operator>>::iterator FindOp(Model& model,
                                                        const Operator* op) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    if (it->get() == op) {
      return it;
    }
  }
  return model.operators.end();
}

Operator* GetOpWithInput(const Model& model, const std::string& array_name) {
  auto it = FindOpWithInput(model, array_name);
  return it == model.operators.end() ? nullptr : it->get();
}

Operator* GetFirstOpWithInput(const Model& model,
                              const std::string& array_name) {
  auto it = FindOpWithInput(model, array_name);
  return it == model.operators.end() ? nullptr : it->get();
}

void ReplaceArrayUsage(Model* model, const std::string& old_array_name,
                       const std::string& new_array_name) {
  for (auto& op_it : model->operators) {
    Operator* op = op_it.get();
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == old_array_name) {
        op->inputs[i] = new_array_name;
      }
    }
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_array_name) {
        op->outputs[i] = new_array_name;
      }
    }
  }
}

std::string FormatArraysList(const Model& model,
                             const std::vector<std::string>& list) {
  if (list.empty()) {
    return "[]";
  }
  std::string result = "";
  if (list.size() > 1) {
    result += "[ ";
  }
  for (std::size_t i = 0; i < list.size(); i++) {
    if (i > 0) {
      result += ", ";
    }
    result += list[i];
  }
  if (list.size() > 1) {
    result += " ]";
  }
  return result;
}

const char* OperatorTypeName(OperatorType type) {
  switch (type) {
#define HANDLE_OPERATORTYPENAME_CASE(c) \
  case OperatorType::k##c:              \
    return #c;
    HANDLE_OPERATORTYPENAME_CASE(Abs)
    HANDLE_OPERATORTYPENAME_CASE(Add)
    HANDLE_OPERATORTYPENAME_CASE(AddN)
    HANDLE_OPERATORTYPENAME_CASE(AveragePool)
    HANDLE_OPERATORTYPENAME_CASE(BatchMatMul)
    HANDLE_OPERATORTYPENAME_CASE(BatchNormalization)
    HANDLE_OPERATORTYPENAME_CASE(Conv)
    HANDLE_OPERATORTYPENAME_CASE(Concatenation)
    HANDLE_OPERATORTYPENAME_CASE(DepthwiseConv)
    HANDLE_OPERATORTYPENAME_CASE(DepthToSpace)
    HANDLE_OPERATORTYPENAME_CASE(SpaceToDepth)
    HANDLE_OPERATORTYPENAME_CASE(FullyConnected)
    HANDLE_OPERATORTYPENAME_CASE(HardSwish)
    HANDLE_OPERATORTYPENAME_CASE(Dequantize)
    HANDLE_OPERATORTYPENAME_CASE(L2Normalization)
    HANDLE_OPERATORTYPENAME_CASE(LocalResponseNormalization)
    HANDLE_OPERATORTYPENAME_CASE(Log)
    HANDLE_OPERATORTYPENAME_CASE(Logistic)
    HANDLE_OPERATORTYPENAME_CASE(LstmCell)
    HANDLE_OPERATORTYPENAME_CASE(MaxPool)
    HANDLE_OPERATORTYPENAME_CASE(L2Pool)
    HANDLE_OPERATORTYPENAME_CASE(FakeQuant)
    HANDLE_OPERATORTYPENAME_CASE(Mul)
    HANDLE_OPERATORTYPENAME_CASE(RandomUniform)
    HANDLE_OPERATORTYPENAME_CASE(Elu)
    HANDLE_OPERATORTYPENAME_CASE(Relu)
    HANDLE_OPERATORTYPENAME_CASE(Relu1)
    HANDLE_OPERATORTYPENAME_CASE(Relu6)
    HANDLE_OPERATORTYPENAME_CASE(PRelu)
    HANDLE_OPERATORTYPENAME_CASE(ReorderAxes)
    HANDLE_OPERATORTYPENAME_CASE(Softmax)
    HANDLE_OPERATORTYPENAME_CASE(LogSoftmax)
    HANDLE_OPERATORTYPENAME_CASE(Div)
    HANDLE_OPERATORTYPENAME_CASE(Tanh)
    HANDLE_OPERATORTYPENAME_CASE(Sin)
    HANDLE_OPERATORTYPENAME_CASE(All)
    HANDLE_OPERATORTYPENAME_CASE(Assert)
    HANDLE_OPERATORTYPENAME_CASE(ExpandDims)
    HANDLE_OPERATORTYPENAME_CASE(Fill)
    HANDLE_OPERATORTYPENAME_CASE(FloorMod)
    HANDLE_OPERATORTYPENAME_CASE(FloorDiv)
    HANDLE_OPERATORTYPENAME_CASE(Greater)
    HANDLE_OPERATORTYPENAME_CASE(GreaterEqual)
    HANDLE_OPERATORTYPENAME_CASE(Identity)
    HANDLE_OPERATORTYPENAME_CASE(Less)
    HANDLE_OPERATORTYPENAME_CASE(LessEqual)
    HANDLE_OPERATORTYPENAME_CASE(MatMul)
    HANDLE_OPERATORTYPENAME_CASE(ReduceMax)  //  Reduction Max
    HANDLE_OPERATORTYPENAME_CASE(Maximum)    //  Element-wise Maximum
    HANDLE_OPERATORTYPENAME_CASE(Merge)
    HANDLE_OPERATORTYPENAME_CASE(ReduceMin)  //  Reduction Min
    HANDLE_OPERATORTYPENAME_CASE(Minimum)    //  Element-wise Minimum
    HANDLE_OPERATORTYPENAME_CASE(Neg)
    HANDLE_OPERATORTYPENAME_CASE(OneHot)
    HANDLE_OPERATORTYPENAME_CASE(Pack)
    HANDLE_OPERATORTYPENAME_CASE(Pad)
    HANDLE_OPERATORTYPENAME_CASE(PadV2)
    HANDLE_OPERATORTYPENAME_CASE(StridedSlice)
    HANDLE_OPERATORTYPENAME_CASE(Range)
    HANDLE_OPERATORTYPENAME_CASE(Rank)
    HANDLE_OPERATORTYPENAME_CASE(Reshape)
    HANDLE_OPERATORTYPENAME_CASE(Squeeze)
    HANDLE_OPERATORTYPENAME_CASE(Rsqrt)
    HANDLE_OPERATORTYPENAME_CASE(SegmentSum)
    HANDLE_OPERATORTYPENAME_CASE(Shape)
    HANDLE_OPERATORTYPENAME_CASE(Slice)
    HANDLE_OPERATORTYPENAME_CASE(Split)
    HANDLE_OPERATORTYPENAME_CASE(SplitV)
    HANDLE_OPERATORTYPENAME_CASE(Sqrt)
    HANDLE_OPERATORTYPENAME_CASE(Square)
    HANDLE_OPERATORTYPENAME_CASE(Switch)
    HANDLE_OPERATORTYPENAME_CASE(Sub)
    HANDLE_OPERATORTYPENAME_CASE(Sum)
    HANDLE_OPERATORTYPENAME_CASE(Tile)
    HANDLE_OPERATORTYPENAME_CASE(Transpose)
    HANDLE_OPERATORTYPENAME_CASE(TransposeConv)
    HANDLE_OPERATORTYPENAME_CASE(Concat)
    HANDLE_OPERATORTYPENAME_CASE(ConcatV2)
    HANDLE_OPERATORTYPENAME_CASE(Cast)
    HANDLE_OPERATORTYPENAME_CASE(Floor)
    HANDLE_OPERATORTYPENAME_CASE(Ceil)
    HANDLE_OPERATORTYPENAME_CASE(Round)
    HANDLE_OPERATORTYPENAME_CASE(Gather)
    HANDLE_OPERATORTYPENAME_CASE(GatherNd)
    HANDLE_OPERATORTYPENAME_CASE(ResizeBilinear)
    HANDLE_OPERATORTYPENAME_CASE(SpaceToBatchND)
    HANDLE_OPERATORTYPENAME_CASE(BatchToSpaceND)
    HANDLE_OPERATORTYPENAME_CASE(Mean)
    HANDLE_OPERATORTYPENAME_CASE(ReduceProd)
    HANDLE_OPERATORTYPENAME_CASE(Svdf)
    HANDLE_OPERATORTYPENAME_CASE(ArgMax)
    HANDLE_OPERATORTYPENAME_CASE(ArgMin)
    HANDLE_OPERATORTYPENAME_CASE(TopK_V2)
    HANDLE_OPERATORTYPENAME_CASE(Unsupported)
    HANDLE_OPERATORTYPENAME_CASE(Exp)
    HANDLE_OPERATORTYPENAME_CASE(DynamicPartition)
    HANDLE_OPERATORTYPENAME_CASE(DynamicStitch)
    HANDLE_OPERATORTYPENAME_CASE(Select)
    HANDLE_OPERATORTYPENAME_CASE(SparseToDense)
    HANDLE_OPERATORTYPENAME_CASE(Equal)
    HANDLE_OPERATORTYPENAME_CASE(NotEqual)
    HANDLE_OPERATORTYPENAME_CASE(Pow)
    HANDLE_OPERATORTYPENAME_CASE(Any)
    HANDLE_OPERATORTYPENAME_CASE(LogicalAnd)
    HANDLE_OPERATORTYPENAME_CASE(LogicalNot)
    HANDLE_OPERATORTYPENAME_CASE(LogicalOr)
    HANDLE_OPERATORTYPENAME_CASE(CTCBeamSearchDecoder)
    HANDLE_OPERATORTYPENAME_CASE(Unpack)
    HANDLE_OPERATORTYPENAME_CASE(ZerosLike)
    HANDLE_OPERATORTYPENAME_CASE(UnidirectionalSequenceLstm)
    HANDLE_OPERATORTYPENAME_CASE(BidirectionalSequenceLstm)
    HANDLE_OPERATORTYPENAME_CASE(BidirectionalSequenceRnn)
    HANDLE_OPERATORTYPENAME_CASE(ResizeNearestNeighbor)
    HANDLE_OPERATORTYPENAME_CASE(LeakyRelu)
    HANDLE_OPERATORTYPENAME_CASE(SquaredDifference)
    HANDLE_OPERATORTYPENAME_CASE(MirrorPad)
    HANDLE_OPERATORTYPENAME_CASE(Unique)
    HANDLE_OPERATORTYPENAME_CASE(UnidirectionalSequenceRnn)
    HANDLE_OPERATORTYPENAME_CASE(ReverseV2)
    HANDLE_OPERATORTYPENAME_CASE(Cos)
    HANDLE_OPERATORTYPENAME_CASE(Where)
    HANDLE_OPERATORTYPENAME_CASE(ReverseSequence)
    HANDLE_OPERATORTYPENAME_CASE(MatrixDiag)
    HANDLE_OPERATORTYPENAME_CASE(MatrixSetDiag)
    HANDLE_OPERATORTYPENAME_CASE(MatrixDiagV2)
    HANDLE_OPERATORTYPENAME_CASE(MatrixSetDiagV2)
    HANDLE_OPERATORTYPENAME_CASE(MatrixDiagV3)
    HANDLE_OPERATORTYPENAME_CASE(MatrixSetDiagV3)
    HANDLE_OPERATORTYPENAME_CASE(ScatterNd)
    default:
      LOG(FATAL) << "Unhandled op type";
#undef HANDLE_OPERATORTYPENAME_CASE
  }
}

std::string HelpfulOperatorTypeName(const Operator& op) {
  if (op.type == OperatorType::kUnsupported) {
    return toco::port::StringF(
        "(Unsupported TensorFlow op: %s)",
        static_cast<const TensorFlowUnsupportedOperator&>(op).tensorflow_op);
  }
  return OperatorTypeName(op.type);
}

bool OperatorSupportsFusedActivation(OperatorType type) {
  switch (type) {
    case OperatorType::kAdd:
    case OperatorType::kAveragePool:
    case OperatorType::kBatchNormalization:
    case OperatorType::kConv:
    case OperatorType::kDepthwiseConv:
    case OperatorType::kDiv:
    case OperatorType::kFullyConnected:
    case OperatorType::kL2Pool:
    case OperatorType::kMaxPool:
    case OperatorType::kMul:
    case OperatorType::kSub:
    case OperatorType::kSquaredDifference:
      return true;
    default:
      return false;
  }
}

void LogSummary(int log_level, const Model& model) {
  VLOG(log_level) << "Operators summary (" << model.operators.size()
                  << " operators):";
  std::unordered_multiset<OperatorType> ops_by_type;
  for (const auto& op : model.operators) {
    ops_by_type.insert(op->type);
  }
  auto it = ops_by_type.begin();
  while (it != ops_by_type.end()) {
    int count = ops_by_type.count(*it);
    VLOG(log_level) << "    " << OperatorTypeName(*it) << ": " << count;
    std::advance(it, count);
  }
}

void LogArray(int log_level, const Model& model, const std::string& name) {
  VLOG(log_level) << "Array: " << name;
  if (!model.HasArray(name)) {
    VLOG(log_level) << "  DOES NOT EXIST";
    return;
  }
  const auto& array = model.GetArray(name);
  VLOG(log_level) << "  Data type: " << ArrayDataTypeName(array.data_type);
  VLOG(log_level) << "  Final type: "
                  << ArrayDataTypeName(array.final_data_type);
  if (array.buffer) {
    VLOG(log_level) << "  Constant Buffer";
  }
  if (array.alloc) {
    VLOG(log_level) << "  Transient Alloc";
  }
  if (array.has_shape()) {
    const Shape& array_shape = array.shape();
    if (array_shape.dimensions_count() == 0) {
      VLOG(log_level) << "  (Zero dimensions)";
    } else {
      std::string message = "  Dims: ";
      bool first = true;
      for (const int dim : array_shape.dims()) {
        if (!first) {
          message += ", ";
        }
        first = false;
        toco::port::AppendF(&message, "%d", dim);
      }
      VLOG(log_level) << message;
    }
  }
  if (array.minmax) {
    VLOG(log_level) << "  MinMax: " << array.minmax->min << " .. "
                    << array.minmax->max;
  }
  if (array.quantization_params) {
    VLOG(log_level) << "  QuantizationParams: zero_point="
                    << static_cast<int>(array.quantization_params->zero_point)
                    << ", scale=" << array.quantization_params->scale;
  }
}

void DumpGraphvizVideoFrame(const Model& model) {
  namespace port = toco::port;

  const auto& dump_options = *GraphVizDumpOptions::singleton();
  if (!dump_options.dump_graphviz_video) {
    return;
  }
  CHECK(!dump_options.dump_graphviz.empty());
  // TODO(benoitjacob): the static data here means that this function
  // is stateful, not reentrant, and effectively leaks memory till exit
  // (since dump_hashes can only grow in size). It also means that it
  // really only is intended to be called for a single model during the
  // process' lifetime. So it's not great design at all. The overriding
  // design aspect here is to make the video-dumping code as unintrusive
  // and self-contained as possible. Eventually, we'll want to have that
  // cleaned-up, but that will require some form of general statefulness
  // in toco (some kind of 'tooling state' data structure) that does
  // not exist at present, and would be premature to design here just for
  // this new video-dumping feature.
  static int dump_id = 0;
  static std::unordered_set<std::size_t> dump_hashes;
  std::string graphviz_dump;
  DumpGraphviz(model, &graphviz_dump,
               toco::port::StringF("VIDEO frame:%05d", dump_id));
  std::size_t hash = std::hash<std::string>{}(graphviz_dump);
  if (!dump_hashes.count(hash)) {
    LOG(INFO) << "DUMPING GRAPHVIZ VIDEO FRAME: " << dump_id;
    dump_hashes.insert(hash);
    const auto result = port::file::SetContents(
        port::file::JoinPath(
            dump_options.dump_graphviz,
            toco::port::StringF("toco_video_%05d.dot", dump_id)),
        graphviz_dump, port::file::Defaults());
    QCHECK(result.ok()) << result.error_message();
    dump_id++;
  }
}

void LogDump(int log_level, const std::string& message, const Model& model) {
  namespace port = toco::port;
  const auto& dump_options = *GraphVizDumpOptions::singleton();

  DumpGraphvizVideoFrame(model);
  if (!dump_options.dump_graphviz.empty()) {
    std::string graphviz_dump;

    DumpGraphviz(model, &graphviz_dump, message);
    const auto result = port::file::SetContents(
        port::file::JoinPath(
            dump_options.dump_graphviz,
            absl::StrCat("toco_", absl::StrReplaceAll(message, {{" ", "_"}}),
                         ".dot")),
        graphviz_dump, port::file::Defaults());
    QCHECK(result.ok()) << result.error_message();
  }

  if (!VLOG_IS_ON(log_level)) {
    return;
  }
  VLOG(log_level) << "BEGIN DUMP OF TOCO MODEL (" << message << ")";
  LogSummary(log_level, model);
  std::unordered_set<std::string> already_printed_arrays;
  for (const auto& op : model.operators) {
    for (const auto& input : op->inputs) {
      if (!already_printed_arrays.count(input)) {
        already_printed_arrays.insert(input);
        LogArray(log_level, model, input);
      }
    }
    VLOG(log_level) << HelpfulOperatorTypeName(*op) << " :";
    VLOG(log_level) << "  " << FormatArraysList(model, op->inputs) << " -> "
                    << FormatArraysList(model, op->outputs);
    if (op->fused_activation_function != FusedActivationFunctionType::kNone) {
      VLOG(log_level) << "    (with fused activation function)";
    }
    for (const auto& output : op->outputs) {
      if (!already_printed_arrays.count(output)) {
        already_printed_arrays.insert(output);
        LogArray(log_level, model, output);
      }
    }
  }
  VLOG(log_level) << "END DUMP OF TOCO MODEL (" << message << ")";
}

// Note remaining raw-array extension in ProcessTensorFlowReshapeOperator().
void ExtendShape(Shape* shape, int new_shape_size) {
  CHECK_GE(new_shape_size, shape->dimensions_count());
  const int size_increase = new_shape_size - shape->dimensions_count();
  auto* shape_dims = shape->mutable_dims();
  shape_dims->insert(shape_dims->begin(), size_increase, 1);
}

// TODO(b/62904716) Remove along with remaining uses.
void UnextendShape(Shape* shape, int new_shape_size) {
  CHECK_LE(new_shape_size, shape->dimensions_count());
  const int size_reduction = shape->dimensions_count() - new_shape_size;
  for (int i = 0; i < size_reduction; i++) {
    CHECK_EQ(shape->dims(i), 1);
  }
  std::vector<int>& shape_dims = *shape->mutable_dims();
  shape_dims.erase(shape_dims.begin(), shape_dims.begin() + size_reduction);
}

// In general, zero-sized dimensions are disallowed, but there are exceptions,
// e.g., if the tensor data itself represents a scalar (rank 0) shape, its
// shape will have dimensions [0]. CheckNonEmptyShapeDimensions is more
// strict, and is appropriate for ops and comparisons where an empty shape
// doesn't make sense.
template <typename Dims>
void CheckValidShapeDimensions(const Dims& dims) {
  if (dims.size() == 1 && dims[0] == 0) {
    return;
  }
  for (const auto& dim : dims) {
    CHECK_GE(dim, 1);
  }
}

void CheckValidShape(const Shape& shape) {
  CheckValidShapeDimensions(shape.dims());
}

bool IsNonEmpty(const Shape& shape) {
  for (int i = 0; i < shape.dimensions_count(); ++i) {
    if (shape.dims(i) < 1) return false;
  }
  return true;
}

void CheckNonEmptyShapeDimensions(const Shape& shape) {
  for (int i = 0; i < shape.dimensions_count(); ++i) {
    CHECK_GE(shape.dims()[i], 1) << "shape has dimension 0 at index << " << i
                                 << ". shape = " << ShapeToString(shape);
  }
}

bool ShapesAgreeUpToBroadcasting(const Shape& shape0, const Shape& shape1) {
  CheckNonEmptyShapeDimensions(shape0);
  CheckNonEmptyShapeDimensions(shape1);

  const Shape* longer = &shape0;
  const Shape* shorter = &shape1;
  if (shape1.dimensions_count() > shape0.dimensions_count()) {
    longer = &shape1;
    shorter = &shape0;
  }

  // Walk dimensions back to front until we run out of dimensions in the shorter
  // shape.
  int longer_index = longer->dimensions_count() - 1;
  int shorter_index = shorter->dimensions_count() - 1;
  while (shorter_index >= 0) {
    const int d_long = longer->dims(longer_index);
    const int d_short = shorter->dims(shorter_index);
    // Broadcasting fails if the dimensions are different *and* neither is 1.
    if ((d_long != d_short) && (d_long != 1) && (d_short != 1)) {
      return false;
    }
    longer_index--;
    shorter_index--;
  }
  return true;
}

bool ShapesAgreeUpToExtending(const Shape& shape0, const Shape& shape1) {
  CheckNonEmptyShapeDimensions(shape0);
  CheckNonEmptyShapeDimensions(shape1);

  const Shape* longer = &shape0;
  const Shape* shorter = &shape1;
  if (shape1.dimensions_count() > shape0.dimensions_count()) {
    longer = &shape1;
    shorter = &shape0;
  }

  // Walk dimensions back to front until we run out of dimensions in the shorter
  // shape.
  int longer_index = longer->dimensions_count() - 1;
  int shorter_index = shorter->dimensions_count() - 1;
  while (shorter_index >= 0) {
    const int d_long = longer->dims(longer_index);
    const int d_short = shorter->dims(shorter_index);
    // Extending fails if the dimensions are different.
    if (d_long != d_short) {
      return false;
    }
    longer_index--;
    shorter_index--;
  }

  // The remaining dimensions in the longer shape must be 1.
  while (longer_index >= 0) {
    const int d_long = longer->dims(longer_index);
    if (d_long != 1) {
      return false;
    }
    longer_index--;
  }

  return true;
}

int RequiredBufferSizeForShape(const Shape& shape) {
  CheckValidShape(shape);
  int max_offset = 1;
  for (const auto& dim : shape.dims()) {
    max_offset *= dim;
  }
  return max_offset;
}

bool IsConstantParameterArray(const Model& model, const std::string& name) {
  if (!model.HasArray(name)) {
    return false;
  }

  return !!model.GetArray(name).buffer;
}

namespace {
template <ArrayDataType A>
bool CompareArrayBuffers(const Array& lhs_array, const Array& rhs_array) {
  CHECK(lhs_array.data_type == rhs_array.data_type) << "Data types must match";
  CHECK(lhs_array.buffer) << "LHS must be constant";
  CHECK(rhs_array.buffer) << "RHS must be constant";
  const auto& lhs_data = lhs_array.GetBuffer<A>().data;
  const auto& rhs_data = rhs_array.GetBuffer<A>().data;
  CHECK_EQ(lhs_data.size(), rhs_data.size())
      << "Buffer sizes must match in element count";
  for (int i = 0; i < lhs_data.size(); ++i) {
    if (lhs_data[i] != rhs_data[i]) {
      return false;
    }
  }
  return true;
}

bool HaveSameMinMax(const Array& lhs_array, const Array& rhs_array) {
  if (lhs_array.minmax || rhs_array.minmax) {
    if (!lhs_array.minmax || !rhs_array.minmax) {
      return false;
    }
    if (!(*lhs_array.minmax == *rhs_array.minmax)) {
      return false;
    }
  }
  return true;
}

bool HaveSameQuantizationParams(const Array& lhs_array,
                                const Array& rhs_array) {
  if (lhs_array.quantization_params || rhs_array.quantization_params) {
    if (!lhs_array.quantization_params || !rhs_array.quantization_params) {
      return false;
    }
    if (!(*lhs_array.quantization_params == *rhs_array.quantization_params)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool CompareConstantArrays(const Array& lhs_array, const Array& rhs_array) {
  bool attrs_equal = lhs_array.shape() == rhs_array.shape() &&
                     lhs_array.data_type == rhs_array.data_type &&
                     lhs_array.final_data_type == rhs_array.final_data_type &&
                     HaveSameMinMax(lhs_array, rhs_array) &&
                     HaveSameQuantizationParams(lhs_array, rhs_array) &&
                     lhs_array.narrow_range == rhs_array.narrow_range;
  if (!attrs_equal) {
    return false;
  }
  switch (lhs_array.data_type) {
    case ArrayDataType::kBool:
      return CompareArrayBuffers<ArrayDataType::kBool>(lhs_array, rhs_array);
    case ArrayDataType::kFloat:
      return CompareArrayBuffers<ArrayDataType::kFloat>(lhs_array, rhs_array);
    case ArrayDataType::kInt8:
      return CompareArrayBuffers<ArrayDataType::kInt8>(lhs_array, rhs_array);
    case ArrayDataType::kUint8:
      return CompareArrayBuffers<ArrayDataType::kUint8>(lhs_array, rhs_array);
    case ArrayDataType::kInt16:
      return CompareArrayBuffers<ArrayDataType::kInt16>(lhs_array, rhs_array);
    case ArrayDataType::kUint16:
      return CompareArrayBuffers<ArrayDataType::kUint16>(lhs_array, rhs_array);
    case ArrayDataType::kInt32:
      return CompareArrayBuffers<ArrayDataType::kInt32>(lhs_array, rhs_array);
    case ArrayDataType::kUint32:
      return CompareArrayBuffers<ArrayDataType::kUint32>(lhs_array, rhs_array);
    case ArrayDataType::kInt64:
      return CompareArrayBuffers<ArrayDataType::kInt64>(lhs_array, rhs_array);
    case ArrayDataType::kUint64:
      return CompareArrayBuffers<ArrayDataType::kUint64>(lhs_array, rhs_array);
    case ArrayDataType::kString:
      return CompareArrayBuffers<ArrayDataType::kString>(lhs_array, rhs_array);
    case ArrayDataType::kComplex64:
      return CompareArrayBuffers<ArrayDataType::kComplex64>(lhs_array,
                                                            rhs_array);
    default:
      LOG(FATAL) << "Unsupported data type: "
                 << ArrayDataTypeName(lhs_array.data_type);
      return false;
  }
}

namespace {
// Take an array name, which may be something like "name:3_5" and make it
// acceptable as a TF node name, say "name_3_5";
std::string SanitizeNameForTFNode(const std::string& array_name) {
  auto node_name = array_name;
  std::replace(node_name.begin(), node_name.end(), ':', '_');
  return node_name;
}

void CheckInputArraysAreNotOutputArrays(const ModelFlags& model_flags) {
  for (const auto& input_array : model_flags.input_arrays()) {
    for (const std::string& output_array : model_flags.output_arrays()) {
      QCHECK_NE(input_array.name(), output_array)
          << "The array " << output_array
          << " is listed in both --input_arrays and --output_arrays.";
    }
  }
}

bool IsAsciiPrintable(const std::string& name) {
  for (char c : name) {
    if (!absl::ascii_isprint(c)) {
      return false;
    }
  }
  return true;
}

std::string DumpAscii(const std::string& name) {
  std::string result;
  port::AppendF(&result, "ASCII | Hex\n");
  port::AppendF(&result, "------+----\n");
  for (char c : name) {
    if (absl::ascii_isprint(c)) {
      port::AppendF(&result, "%c     | %x\n", c, c);
    } else {
      port::AppendF(&result, "      | %x   Not ASCII printable!\n", c);
    }
  }
  return result;
}

void CheckNonAsciiIOArrays(const ModelFlags& model_flags) {
  if (model_flags.allow_nonascii_arrays()) {
    return;
  }
  for (const auto& input_array : model_flags.input_arrays()) {
    QCHECK(IsAsciiPrintable(input_array.name()))
        << "Non-ASCII-printable character found in --input_arrays: "
        << input_array.name()
        << ". Pass --allow_nonascii_arrays to allow that. "
        << "Here is a dump of the string:\n\n"
        << DumpAscii(input_array.name());
  }
  for (const std::string& output_array : model_flags.output_arrays()) {
    QCHECK(IsAsciiPrintable(output_array))
        << "Non-ASCII-printable character found in --output_arrays: "
        << output_array << ". Pass --allow_nonascii_arrays to allow that. "
        << "Here is a dump of the string:\n\n"
        << DumpAscii(output_array);
  }
}

void CheckNonExistentIOArrays(const Model& model) {
  // "non-existent" is interpreted in the stronger sense of
  // "not actually produced/consumed by an op".
  // Rationale: we have to artificially fix up TensorFlow graphs by creating
  // any array that it refers to, so just checking that arrays exist isn't
  // sufficient. The real invariant here is whether arrays are produced/consumed
  // by something.
  if (model.flags.allow_nonexistent_arrays()) {
    return;
  }
  static constexpr char general_comment[] =
      "Is it a typo? This should not happen. If you trigger this error "
      "please send a bug report (with code to reproduce this error), to the "
      "TensorFlow Lite team.";
  for (const std::string& output_array : model.flags.output_arrays()) {
    if (IsConstantParameterArray(model, output_array)) {
      continue;  // It is OK to request that a constant be an output.
    }
    QCHECK(GetOpWithOutput(model, output_array))
        << "Specified output array \"" << output_array
        << "\" is not produced by any op in this graph. " << general_comment;
  }
  for (const auto& rnn_state : model.flags.rnn_states()) {
    if (!rnn_state.discardable()) {
      // Check that all RNN states are consumed
      QCHECK(GetOpWithInput(model, rnn_state.state_array()))
          << "Specified RNN state \"" << rnn_state.state_array()
          << "\" is not consumed by any op in this graph. " << general_comment;
      // Check that all RNN back-edge source arrays are produced
      QCHECK(GetOpWithOutput(model, rnn_state.back_edge_source_array()))
          << "Specified RNN back-edge source array \""
          << rnn_state.back_edge_source_array()
          << "\" is not produced by any op in this graph. " << general_comment;
    }
  }
}

}  // namespace

void CheckNoMissingArray(const Model& model) {
  for (const auto& op : model.operators) {
    for (const auto& input : op->inputs) {
      CHECK(model.HasArray(input) || model.optional_arrays.count(input))
          << "Input: " << input << " missing for op: " << op->outputs[0] << ".";
    }
    for (const auto& output : op->outputs) {
      CHECK(model.HasArray(output)) << "Output: " << output << " missing.";
    }
  }
  CheckNonExistentIOArrays(model);
}

void FixNoMissingArray(Model* model) {
  for (const auto& op : model->operators) {
    for (const auto& input : op->inputs) {
      if (!model->HasArray(input) && !model->IsOptionalArray(input)) {
        model->GetOrCreateArray(input);
      }
    }
    for (const auto& output : op->outputs) {
      if (!model->HasArray(output) && !model->IsOptionalArray(output)) {
        model->GetOrCreateArray(output);
      }
    }
  }
  if (model->flags.allow_nonexistent_arrays()) {
    for (const std::string& output_array : model->flags.output_arrays()) {
      model->GetOrCreateArray(output_array);
    }
    for (const auto& rnn_state : model->flags.rnn_states()) {
      model->GetOrCreateArray(rnn_state.state_array());
      model->GetOrCreateArray(rnn_state.back_edge_source_array());
    }
  }
}

void CheckNoOrphanedArray(const Model& model) {
  std::unordered_set<std::string> arrays_without_known_use;
  for (const auto& array : model.GetArrayMap()) {
    if (IsDiscardableArray(model, array.first)) {
      arrays_without_known_use.insert(array.first);
    }
  }
  for (const auto& op : model.operators) {
    for (const auto& input : op->inputs) {
      arrays_without_known_use.erase(input);
    }
    for (const auto& output : op->outputs) {
      arrays_without_known_use.erase(output);
    }
  }
  for (const auto& rnn_state : model.flags.rnn_states()) {
    arrays_without_known_use.erase(rnn_state.state_array());
    arrays_without_known_use.erase(rnn_state.back_edge_source_array());
  }
  if (!arrays_without_known_use.empty()) {
    for (const auto& array : arrays_without_known_use) {
      LOG(INFO) << "Error: Orphaned array: " << array;
    }
  }
  CHECK(arrays_without_known_use.empty());
}

void FixNoOrphanedArray(Model* model) {
  std::unordered_set<std::string> arrays_without_known_use;
  for (const auto& array : model->GetArrayMap()) {
    arrays_without_known_use.insert(array.first);
  }
  for (const auto& op : model->operators) {
    for (const auto& input : op->inputs) {
      arrays_without_known_use.erase(input);
    }
    for (const auto& output : op->outputs) {
      arrays_without_known_use.erase(output);
    }
  }
  for (const auto& rnn_state : model->flags.rnn_states()) {
    arrays_without_known_use.erase(rnn_state.state_array());
    arrays_without_known_use.erase(rnn_state.back_edge_source_array());
  }
  for (const auto& array : arrays_without_known_use) {
    if (IsDiscardableArray(*model, array)) {
      model->EraseArray(array);
    }
  }
}

// Apply checks to arrays individually (for-each fashion).
//
// Check consistency of array fields, check name.
void CheckEachArray(const Model& model) {
  for (const auto& array_entry : model.GetArrayMap()) {
    const auto& array = array_entry.second;
    // It's OK to have a buffer or an alloc, but not both.
    // (Since allocs are for transient arrays without a buffer).
    CHECK(!array->buffer || !array->alloc) << "Tensor: " << array_entry.first;
    if (array->buffer) {
      // If there is a buffer, its type should be consistent with data_type.
      CHECK(array->buffer->type == array->data_type)
          << "Tensor: " << array_entry.first;
      // The presence of a fixed buffer should imply the presence of a fixed
      // shape.
      CHECK(array->has_shape()) << array_entry.first;
      // Constant buffer should has a valid shape.
      CheckValidShape(array->shape());
      // The shape flat-size should agree with the buffer length.
      CHECK_EQ(array->buffer->Length(),
               RequiredBufferSizeForShape(array->shape()))
          << "Tensor: " << array_entry.first;
    }

    // Check name.  Either "name_with_suffix_8", "name_with_port:3", but not
    // "name_with_both:3_8".
    const std::string& name = array_entry.first;
    auto colon_pos = name.find_first_of(':');
    if (colon_pos != std::string::npos) {
      CHECK_EQ(name.substr(colon_pos + 1).find_first_not_of("0123456789"),
               std::string::npos)
          << "Array '" << name << "' has non-digit characters after colon.";
    }
    CHECK_GT(colon_pos, 0) << "Array '" << name
                           << "' must not start with a colon.";
  }
}

void CheckOperatorOrdering(const Model& model) {
  std::unordered_set<std::string> arrays_behind_us;
  for (const auto& array_entry : model.GetArrayMap()) {
    if (!GetOpWithOutput(model, array_entry.first)) {
      arrays_behind_us.insert(array_entry.first);
    }
  }
  arrays_behind_us.insert(model.optional_arrays.begin(),
                          model.optional_arrays.end());
  for (const auto& op : model.operators) {
    for (const auto& input : op->inputs) {
      if (!IsConstantParameterArray(model, input)) {
        CHECK(arrays_behind_us.count(input));
      }
    }
    for (const auto& output : op->outputs) {
      CHECK(!arrays_behind_us.count(output));
      arrays_behind_us.insert(output);
    }
  }
  for (const std::string& output_array : model.flags.output_arrays()) {
    CHECK(arrays_behind_us.count(output_array));
  }
}

void FixOperatorOrdering(Model* model) {
  std::unordered_set<std::string> arrays_behind_us;
  for (const auto& array_entry : model->GetArrayMap()) {
    if (!GetOpWithOutput(*model, array_entry.first)) {
      arrays_behind_us.insert(array_entry.first);
    }
  }
  arrays_behind_us.insert(model->optional_arrays.begin(),
                          model->optional_arrays.end());
  std::vector<std::unique_ptr<Operator>> old_operators;
  std::swap(old_operators, model->operators);
  std::set<std::size_t> remaining;
  for (std::size_t i = 0; i < old_operators.size(); i++) {
    remaining.insert(i);
  }
  std::unordered_map<std::string, std::string> reason_why_leftover;
  while (true) {
    bool inserted_something = false;
    for (const auto& i : remaining) {
      bool can_insert = true;
      auto& op = old_operators[i];
      CHECK(op);
      for (const auto& input : op->inputs) {
        if (!IsConstantParameterArray(*model, input) &&
            !arrays_behind_us.count(input)) {
          for (const std::string& output : op->outputs) {
            reason_why_leftover[output] = input;
          }
          can_insert = false;
          break;
        }
      }
      if (can_insert) {
        model->operators.emplace_back(nullptr);
        for (const auto& output : op->outputs) {
          arrays_behind_us.insert(output);
        }
        std::swap(op, model->operators.back());
        remaining.erase(i);
        inserted_something = true;
        break;
      }
    }
    if (!inserted_something) {
      break;
    }
  }
  if (!remaining.empty()) {
    LOG(ERROR)
        << "No viable ordering of operators was found. "
        << "Here is a 'backtrace' of at least one part of the graph that is "
        << "problematic. It starts with the first operator that has as "
        << "problematic input array, and then walks back the graph to "
        << "the operator that produced that input array, etc., until we find "
        << "the root cause:";
    LOG(ERROR) << "BEGIN TRACE OF OPERATOR WITH BAD INPUT";
    LOG(ERROR) << "Here is the first-encountered operator with a bad input: ";
    const Operator* bad_op = old_operators[*remaining.begin()].get();
    std::unordered_set<std::string> bad_inputs_already_traced;
    // The following while(true) loop should always end with a LOG(FATAL).
    while (true) {
      LOG(ERROR) << HelpfulOperatorTypeName(*bad_op) << " : "
                 << FormatArraysList(*model, bad_op->inputs) << " -> "
                 << FormatArraysList(*model, bad_op->outputs);
      bool found_bad_output = false;
      std::string bad_output;
      for (const std::string& output : bad_op->outputs) {
        if (reason_why_leftover.count(output)) {
          found_bad_output = true;
          bad_output = output;
          break;
        }
      }
      CHECK(found_bad_output);
      const std::string& bad_input = reason_why_leftover[bad_output];
      LOG(ERROR) << "The bad input here is: " << bad_input;
      if (bad_inputs_already_traced.count(bad_input)) {
        LOG(FATAL)
            << "Cycle found! We already encountered that "
            << "input array, " << bad_input << ", earlier in the "
            << "above trace! We expect graphs to be acyclic, even "
            << "RNNs. Let us know if some graph actually needs to have "
            << "cycles, but first, please check if it really is "
            << "an *inference* graph. *Training* graphs are out-of-scope "
            << "for toco.";
      }
      bad_inputs_already_traced.insert(bad_input);
      bad_op = nullptr;
      for (const auto& i : remaining) {
        const Operator* op = old_operators[i].get();
        for (const std::string& output : op->outputs) {
          if (bad_input == output) {
            bad_op = op;
            break;
          }
        }
        if (bad_op) {
          break;
        }
      }
      if (!bad_op) {
        LOG(ERROR) << "And that's the root cause: "
                   << "that array, " << bad_input << ", isn't produced by any "
                   << "operator, or provided in any other way.";
        LOG(ERROR) << "END TRACE OF OPERATOR WITH BAD INPUT";
        LOG(FATAL) << "(The above was a multi-line fatal error)";
      }
      LOG(ERROR) << "And that array is the output of the following operator:";
    }
  }
  CHECK(remaining.empty())
      << "Should never get here! In case of bad graph, "
      << "the above code should have generated a FATAL error already!";
}

void CheckInvariants(const Model& model) {
  CheckInputArraysAreNotOutputArrays(model.flags);
  CheckNonAsciiIOArrays(model.flags);
  CheckNoMissingArray(model);
  CheckNoOrphanedArray(model);
  CheckEachArray(model);
  CheckOperatorOrdering(model);
}

void CheckCountInRange(const ::toco::ModelFlags::ModelCheck& model_check,
                       const int count, const std::string& count_description) {
  if (model_check.count_min() >= 0) {
    CHECK_GE(count, model_check.count_min())
        << "Mismatch in " << count_description << ": count  was " << count
        << ", but the specified "
        << (model_check.count_max() > model_check.count_min() ? "minimum"
                                                              : "value")
        << " was " << model_check.count_min() << ".";
  }
  if (model_check.count_max() > model_check.count_min()) {
    CHECK_LE(count, model_check.count_max())
        << "Mismatch in " << count_description << ": count  was " << count
        << ", but the specified maximum was " << model_check.count_max() << ".";
  }
}

void CheckModelCounts(const Model& model) {
  std::unordered_multiset<OperatorType> ops_by_type;
  std::unordered_map<std::string, OperatorType> op_type_by_name;
  if (model.flags.model_checks_size() == 0) {
    return;
  }

  for (const auto& op : model.operators) {
    ops_by_type.insert(op->type);
    op_type_by_name[OperatorTypeName(op->type)] = op->type;
  }
  for (const auto& model_check : model.flags.model_checks()) {
    std::string count_type = model_check.count_type();
    if (count_type == "None") {
      continue;
    } else if (count_type == "Arrays") {
      CheckCountInRange(model_check, model.GetArrayMap().size(),
                        "count of arrays");
    } else if (count_type == "Total") {
      CheckCountInRange(model_check, model.operators.size(),
                        "count of all operator instances");
    } else {
      // The check type is not itself checked against the set of valid
      // operators, mainly because the enum set cannot be iterated in C++.
      const int found_count =
          op_type_by_name.count(count_type) > 0
              ? ops_by_type.count(op_type_by_name[count_type])
              : 0;
      CheckCountInRange(model_check, found_count,
                        "count of instances of " + count_type + " operator");
    }
  }
}

void FixEdgeArrays(Model* model) {
  for (const std::string& output_array_name : model->flags.output_arrays()) {
    if (!GetOpWithOutput(*model, output_array_name)) {
      // Output has no operator producing it. Change that by inserting a copy.
      LOG(WARNING) << "Fixing constant output array " << output_array_name
                   << " by inserting a copy. This is not optimal.";
      std::string intermediate_array_name =
          AvailableArrayName(*model, output_array_name + "_copy");
      CloneArray(model, output_array_name, intermediate_array_name);
      InsertCopyOperator(model, intermediate_array_name, output_array_name);
    }
  }
}

void DedupeConstantArrays(Model* model, size_t min_size) {
  // Walk all 0..N and compare with the remaining n+1..N.
  // This lets us avoid N^2 comparisons and erase duplicate arrays while
  // iterating.
  const auto& array_map = model->GetArrayMap();
  for (auto lhs_array_it = array_map.begin(); lhs_array_it != array_map.end();
       ++lhs_array_it) {
    const auto& lhs_array_name = lhs_array_it->first;
    const auto& lhs_array = *lhs_array_it->second;
    if (!IsConstantParameterArray(*model, lhs_array_name)) {
      // Not a constant array; skip.
      continue;
    }
    ArrayDataType final_data_type =
        lhs_array.final_data_type != ArrayDataType::kNone
            ? lhs_array.final_data_type
            : lhs_array.data_type;
    // Ignore small arrays, don't check string arrays because it is not possible
    // to estimate its size.
    if (final_data_type != ArrayDataType::kString) {
      size_t array_byte_size =
          lhs_array.buffer->Length() * ElementSize(final_data_type);
      if (array_byte_size < min_size) {
        // Too small; skip.
        continue;
      }
    }

    auto next_lhs_array_it = lhs_array_it;
    ++next_lhs_array_it;
    for (auto rhs_array_it = next_lhs_array_it;
         rhs_array_it != array_map.end();) {
      const auto& rhs_array_name = rhs_array_it->first;
      const auto& rhs_array = *rhs_array_it->second;
      ++rhs_array_it;
      if (!IsConstantParameterArray(*model, rhs_array_name)) {
        // Not a constant array; skip.
        continue;
      }
      if (!IsDiscardableArray(*model, rhs_array_name)) {
        // Can't remove the array as it's not discardable (such as an IO edge).
        continue;
      }
      if (!CompareConstantArrays(lhs_array, rhs_array)) {
        // Arrays aren't equal; skip.
        continue;
      }

      // Arrays can be deduped!
      VLOG(1) << "Deduplicating arrays; using " << lhs_array_name
              << " in place of " << rhs_array_name;
      ReplaceArrayUsage(model, rhs_array_name, lhs_array_name);
      // Note: rhs_array_it above is already incremented so this is safe.
      model->EraseArray(rhs_array_name);
    }
  }
}

namespace {
void CopyArrayAttribs(const Array& source_array, Array* target_array) {
  target_array->data_type = source_array.data_type;
  target_array->final_data_type = source_array.final_data_type;
  if (source_array.has_shape()) {
    target_array->copy_shape(source_array.shape());
  }

  if (source_array.minmax) {
    target_array->GetOrCreateMinMax() = source_array.GetMinMax();
  } else {
    target_array->minmax.reset();
  }

  if (source_array.quantization_params) {
    target_array->GetOrCreateQuantizationParams() =
        source_array.GetQuantizationParams();
  } else {
    target_array->quantization_params.reset();
  }
}
}  // namespace

void InsertCopyOperator(Model* model, const std::string& source_array_name,
                        const std::string& target_array_name) {
  // Reshape to the same size. This should be a no-op.
  const Array& source_array = model->GetArray(source_array_name);
  std::vector<int> shape = source_array.shape().dims();

  // Drop constant data from the target array as the copy will be done at
  // runtime.
  Array& target_array = model->GetOrCreateArray(target_array_name);
  target_array.buffer.reset();
  CopyArrayAttribs(source_array, &target_array);

  // Insert copy operator.
  auto* copy_op = new TensorFlowReshapeOperator;
  copy_op->inputs = {
      source_array_name,
      CreateInt32Array(
          model, AvailableArrayName(*model, target_array_name + "_copy_shape"),
          shape)};
  copy_op->outputs = {target_array_name};
  if (target_array.has_shape()) {
    copy_op->shape = target_array.shape().dims();
  }
  model->operators.emplace_back(copy_op);
}

void CloneArray(Model* model, const std::string& source_array_name,
                const std::string& target_array_name) {
  CHECK(!model->HasArray(target_array_name));
  const Array& source_array = model->GetArray(source_array_name);
  Array& target_array = model->GetOrCreateArray(target_array_name);
  CopyArrayAttribs(source_array, &target_array);

  if (!source_array.buffer) {
    return;
  }

  switch (source_array.data_type) {
    case ArrayDataType::kBool:
      CopyArrayBuffer<ArrayDataType::kBool>(source_array, &target_array);
      break;
    case ArrayDataType::kFloat:
      CopyArrayBuffer<ArrayDataType::kFloat>(source_array, &target_array);
      break;
    case ArrayDataType::kInt8:
      CopyArrayBuffer<ArrayDataType::kInt8>(source_array, &target_array);
      break;
    case ArrayDataType::kUint8:
      CopyArrayBuffer<ArrayDataType::kUint8>(source_array, &target_array);
      break;
    case ArrayDataType::kInt16:
      CopyArrayBuffer<ArrayDataType::kInt16>(source_array, &target_array);
      break;
    case ArrayDataType::kUint16:
      CopyArrayBuffer<ArrayDataType::kUint16>(source_array, &target_array);
      break;
    case ArrayDataType::kInt32:
      CopyArrayBuffer<ArrayDataType::kInt32>(source_array, &target_array);
      break;
    case ArrayDataType::kUint32:
      CopyArrayBuffer<ArrayDataType::kUint32>(source_array, &target_array);
      break;
    case ArrayDataType::kInt64:
      CopyArrayBuffer<ArrayDataType::kInt64>(source_array, &target_array);
      break;
    case ArrayDataType::kUint64:
      CopyArrayBuffer<ArrayDataType::kUint64>(source_array, &target_array);
      break;
    case ArrayDataType::kString:
      CopyArrayBuffer<ArrayDataType::kString>(source_array, &target_array);
      break;
    case ArrayDataType::kComplex64:
      CopyArrayBuffer<ArrayDataType::kComplex64>(source_array, &target_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type: "
                 << ArrayDataTypeName(source_array.data_type);
      return;
  }
}

void MakeArrayDims(int num_dims, int batch, int height, int width, int depth,
                   std::vector<int>* out_dims) {
  CHECK(out_dims->empty());
  if (num_dims == 0) {
    return;
  } else if (num_dims == 1) {
    CHECK_EQ(batch, 1);
    *out_dims = {depth};
  } else if (num_dims == 2) {
    *out_dims = {batch, depth};
  } else if (num_dims == 3) {
    CHECK_EQ(batch, 1);
    *out_dims = {height, width, depth};
  } else if (num_dims == 4) {
    *out_dims = {batch, height, width, depth};
  } else {
    LOG(FATAL) << "Should not get here: " << num_dims;
  }
}

void CreateOrCheckRnnStateArray(const std::string& name, int size,
                                int state_num_dims, Model* model) {
  int batch = 1;
  int num_dims = -1;
  if (state_num_dims > 0) {
    num_dims = state_num_dims;
  } else {
    // state_num_dims is not given. We will infer it from an input tensor.
    for (const auto& input_array : model->flags.input_arrays()) {
      // Pick 'num_dims' and 'batch' from the first input_arrays, unless we find
      // a better match by name.
      if (input_array.name() == name || num_dims == -1) {
        num_dims = input_array.shape().dims_size();
        if (num_dims > 0) {
          batch = input_array.shape().dims(0);
        }
      }
    }
  }
  Array& array = model->GetOrCreateArray(name);
  if (array.has_shape()) {
    num_dims = array.shape().dimensions_count();
  }
  if (!array.has_shape() && num_dims >= 0) {
    Shape* shape = array.mutable_shape();
    std::vector<int> dims;
    MakeArrayDims(num_dims, batch, 1, 1, size, &dims);
    *shape->mutable_dims() = dims;
  }
}

void ResolveModelFlags(const ModelFlags& model_flags, Model* model) {
  // Merge info about input_arrays from model_flags into model->flags
  for (const auto& specified_input_array : model_flags.input_arrays()) {
    toco::InputArray* dst_input_array = nullptr;
    for (int i = 0; i < model->flags.input_arrays_size(); i++) {
      toco::InputArray* candidate_dst_input_array =
          model->flags.mutable_input_arrays(i);
      if (candidate_dst_input_array->name() == specified_input_array.name()) {
        // specified_input_array from model_flags maps to dst_input_array
        // in model->flags
        dst_input_array = candidate_dst_input_array;
        break;
      }
    }
    if (!dst_input_array) {
      // Specified_input_array from model_flags is not found in model->flags.
      // Match a name-less specified input array when there can be no ambiguity
      // as there is only 1 input array.
      if (model->flags.input_arrays_size() == 1 &&
          model_flags.input_arrays_size() == 1 &&
          !specified_input_array.has_name()) {
        dst_input_array = model->flags.mutable_input_arrays(0);
      }
    }
    if (!dst_input_array) {
      // Still no match, so create a new input array to copy
      // specified_input_array into.
      dst_input_array = model->flags.add_input_arrays();
      dst_input_array->set_name(specified_input_array.name());
    }

#define RESOLVE_MODEL_FLAG(field_name)                                       \
  if (specified_input_array.has_##field_name()) {                            \
    if (dst_input_array->has_##field_name()) {                               \
      QCHECK_EQ(dst_input_array->field_name(),                               \
                specified_input_array.field_name())                          \
          << "For input array '" << dst_input_array->name() << "', "         \
          << "specified " #field_name " flag with value: "                   \
          << specified_input_array.field_name()                              \
          << " does not agree with already defined " #field_name             \
             " of this model, with value: "                                  \
          << specified_input_array.field_name();                             \
    } else {                                                                 \
      dst_input_array->set_##field_name(specified_input_array.field_name()); \
    }                                                                        \
  }
    RESOLVE_MODEL_FLAG(std_value);
    RESOLVE_MODEL_FLAG(mean_value);
#undef RESOLVE_MODEL_FLAG

    if (specified_input_array.has_shape()) {
      if (dst_input_array->has_shape()) {
        QCHECK_EQ(specified_input_array.shape().dims_size(),
                  dst_input_array->shape().dims_size())
            << "For input array '" << specified_input_array.name() << "', "
            << "size of specified input shape flag with size: "
            << specified_input_array.shape().dims_size()
            << " does not agree with already defined input shape"
               " of this model, with size: "
            << dst_input_array->shape().dims_size();
        // We treat the first dimension as a special case, since it is often
        // a batch size and the input_shape flag is effectively overriding
        // the model.
        for (int i = 1; i < specified_input_array.shape().dims_size(); i++) {
          QCHECK_EQ(specified_input_array.shape().dims(i),
                    dst_input_array->shape().dims(i))
              << "At dimension number " << i << " of input array "
              << specified_input_array.name() << ", the specified shape's "
              << "dimension flag with dimension: "
              << specified_input_array.shape().dims(i)
              << " does not agree with already defined shape"
              << " of this model, with dimension: "
              << dst_input_array->shape().dims(i);
        }
      } else {
        *dst_input_array->mutable_shape() = specified_input_array.shape();
      }
    }

    if (specified_input_array.has_data_type()) {
      QCHECK(!dst_input_array->has_data_type());
      dst_input_array->set_data_type(specified_input_array.data_type());
    }
  }

  if (model_flags.output_arrays_size() > 0) {
    model->flags.mutable_output_arrays()->CopyFrom(model_flags.output_arrays());
  }

#define RESOLVE_MODEL_FLAG(name)                                           \
  if (model_flags.has_##name()) {                                          \
    if (model->flags.has_##name()) {                                       \
      QCHECK_EQ(model_flags.name(), model->flags.name())                   \
          << "Specified " #name " flag with value: " << model_flags.name() \
          << " does not agree with already defined " #name                 \
             " of this model, with value: "                                \
          << model->flags.name();                                          \
    } else {                                                               \
      model->flags.set_##name(model_flags.name());                         \
    }                                                                      \
  }

  RESOLVE_MODEL_FLAG(variable_batch)

#undef RESOLVE_MODEL_FLAG

  if (!model_flags.rnn_states().empty()) {
    model->flags.mutable_rnn_states()->CopyFrom(model_flags.rnn_states());
  }

  if (model->flags.model_checks_size() == 0) {
    model->flags.mutable_model_checks()->CopyFrom(model_flags.model_checks());
  }

  QCHECK_GT(model->flags.output_arrays_size(), 0)
      << "This model does not define output arrays, so a "
         "--output_arrays flag must be given on the command-line.";

  for (auto& input_array_proto : *model->flags.mutable_input_arrays()) {
    auto& input_array = model->GetOrCreateArray(input_array_proto.name());
    if (input_array_proto.has_data_type()) {
      const ArrayDataType specified_type =
          ConvertIODataTypeToArrayDataType(input_array_proto.data_type());
      QCHECK(specified_type != ArrayDataType::kNone);
      if (input_array.data_type != ArrayDataType::kNone) {
        QCHECK(specified_type == input_array.data_type)
            << "For input array " << input_array_proto.name()
            << " the specified input data type "
            << IODataType_Name(input_array_proto.data_type())
            << " conflicts with the existing type.";
      }
      input_array.data_type = specified_type;
    }

    if (input_array.data_type == ArrayDataType::kNone) {
      // We start out with a float input array;
      // that may get replaced by a uint8 array later, by
      // MakeInitialDequantizeOp.
      input_array.data_type = ArrayDataType::kFloat;
    }

    // Compare/merge the model->flags describing the input_shape with
    // the actual input array's shape.
    if (!input_array.has_shape()) {
      if (input_array_proto.has_shape()) {
        auto& input_array_dims = *input_array.mutable_shape()->mutable_dims();
        CheckValidShapeDimensions(input_array_proto.shape().dims());
        for (const auto& dim : input_array_proto.shape().dims()) {
          input_array_dims.push_back(dim);
        }
      }
    } else {
      if (input_array_proto.has_shape()) {
        // If an input shape was specified on the flags ensure that it matches
        // the actual shape in the model.
        const auto& input_array_dims =
            *input_array.mutable_shape()->mutable_dims();
        CHECK_EQ(input_array_dims.size(),
                 input_array_proto.shape().dims_size());
        for (int i = 0; i < input_array_dims.size(); i++) {
          CHECK_EQ(input_array_dims[i], input_array_proto.shape().dims(i));
        }
      } else {
        for (int i = 0; i < input_array.shape().dimensions_count(); i++) {
          input_array_proto.mutable_shape()->add_dims(
              input_array.shape().dims(i));
        }
      }
    }

    const float mean_value = input_array_proto.mean_value();
    const float std_value = input_array_proto.std_value();
    MinMax input_minmax;
    float qmin = 0, qmax = 255;
    if (input_array.data_type == ArrayDataType::kInt16) {
      qmin = -32768;
      qmax = 32767;
    }
    input_minmax.min = (qmin - mean_value) / std_value;
    input_minmax.max = (qmax - mean_value) / std_value;
    if (!input_array.minmax) {
      input_array.GetOrCreateMinMax() = input_minmax;
    }
  }

  // Creation of the RNN state arrays
  for (const auto& rnn_state : model->flags.rnn_states()) {
    CreateOrCheckRnnStateArray(rnn_state.state_array(), rnn_state.size(),
                               rnn_state.num_dims(), model);
  }

  model->flags.set_change_concat_input_ranges(
      model_flags.change_concat_input_ranges());
  model->flags.set_allow_nonascii_arrays(model_flags.allow_nonascii_arrays());
  model->flags.set_allow_nonexistent_arrays(
      model_flags.allow_nonexistent_arrays());

  CHECK(!model->flags.has_arrays_extra_info());
  *model->flags.mutable_arrays_extra_info() = model_flags.arrays_extra_info();
}

void CheckIsReadyForQuantization(const Model& model) {
  for (const auto& op : model.operators) {
    for (const auto& input : op->inputs) {
      const auto& input_array = model.GetArray(input);
      if (input_array.data_type != ArrayDataType::kFloat) {
        // The array is not floats, no quantization needed.
        continue;
      }
      if (input_array.minmax) {
        // The array has minmax, we're good.
        continue;
      }
      if (input_array.buffer) {
        // The array has a constant buffer, so we can
        // fall back to computing the minmax from actual array entries
        // (with a WARNING about possible accuracy implications).
        continue;
      }
      LOG(FATAL)
          << "Array " << input << ", which is an input to the "
          << HelpfulOperatorTypeName(*op) << " operator producing the output "
          << "array " << op->outputs[0] << ", is lacking min/max data, "
          << "which is necessary for quantization. If accuracy matters, either "
          << "target a non-quantized output format, or run quantized training "
          << "with your model from a floating point checkpoint to change the "
          << "input graph to contain min/max information. If you don't care "
          << "about accuracy, you can pass --default_ranges_min= and "
          << "--default_ranges_max= for easy experimentation.";
    }
  }
}

int ElementSize(ArrayDataType data_type) {
  switch (data_type) {
    case ArrayDataType::kBool:
      return sizeof(bool);
    case ArrayDataType::kFloat:
      return 4;
    case ArrayDataType::kInt8:
      return 1;
    case ArrayDataType::kUint8:
      return 1;
    case ArrayDataType::kInt16:
      return 2;
    case ArrayDataType::kUint16:
      return 2;
    case ArrayDataType::kInt32:
      return 4;
    case ArrayDataType::kUint32:
      return 4;
    case ArrayDataType::kInt64:
      return 8;
    case ArrayDataType::kUint64:
      return 8;
    case ArrayDataType::kComplex64:
      return 8;
    case ArrayDataType::kComplex128:
      return 16;
    case ArrayDataType::kFloat64:
      return 8;

    // Usually not critical limitation because strings are only input and/or
    // output.
    case ArrayDataType::kString:
      LOG(FATAL) << "Transient arrays with strings are not supported yet";
      return 0;
    default:
      LOG(FATAL) << "Unknown data_type = " << static_cast<int>(data_type);
      return 0;
  }
}

void DropMinMax(Model* model, const std::string& array_name) {
  auto& array = model->GetArray(array_name);
  if (!!array.minmax) {
    LOG(WARNING) << "Dropping MinMax information in array " << array_name
                 << ". Expect inaccuracy in quantized inference.";
    array.minmax = nullptr;
  }
}

bool IsAllocatableTransientArray(const Model& model,
                                 const std::string& array_name) {
  // Optional array is not transient
  if (model.IsOptionalArray(array_name)) return false;
  // The model's input and output arrays are externally allocated.
  // They are not transient arrays.
  if (IsInputArray(model, array_name) || IsOutputArray(model, array_name)) {
    return false;
  }
  const auto& array = &model.GetArray(array_name);
  // An array with a constant buffer isn't a transient array.
  if (!!array->buffer) {
    return false;
  }
  // An array without shape isn't allocatable.
  if (!array->has_shape()) {
    return false;
  }

  // The size of string tensors is rarely known ahead of time, so all transient
  // tensors of this type will need to be dynamically allocated.
  if (array->final_data_type == ArrayDataType::kString ||
      array->data_type == ArrayDataType::kString) {
    return false;
  }

  return true;
}

std::string AvailableArrayName(const Model& model, const std::string& name) {
  std::string sanitized_name = SanitizeNameForTFNode(name);
  if (!model.HasArray(sanitized_name) &&
      !model.IsOptionalArray(sanitized_name)) {
    return sanitized_name;
  }
  const int kNumSuffixesToTry = 1000;
  for (int i = 0; i < kNumSuffixesToTry; i++) {
    const std::string& name_with_suffix =
        toco::port::StringF("%s_%d", sanitized_name, i);
    if (!model.HasArray(name_with_suffix) &&
        !model.IsOptionalArray(name_with_suffix)) {
      return name_with_suffix;
    }
  }
  LOG(FATAL) << "Could not find an available array name starting with "
             << sanitized_name << ". Tried " << kNumSuffixesToTry
             << " suffixes, all were taken!";
  return "";
}

std::string ShapeToString(const Shape& shape) {
  if (shape.dimensions_count() == 0) {
    return "[]";
  }

  return absl::StrCat("[ ", absl::StrJoin(shape.dims(), ", "), " ]");
}

void PrintArrayShape(Model* model, const std::string& name) {
  if (!model->GetArray(name).has_shape()) {
    LOG(INFO) << name << " has no shape";
    return;
  }
  LOG(INFO) << name
            << " has shape: " << ShapeToString(model->GetArray(name).shape());
}

bool IsArrayFullyConnectedWeights(const Model& model, const std::string& name) {
  bool is_fc_weights = false;
  bool is_something_else = false;
  for (const auto& op : model.operators) {
    for (int input_index = 0; input_index < op->inputs.size(); input_index++) {
      if (op->inputs[input_index] == name) {
        if (op->type == OperatorType::kFullyConnected && input_index == 1) {
          is_fc_weights = true;
        } else {
          is_something_else = true;
        }
      }
    }
  }
  CHECK(!(is_fc_weights && is_something_else));
  return is_fc_weights;
}

std::string CreateInt32Array(Model* model, const std::string& param_name,
                             const std::vector<int>& value) {
  auto param_array_name = AvailableArrayName(*model, param_name);
  auto& param_array = model->GetOrCreateArray(param_array_name);
  param_array.mutable_shape()->ReplaceDims({static_cast<int>(value.size())});
  param_array.data_type = ArrayDataType::kInt32;
  auto& param_array_data =
      param_array.GetMutableBuffer<ArrayDataType::kInt32>().data;
  param_array_data.resize(RequiredBufferSizeForShape(param_array.shape()));
  for (int i = 0; i < value.size(); ++i) {
    param_array_data[i] = value[i];
  }
  return param_array_name;
}

bool EstimateArithmeticOpsCount(const Model& model, const Operator& op,
                                int64_t* result) {
  switch (op.type) {
    case OperatorType::kFullyConnected:
    case OperatorType::kConv:
    case OperatorType::kDepthwiseConv: {
      const auto& output_array = model.GetArray(op.outputs[0]);
      const auto& weights_array = model.GetArray(op.inputs[1]);
      if (!output_array.has_shape() || !weights_array.has_shape()) {
        return false;
      }
      int64_t cols = 1;
      for (int i = 0; i < output_array.shape().dimensions_count() - 1; i++) {
        cols *= output_array.shape().dims(i);
      }
      const int64_t cost_per_col =
          2 * RequiredBufferSizeForShape(weights_array.shape());
      *result = cost_per_col * cols;
      if (op.inputs.size() > 2) {
        // There is a bias vector. One more op per output value.
        *result += RequiredBufferSizeForShape(output_array.shape());
      }
      break;
    }
    case OperatorType::kTransposeConv: {
      const auto& input_array = model.GetArray(op.inputs[2]);
      const auto& weights_array = model.GetArray(op.inputs[1]);
      if (!input_array.has_shape() || !weights_array.has_shape()) {
        return false;
      }
      const Shape& input = input_array.shape();
      const Shape& weights = weights_array.shape();
      // Compute op count from the seven nested loops of
      // tflite::reference_ops::TransposeConv():
      *result = 2 * input.dims(0) * input.dims(1) * input.dims(2) *
                input.dims(3) * weights.dims(1) * weights.dims(2) *
                weights.dims(0);
      // Note that tflite::optimized_ops::TransposeConv() uses an im2col matrix
      // and has a higher op count, by a factor of (output_height*output_width)
      // vs. (input_height*input_width). Yet it generally performs better
      // because of coherent memory access. (At least for 2x2 striding. But not
      // likely for all cases.)
      break;
    }
    case OperatorType::kAdd:
    case OperatorType::kSub:
    case OperatorType::kMul: {
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      *result = RequiredBufferSizeForShape(output_array.shape());
      break;
    }
    case OperatorType::kAddN: {
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      // AddN cost is roughly the same cost as N-1 Adds.
      const int64_t num_adds = op.inputs.size() - 1;
      *result = num_adds * RequiredBufferSizeForShape(output_array.shape());
      break;
    }
    case OperatorType::kLogistic:
    case OperatorType::kSoftmax:
    case OperatorType::kLogSoftmax:
    case OperatorType::kTanh: {
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      // As a very rough ballpark, the cost of evaluating a math function
      // such as tanh or logistic is about 32 multiplications, and about as
      // many additions/subtractions. (Just a power-of-two order-of-magnitude
      // from looking at actual implementations that we use in runtime/ code).
      *result = 64 * RequiredBufferSizeForShape(output_array.shape());
      break;
    }
    case OperatorType::kMaxPool: {
      const auto& maxpool = *static_cast<const MaxPoolOperator*>(&op);
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      *result = RequiredBufferSizeForShape(output_array.shape()) *
                maxpool.kheight * maxpool.kwidth;
      break;
    }
    case OperatorType::kAveragePool: {
      const auto& avgpool = *static_cast<const AveragePoolOperator*>(&op);
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      *result = RequiredBufferSizeForShape(output_array.shape()) *
                avgpool.kheight * avgpool.kwidth;
      break;
    }
    case OperatorType::kL2Pool: {
      const auto* maxpool = static_cast<const MaxPoolOperator*>(&op);
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      // The sum of squares requires (kheight*kwidth) multiply-adds,
      // and then there is the sqrt which we ballpark at 32 ops.
      const int64_t cost_per_val = 2 * maxpool->kheight * maxpool->kwidth + 32;
      *result = RequiredBufferSizeForShape(output_array.shape()) * cost_per_val;
      break;
    }
    case OperatorType::kL2Normalization: {
      const auto& output_array = model.GetArray(op.outputs[0]);
      if (!output_array.has_shape()) {
        return false;
      }
      // Computing the squared L2 norm is N multiply-adds so 2N ops,
      // then the single inverse-sqrt is negligible, then we multiply each
      // value by the resulting multiplier, so an extra N ops. count 3N ops.
      *result = 3 * RequiredBufferSizeForShape(output_array.shape());
      break;
    }
    default:
      *result = 0;
      break;
  }
  return true;
}

bool EstimateArithmeticOpsCount(const Model& model, int64_t* result) {
  int64_t total = 0;
  for (const auto& op : model.operators) {
    int64_t num_ops;
    if (!EstimateArithmeticOpsCount(model, *op, &num_ops)) {
      return false;
    }
    total += num_ops;
  }
  *result = total;
  return true;
}

std::string FormattedNumber(int64_t x) {
  const int64_t million = 1000000;
  const int64_t billion = 1000000000;
  if (x < 10000) {
    return toco::port::StringF("%d ", x);
  } else if (x < billion) {
    return toco::port::StringF("%.3f M", static_cast<double>(x) / million);
  } else {
    return toco::port::StringF("%.3f G", static_cast<double>(x) / billion);
  }
}

void GetShuffleShape(AxesOrder input_axes_order, AxesOrder output_axes_order,
                     std::vector<int>* shuffle) {
  CHECK_EQ(AxesCount(input_axes_order), AxesCount(output_axes_order));
  shuffle->resize(4);
  for (int i = 0; i < 4; i++) {
    (*shuffle)[i] = i;
  }
  if (input_axes_order == output_axes_order) {
    // nothing to do
  } else if (AxesCount(input_axes_order) == 2) {
    shuffle->resize(2);
    (*shuffle)[0] = 1;
    (*shuffle)[1] = 0;
  } else if (input_axes_order == AxesOrder::kOHWI &&
             output_axes_order == AxesOrder::kHWIO) {
    // 3210 <- 3210
    // HWIO <- OHWI
    *shuffle = {1, 2, 3, 0};
  } else if (input_axes_order == AxesOrder::kHWIO &&
             output_axes_order == AxesOrder::kOHWI) {
    // 3210 <- 3210
    // OHWI <- HWIO
    *shuffle = {3, 0, 1, 2};
  } else if (input_axes_order == AxesOrder::kOHWI &&
             output_axes_order == AxesOrder::kHWOI) {
    *shuffle = {1, 2, 0, 3};
  } else {
    LOG(FATAL) << "Bad shuffle";
  }
}

void ExtendShuffle(const std::vector<int>& input_shuffle, int newdim,
                   std::vector<int>* extended_shuffle) {
  *extended_shuffle = input_shuffle;
  CHECK(newdim >= input_shuffle.size());
  const int pad_size = newdim - input_shuffle.size();
  extended_shuffle->resize(newdim);
  for (int i = 0; i < pad_size; i++) {
    (*extended_shuffle)[i] = i;
  }
  for (int i = pad_size; i < newdim; i++) {
    (*extended_shuffle)[i] = input_shuffle[i - pad_size] + pad_size;
  }
}

void ShuffleDims(const Shape& input_shape, AxesOrder input_axes_order,
                 AxesOrder output_axes_order, Shape* output_shape) {
  if (input_axes_order == AxesOrder::kHWIM &&
      output_axes_order == AxesOrder::k1HWO) {
    // This special case isn't just a permutation, the IM pair of dims get
    // merged into the 3 dim, so we have to special-case it.
    *output_shape = Shape({1, input_shape.dims(0), input_shape.dims(1),
                           input_shape.dims(3) * input_shape.dims(2)});
  } else {
    std::vector<int> shuffle;
    GetShuffleShape(input_axes_order, output_axes_order, &shuffle);
    std::vector<int>* output_dims = output_shape->mutable_dims();
    output_dims->resize(input_shape.dimensions_count());
    for (int i = 0; i < input_shape.dimensions_count(); i++) {
      (*output_dims)[i] = input_shape.dims(shuffle[i]);
    }
  }
}

template <typename T>
void ShuffleArrayTemplate(const Shape& input_shape, AxesOrder input_axes_order,
                          AxesOrder output_axes_order,
                          const Shape& output_shape, const T* input_data,
                          T* output_data) {
  if (input_axes_order == AxesOrder::kHWIM &&
      output_axes_order == AxesOrder::k1HWO) {
    // This special case isn't just a permutation, the IM pair of dims get
    // merged into the O dim, so we have to special-case it. Fortunately,
    // as far as array shuffling is concerned, it's just the identity
    // transformation.
    memcpy(output_data, input_data,
           RequiredBufferSizeForShape(input_shape) * sizeof(output_data[0]));
    return;
  }
  CHECK(input_shape.dimensions_count() == output_shape.dimensions_count());
  const int dim = input_shape.dimensions_count();
  CHECK_LE(dim, 4);
  std::vector<int> shuffle;
  GetShuffleShape(input_axes_order, output_axes_order, &shuffle);
  CHECK(shuffle.size() >= dim);
  for (int i = 0; i < dim; i++) {
    CHECK(shuffle[i] >= 0 && shuffle[i] < dim);
    CHECK(input_shape.dims(shuffle[i]) == output_shape.dims(i));
  }
  Shape extended_input_shape = input_shape;
  ExtendShape(&extended_input_shape, 4);
  Shape extended_output_shape = output_shape;
  ExtendShape(&extended_output_shape, 4);
  std::vector<int> extended_shuffle;
  ExtendShuffle(shuffle, 4, &extended_shuffle);

  const std::vector<int>& extended_input_dims = extended_input_shape.dims();
  const std::vector<int>& extended_output_dims = extended_output_shape.dims();

  // TODO(starka): Rework to handle different numbers of dimensions.
  int input_strides[4];
  input_strides[3] = 1;
  input_strides[2] = extended_input_dims[3];
  input_strides[1] = input_strides[2] * extended_input_dims[2];
  input_strides[0] = input_strides[1] * extended_input_dims[1];
  const int input_stride_0 = input_strides[extended_shuffle[3]];
  const int input_stride_1 = input_strides[extended_shuffle[2]];
  const int input_stride_2 = input_strides[extended_shuffle[1]];
  const int input_stride_3 = input_strides[extended_shuffle[0]];

  const int output_size_0 = extended_output_dims[3];
  const int output_size_1 = extended_output_dims[2];
  const int output_size_2 = extended_output_dims[1];
  const int output_size_3 = extended_output_dims[0];
  const int output_stride_0 = 1;
  const int output_stride_1 = output_size_0;
  const int output_stride_2 = output_stride_1 * output_size_1;
  const int output_stride_3 = output_stride_2 * output_size_2;

  for (int i3 = 0; i3 < output_size_3; i3++) {
    const T* const input_ptr_3 = input_data + i3 * input_stride_3;
    T* const output_ptr_3 = output_data + i3 * output_stride_3;
    for (int i2 = 0; i2 < output_size_2; i2++) {
      const T* const input_ptr_2 = input_ptr_3 + i2 * input_stride_2;
      T* const output_ptr_2 = output_ptr_3 + i2 * output_stride_2;
      for (int i1 = 0; i1 < output_size_1; i1++) {
        const T* input_ptr = input_ptr_2 + i1 * input_stride_1;
        T* output_ptr = output_ptr_2 + i1 * output_stride_1;
        T* const output_ptr_end = output_ptr + output_size_0 * output_stride_0;
        while (output_ptr != output_ptr_end) {
          *output_ptr = *input_ptr;
          input_ptr += input_stride_0;
          output_ptr += output_stride_0;
        }
      }
    }
  }
}

void ShuffleArray(const Shape& input_shape, AxesOrder input_axes_order,
                  AxesOrder output_axes_order, const Shape& output_shape,
                  const uint8* input_data, uint8* output_data) {
  ShuffleArrayTemplate<uint8>(input_shape, input_axes_order, output_axes_order,
                              output_shape, input_data, output_data);
}

void ShuffleArray(const Shape& input_shape, AxesOrder input_axes_order,
                  AxesOrder output_axes_order, const Shape& output_shape,
                  const float* input_data, float* output_data) {
  ShuffleArrayTemplate<float>(input_shape, input_axes_order, output_axes_order,
                              output_shape, input_data, output_data);
}

int AxesCount(AxesOrder axes_order) {
  switch (axes_order) {
    case AxesOrder::kOneAxis:
      return 1;
    case AxesOrder::kRC:
      return 2;
    case AxesOrder::kCR:
      return 2;
    case AxesOrder::kHWIO:
      return 4;
    case AxesOrder::kOHWI:
      return 4;
    case AxesOrder::kHWIM:
      return 4;
    case AxesOrder::k1HWO:
      return 4;
    case AxesOrder::kNHWC:
      return 4;
    case AxesOrder::kHWOI:
      return 4;
    default:
      LOG(FATAL) << "Bad AxesOrder";
      return 0;
  }
}

bool IsDiscardableArray(const Model& model, const std::string& array_name) {
  if (IsInputArray(model, array_name) || IsOutputArray(model, array_name)) {
    return false;
  }
  for (const auto& rnn_state : model.flags.rnn_states()) {
    if (!rnn_state.discardable()) {
      if (array_name == rnn_state.state_array()) {
        return false;
      }
      if (array_name == rnn_state.back_edge_source_array()) {
        return false;
      }
    }
  }
  return true;
}

bool ReshapeIsEquivalentToTranspose(const Model& model,
                                    const TensorFlowReshapeOperator* op,
                                    bool allow_extra_unary_dims) {
  CHECK(!op->shape.empty());
  CHECK(model.HasArray(op->inputs[0]));
  CHECK(model.HasArray(op->outputs[0]));

  const auto& input_array = model.GetArray(op->inputs[0]);
  const auto& output_array = model.GetArray(op->outputs[0]);

  CHECK(input_array.has_shape());
  CHECK(output_array.has_shape());

  std::vector<int> in_shape = input_array.shape().dims();
  std::vector<int> out_shape = output_array.shape().dims();

  // If the reshape changes the number of dimensions so it cannot be interpreted
  // as a transpose.
  if (!allow_extra_unary_dims && in_shape.size() != out_shape.size()) {
    return false;
  }

  in_shape.erase(std::remove(in_shape.begin(), in_shape.end(), 1),
                 in_shape.end());
  out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1),
                  out_shape.end());
  return in_shape == out_shape;
}

void CheckFinalDataTypesSatisfied(const Model& model) {
  for (const auto& array_entry : model.GetArrayMap()) {
    const auto& array = *array_entry.second;
    if (array.data_type == ArrayDataType::kBool) {
      // Boolean values are never quantized.
      continue;
    }

    // If the final data type is int16, the data type may be float, for example
    // after dequantization.
    if (array.final_data_type != ArrayDataType::kNone &&
        array.final_data_type != ArrayDataType::kInt16) {
      CHECK(array.data_type == array.final_data_type)
          << "Array \"" << array_entry.first
          << "\" has mis-matching actual and final data types (data_type="
          << ArrayDataTypeName(array.data_type)
          << ", final_data_type=" << ArrayDataTypeName(array.final_data_type)
          << ").";
    }
  }
}

ArrayDataType ConvertIODataTypeToArrayDataType(IODataType type) {
  switch (type) {
    case FLOAT:
      return ArrayDataType::kFloat;
    case UINT8:
    case QUANTIZED_UINT8:
      return ArrayDataType::kUint8;
    case INT8:
    case QUANTIZED_INT8:
      return ArrayDataType::kInt8;
    case INT16:
    case QUANTIZED_INT16:
      return ArrayDataType::kInt16;
    case INT32:
      return ArrayDataType::kInt32;
    case UINT32:
      return ArrayDataType::kUint32;
    case INT64:
      return ArrayDataType::kInt64;
    case UINT64:
      return ArrayDataType::kUint64;
    case BOOL:
      return ArrayDataType::kBool;
    case STRING:
      return ArrayDataType::kString;
    case COMPLEX64:
      return ArrayDataType::kComplex64;
    case COMPLEX128:
      return ArrayDataType::kComplex128;
    case FLOAT16:
      return ArrayDataType::kFloat16;
    case FLOAT64:
      return ArrayDataType::kFloat64;
    case RESOURCE:
    case VARIANT:
    default:
      return ArrayDataType::kNone;
  }
}

void FinishBuildingRNNStates(Model* model) {
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (!model->HasArray(rnn_state.back_edge_source_array()) ||
        !model->HasArray(rnn_state.state_array())) {
      CHECK(model->HasArray(rnn_state.back_edge_source_array()));
      CHECK(model->HasArray(rnn_state.state_array()));
      continue;
    }
    const auto& src_array = model->GetArray(rnn_state.back_edge_source_array());
    auto& dst_array = model->GetArray(rnn_state.state_array());
    if (src_array.data_type == ArrayDataType::kNone &&
        dst_array.data_type == ArrayDataType::kNone) {
      dst_array.data_type = ArrayDataType::kFloat;
    }
  }
}

// Returns the array names that match the ArraysExtraInfo's name and
// name_regexp. The regexp match is for a full match.
std::unordered_set<std::string> ScanArrayNames(
    const Model& model, const toco::ArraysExtraInfo_Entry& entry) {
  std::unordered_set<std::string> matches;
  if (model.HasArray(entry.name())) {
    matches.insert(entry.name());
  }
  if (!entry.name_regexp().empty()) {
    const auto& arrays = model.GetArrayMap();
    const RE2 name_regexp = {entry.name_regexp()};
    for (auto it = arrays.begin(); it != arrays.end(); ++it) {
      if (RE2::FullMatch(it->first, name_regexp)) {
        matches.insert(it->first);
      }
    }
  }
  return matches;
}

void UseArraysExtraInfo(Model* model, bool quantize_output) {
  for (const auto& entry : model->flags.arrays_extra_info().entries()) {
    const auto matches = ScanArrayNames(*model, entry);
    if (matches.empty()) {
      LOG(ERROR) << "arrays_extra_info_file: No matching arrays found for "
                 << (entry.has_name() ? entry.name() : "")
                 << (entry.has_name_regexp() ? entry.name_regexp() : "");
      continue;
    }
    for (const auto& matched_name : matches) {
      auto& array = model->GetArray(matched_name);
      if (entry.has_min() || entry.has_max()) {
        CHECK_EQ(entry.has_min(), entry.has_max());
        auto& minmax = array.GetOrCreateMinMax();
        minmax.min = entry.min();
        minmax.max = entry.max();
      }
      if (entry.has_data_type() && quantize_output) {
        array.final_data_type =
            ConvertIODataTypeToArrayDataType(entry.data_type());
      }
      if (entry.has_shape()) {
        array.clear_shape();
        // Make sure to create the shape even if there are no dims, to
        // correctly record 0-D shapes.
        array.mutable_shape();
        for (const auto& dim : entry.shape().dims()) {
          array.mutable_shape()->mutable_dims()->push_back(dim);
        }
      }
      if (entry.has_constant_float_value()) {
        CHECK(array.has_shape());
        if (array.data_type == ArrayDataType::kFloat) {
          auto& data = array.GetMutableBuffer<ArrayDataType::kFloat>().data;
          data.resize(RequiredBufferSizeForShape(array.shape()));
          for (float& f : data) {
            f = entry.constant_float_value();
          }
        }
      }
    }
  }
}

void UndoWeightsShuffling(Model* model) {
  for (const auto& op : model->operators) {
    if (op->type != toco::OperatorType::kFullyConnected) {
      continue;
    }
    const auto& fc_op = static_cast<toco::FullyConnectedOperator&>(*op);
    if (fc_op.weights_format == FullyConnectedWeightsFormat::kDefault) {
      continue;
    }
    const std::string& weights_name = fc_op.inputs[1];
    QCHECK_EQ(CountOpsWithInput(*model, weights_name), 1);
    auto& weights_array = model->GetArray(weights_name);
    QCHECK(weights_array.data_type == ArrayDataType::kUint8);
    auto& weights_data =
        weights_array.GetMutableBuffer<toco::ArrayDataType::kUint8>().data;
    const auto& weights_shape = weights_array.shape();
    QCHECK_EQ(weights_shape.dimensions_count(), 2);
    const int rows = weights_shape.dims(0);
    const int cols = weights_shape.dims(1);
    QCHECK_EQ(rows % 4, 0);
    QCHECK_EQ(cols % 16, 0);
    CHECK_EQ(rows * cols, weights_data.size());
    // Compute the de-shuffled weights
    std::vector<uint8> deshuffled_data(weights_data.size());
    uint8* shuffled_data_ptr = weights_data.data();
    for (int r = 0; r < rows; r += 4) {
      for (int c = 0; c < cols; c += 16) {
        for (int i = 0; i < 4; i++) {
          uint8* deshuffled_data_ptr =
              deshuffled_data.data() + (r + i) * cols + c;
          for (int j = 0; j < 16; j++) {
            uint8 shuffled_val = *shuffled_data_ptr++;
            // Deshuffling isn't only about deshuffling the storage layout,
            // it's also about undoing the flipping of the sign bit, which is
            // performed on the shuffled weights.
            uint8 deshuffled_val = shuffled_val ^ 0x80;
            *deshuffled_data_ptr++ = deshuffled_val;
          }
        }
      }
    }
    CHECK_EQ(shuffled_data_ptr, weights_data.data() + rows * cols);
    // Switch this FC op to using the deshuffled weights.
    weights_data = std::move(deshuffled_data);
  }
}

void CopyMinMaxAndQuantizationRelatedFields(const Array& src, Array* dst) {
  if (src.minmax) {
    dst->GetOrCreateMinMax() = src.GetMinMax();
  }
  if (src.quantization_params) {
    dst->GetOrCreateQuantizationParams() = src.GetQuantizationParams();
  }
  dst->narrow_range = src.narrow_range;
}

}  // namespace toco
