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

#include <deque>
#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

const char kSuffix[] = "LayoutOptimizer";
const char kPermNHWCToNCHW[] = "PermConstNHWCToNCHW";
const char kPermNCHWToNHWC[] = "PermConstNCHWToNHWC";
const char kTransposeNHWCToNCHW[] = "TransposeNHWCToNCHW";
const char kTransposeNCHWToNHWC[] = "TransposeNCHWToNHWC";
const char kDimMapNHWCToNCHW[] = "DimMapNHWCToNCHW";
const char kDimMapNCHWToNHWC[] = "DimMapNCHWToNHWC";
const char kVecPermuteNHWCToNCHW[] = "VecPermuteNHWCToNCHW";
const char kVecPermuteNCHWToNHWC[] = "VecPermuteNCHWToNHWC";
const char kReshapeNHWCToNCHW[] = "ReshapeNHWCToNCHW";
const char kReshapeConst[] = "ReshapeConst";

std::set<string> GetOpsFormatSupported() {
  std::set<string> ops_format_supported = {
      "AvgPool",
      "AvgPoolGrad",
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "BiasAdd",
      "BiasAddGrad",
      "DepthwiseConv2dNative",
      "DepthwiseConv2dNativeBackpropInput",
      "DepthwiseConv2dNativeBackpropFilter",
      "FusedBatchNorm",
      "FusedBatchNormV2",
      "FusedBatchNormGrad",
      "FusedBatchNormGradV2",
      "FusedConv2DBiasActivation",
      "MaxPool",
      "MaxPoolV2",
      "MaxPoolGrad",
      "MaxPoolGradGrad",
      "MaxPoolGradV2",
      "MaxPoolGradGradV2",
      "SpaceToDepth",
      "DepthToSpace"};
  return ops_format_supported;
}

std::set<string> GetOpsFormatAgnostic() {
  std::set<string> ops_format_agnostic = {"Abs",
                                          "Add",
                                          "AddN",
                                          "AddV2",
                                          "Acos",
                                          "Acosh",
                                          "All",
                                          "Angle",
                                          "Any",
                                          "ApproximateEqual",
                                          "Asin",
                                          "Asinh",
                                          "Atan",
                                          "Atan2",
                                          "Atanh",
                                          "Betainc",
                                          "Bitcast",
                                          "Cast",
                                          "Ceil",
                                          "CheckNumerics",
                                          "Complex",
                                          "ComplexAbs",
                                          "Concat",
                                          "ConcatV2",
                                          "Conj",
                                          "Cos",
                                          "Cosh",
                                          "Digamma",
                                          "Div",
                                          "Elu",
                                          "EluGrad",
                                          "Enter",
                                          "Equal",
                                          "Erf",
                                          "Erfc",
                                          "Exit",
                                          "Exp",
                                          "Expm1",
                                          "Fill",
                                          "Floor",
                                          "FloorDiv",
                                          "FloorMod",
                                          "Greater",
                                          "GreaterEqual",
                                          "GuaranteeConst",
                                          "HistogramSummary",
                                          "Identity",
                                          "IdentityN",
                                          "Igamma",
                                          "Igammac",
                                          "Imag",
                                          "Inv",
                                          "InvGrad",
                                          "IsFinite",
                                          "IsInf",
                                          "IsNan",
                                          "Less",
                                          "LessEqual",
                                          "Lgamma",
                                          "Log",
                                          "LogicalAnd",
                                          "LogicalNot",
                                          "LogicalOr",
                                          "Log1p",
                                          "Max",
                                          "Maximum",
                                          "Mean",
                                          "Merge",
                                          "Min",
                                          "Minimum",
                                          "Mod",
                                          "Mul",
                                          "Neg",
                                          "NextIteration",
                                          "NotEqual",
                                          "OnesLike",
                                          "Pad",
                                          "PreventGradient",
                                          "Prod",
                                          "Polygamma",
                                          "Pow",
                                          "Real",
                                          "RealDiv",
                                          "Reciprocal",
                                          "ReciprocalGrad",
                                          "Relu",
                                          "Relu6",
                                          "Relu6Grad",
                                          "ReluGrad",
                                          "Rint",
                                          "Select",
                                          "Selu",
                                          "SeluGrad",
                                          "Shape",
                                          "ShapeN",
                                          "Sigmoid",
                                          "SigmoidGrad",
                                          "Sign",
                                          "Sin",
                                          "Sinh",
                                          "Slice",
                                          "Snapshot",
                                          "Softplus",
                                          "SoftplusGrad",
                                          "Split",
                                          "SplitV",
                                          "StridedSlice",
                                          "StridedSliceGrad",
                                          "Switch",
                                          "Tile",
                                          "TruncateDiv",
                                          "TruncateMod",
                                          "ReverseV2",
                                          "Round",
                                          "Rsqrt",
                                          "RsqrtGrad",
                                          "Sqrt",
                                          "SqrtGrad",
                                          "Square",
                                          "SquaredDifference",
                                          "Squeeze",
                                          "StopGradient",
                                          "Sub",
                                          "Sum",
                                          "Tan",
                                          "Tanh",
                                          "TanhGrad",
                                          "ZerosLike",
                                          "Zeta"};
  return ops_format_agnostic;
}

bool EndWith(const string& str, const string& ending) {
  if (str.size() < ending.size()) return false;
  if (str.substr(str.size() - ending.size(), ending.size()) == ending)
    return true;
  return false;
}

bool IsNodeByLayoutOptimizer(const string& node_name) {
  const string suffix = kSuffix;
  return EndWith(node_name, suffix);
}

bool IsNodeType(const string& node_name, const string& type) {
  const string suffix = strings::StrCat(type, "-", kSuffix);
  return EndWith(node_name, suffix);
}

bool IsTransposeNHWCToNCHW(const string& node_name) {
  return IsNodeType(node_name, kTransposeNHWCToNCHW);
}

bool IsTransposeNCHWToNHWC(const string& node_name) {
  return IsNodeType(node_name, kTransposeNCHWToNHWC);
}

bool IsDimMapNHWCToNCHW(const string& node_name) {
  return IsNodeType(node_name, kDimMapNHWCToNCHW);
}

bool IsDimMapNCHWToNHWC(const string& node_name) {
  return IsNodeType(node_name, kDimMapNCHWToNHWC);
}

bool IsVecPermuteNHWCToNCHW(const string& node_name) {
  return IsNodeType(node_name, kVecPermuteNHWCToNCHW);
}

bool IsVecPermuteNCHWToNHWC(const string& node_name) {
  return IsNodeType(node_name, kVecPermuteNCHWToNHWC);
}

bool IsConcat(const NodeDef& node) {
  const auto op = node.op();
  return op == "Concat" || op == "ConcatV2";
}

bool IsConcatV1(const NodeDef& node) {
  const auto op = node.op();
  return op == "Concat";
}

bool IsMaxPoolV2(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPoolV2";
}

bool IsMaxPoolGradV1(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPoolGrad";
}

bool IsMaxPoolGradV2(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPoolGradV2";
}

bool IsMaxPoolGradGradV1(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPoolGradGrad";
}

bool IsMaxPoolGradGradV2(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPoolGradGradV2";
}

bool IsUnaryGrad(const NodeDef& node) {
  bool is_unary_grad =
      IsEluGrad(node) || IsInvGrad(node) || IsReciprocalGrad(node) ||
      IsRelu6Grad(node) || IsReluGrad(node) || IsRsqrtGrad(node) ||
      IsSeluGrad(node) || IsSigmoidGrad(node) || IsSoftplusGrad(node) ||
      IsSoftsignGrad(node) || IsSqrtGrad(node) || IsTanhGrad(node);
  return is_unary_grad;
}

bool IsComparisonOp(const NodeDef& node) {
  bool is_compare = IsApproximateEqual(node) || IsEqual(node) ||
                    IsGreater(node) || IsGreaterEqual(node) || IsLess(node) ||
                    IsLessEqual(node) || IsNotEqual(node);
  return is_compare;
}

bool IsReduceOp(const NodeDef& node) {
  return IsSum(node) || IsMean(node) || IsProd(node) || IsMax(node) ||
         IsMin(node) || IsAll(node) || IsAny(node);
}

bool IsBinaryOp(const NodeDef& node) {
  bool is_binary =
      IsAdd(node) || IsAtan2(node) || IsComparisonOp(node) || IsComplex(node) ||
      IsDiv(node) || IsFloorDiv(node) || IsIgamma(node) || IsIgammac(node) ||
      IsLogicalAnd(node) || IsLogicalOr(node) || IsMaximum(node) ||
      IsMinimum(node) || IsMod(node) || IsMul(node) || IsPolygamma(node) ||
      IsPow(node) || IsRealDiv(node) || IsSquaredDifference(node) ||
      IsSub(node) || IsTruncateDiv(node) || IsTruncateMod(node) || IsZeta(node);
  return is_binary;
}

std::vector<int> NonControlInputs(const NodeDef& node) {
  std::vector<int> pos;
  for (int i = 0; i < node.input_size(); i++) {
    if (!IsControlInput(node.input(i))) {
      pos.push_back(i);
    }
  }
  return pos;
}

std::vector<int> DataInputPosConcat(const NodeDef& node) {
  int n = node.attr().at("N").i();
  std::vector<int> input_pos;
  int start = (IsConcatV1(node)) ? 1 : 0;
  int end = start + n;
  for (int i = start; i < end; i++) {
    input_pos.push_back(i);
  }
  return input_pos;
}

std::vector<int> DataInputPos(const NodeDef& node) {
  if (IsSplit(node) || IsHistogramSummary(node)) {
    return {1};
  }
  if (IsStridedSliceGrad(node)) {
    return {4};
  }
  if (IsBinaryOp(node) || IsUnaryGrad(node)) {
    return {0, 1};
  }
  if (IsBetainc(node) || IsSelect(node)) {
    return {0, 1, 2};
  }
  if (IsShapeN(node) || IsIdentityN(node) || IsAddN(node) || IsMerge(node)) {
    return NonControlInputs(node);
  }
  if (IsConcat(node)) {
    return DataInputPosConcat(node);
  }
  if (node.input_size() > 0 && !IsControlInput(node.input(0))) {
    return {0};
  }
  return {};
}

bool IsHostMemory(const NodeDef& node, int output_port) {
  DeviceNameUtils::ParsedName parsed_name;
  if (DeviceNameUtils::ParseFullName(node.device(), &parsed_name)) {
    DeviceType device_type(parsed_name.type);
    Status s = FindKernelDef(device_type, node, nullptr, nullptr);
    if (s.ok()) {
      tensorflow::MemoryTypeVector in_mtypes;
      tensorflow::MemoryTypeVector out_mtypes;
      s = tensorflow::MemoryTypesForNode(OpRegistry::Global(), device_type,
                                         node, &in_mtypes, &out_mtypes);
      if (s.ok()) {
        if (out_mtypes[output_port] == HOST_MEMORY) {
          return true;
        }
      }
    } else {
      return true;
    }
  }
  return false;
}

class GraphProcessor {
 public:
  GraphProcessor(const GraphProperties& graph_properties,
                 const VirtualPlacer& virtual_placer,
                 const std::unordered_set<string>& nodes_to_preserve,
                 GraphDef* graph, NodeMap* node_map)
      : graph_properties_(graph_properties),
        virtual_placer_(virtual_placer),
        nodes_to_preserve_(nodes_to_preserve),
        graph_(graph),
        node_map_(node_map) {}

 protected:
  NodeDef* AddNodePermConst(const string& name, const string& device,
                            const std::vector<int>& permutation) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(name, node);
    node->set_name(name);
    node->set_op("Const");
    AttrValue attr_data_type;
    attr_data_type.set_type(DT_INT32);
    node->mutable_attr()->insert({"dtype", attr_data_type});
    AttrValue attr_tensor;
    Tensor tensor(DT_INT32, TensorShape({4}));
    for (int i = 0; static_cast<size_t>(i) < permutation.size(); i++) {
      tensor.flat<int>()(i) = permutation[i];
    }
    tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
    node->mutable_attr()->insert({"value", attr_tensor});
    string device_name;
    if (device.empty()) {
      device_name = virtual_placer_.get_canonical_device_name(*node);
    } else {
      device_name = device;
    }
    node->set_device(device_name);
    return node;
  }

  NodeDef* AddNodeConstScalar(const string& name, const string& device,
                              DataType dtype, int value) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(name, node);
    node->set_name(name);
    node->set_op("Const");
    AttrValue attr_data_type;
    attr_data_type.set_type(dtype);
    node->mutable_attr()->insert({"dtype", attr_data_type});
    AttrValue attr_tensor;
    Tensor tensor(dtype, TensorShape({}));
    tensor.scalar<int>()() = value;
    tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
    node->mutable_attr()->insert({"value", attr_tensor});
    string device_name;
    if (device.empty()) {
      device_name = virtual_placer_.get_canonical_device_name(*node);
    } else {
      device_name = device;
    }
    node->set_device(device_name);
    return node;
  }

  string LayoutOptimizerNode(const string& base_name) {
    return strings::StrCat(base_name, "-", kSuffix);
  }

  const GraphProperties& graph_properties_;
  const VirtualPlacer& virtual_placer_;
  const std::unordered_set<string>& nodes_to_preserve_;
  GraphDef* graph_;
  NodeMap* node_map_;
};

struct OptimizeContext {
  OptimizeContext(GraphDef* graph, NodeDef* node, NodeMap* node_map,
                  const GraphProperties& graph_properties,
                  const VirtualPlacer& virtual_placer,
                  const std::unordered_set<string>& nodes_to_preserve,
                  bool is_in_frame)
      : graph(graph),
        node(node),
        node_map(node_map),
        graph_properties(graph_properties),
        virtual_placer(virtual_placer),
        nodes_to_preserve(nodes_to_preserve),
        is_in_frame(is_in_frame) {}
  GraphDef* graph;
  NodeDef* node;
  NodeMap* node_map;
  const GraphProperties& graph_properties;
  const VirtualPlacer& virtual_placer;
  const std::unordered_set<string>& nodes_to_preserve;
  bool is_in_frame;
};

class NodeProcessor : public GraphProcessor {
 public:
  explicit NodeProcessor(const OptimizeContext& opt_cxt)
      : GraphProcessor(opt_cxt.graph_properties, opt_cxt.virtual_placer,
                       opt_cxt.nodes_to_preserve, opt_cxt.graph,
                       opt_cxt.node_map),
        node_(opt_cxt.node),
        is_in_frame_(opt_cxt.is_in_frame) {}
  virtual ~NodeProcessor() {}
  virtual Status ConvertNode() {
    if (ShouldProcess()) {
      UpdateAttrDataFormat();
      UpdateAttrKSize();
      UpdateAttrStrides();
      UpdateAttrDilations();
      UpdateAttrShape();
      TF_RETURN_IF_ERROR(AddLayoutTransposeToInputs());
      TF_RETURN_IF_ERROR(AddLayoutTransposeToOutputs());
      TF_RETURN_IF_ERROR(CustomizedProcessing());
    }
    return Status::OK();
  }

 protected:
  bool IsPortDimsN(const NodeDef& node, int port, int n) const {
    if (node.attr().find("_output_shapes") != node.attr().end()) {
      if (node.attr().at("_output_shapes").list().shape_size() > port) {
        auto shape = node.attr().at("_output_shapes").list().shape(port);
        if (shape.unknown_rank()) {
          return false;
        }
        if (shape.dim_size() == n) {
          return true;
        }
      }
    }
    return false;
  }

  bool IsPortZeroDimsN(const NodeDef& node, int n) const {
    return IsPortDimsN(node, 0, n);
  }

  bool IsPortZeroDimsFour(const NodeDef& node) const {
    return NodeProcessor::IsPortZeroDimsN(node, 4) ||
           IsTransposeNCHWToNHWC(node.name());
  }

  bool IsPortDimsFour(const NodeDef& node, int port) const {
    return NodeProcessor::IsPortDimsN(node, port, 4) ||
           IsTransposeNCHWToNHWC(node.name());
  }

  bool IsNHWC() const {
    if (node_->attr().find("data_format") != node_->attr().end()) {
      if (node_->attr().at("data_format").s().compare("NHWC") == 0) {
        return true;
      }
    }
    return false;
  }

  bool HasOutputs() const {
    auto outputs = node_map_->GetOutputs(node_->name());
    return !outputs.empty();
  }

  Status HasAttribute(const NodeDef& node, const string& attr) const {
    if (node.attr().find(attr) == node.attr().end()) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("Missing attribute ", attr));
    }
    return Status::OK();
  }

  bool MustPreserve() const {
    return nodes_to_preserve_.find(node_->name()) != nodes_to_preserve_.end();
  }

  bool IsOnGPU() const {
    string device_name;
    if (node_->device().empty()) {
      device_name = virtual_placer_.get_canonical_device_name(*node_);
    } else {
      device_name = node_->device();
    }
    string device;
    string not_used;
    if (DeviceNameUtils::SplitDeviceName(device_name, &not_used, &device) &&
        str_util::StrContains(str_util::Lowercase(device),
                              str_util::Lowercase(DEVICE_GPU))) {
      return true;
    }
    return false;
  }

  virtual bool ShouldProcess() const {
    return !MustPreserve() && IsNHWC() && IsPortZeroDimsFour(*node_) &&
           HasOutputs() && IsOnGPU();
  }

  virtual void UpdateAttrShape() {
    if (node_->attr().find("_output_shapes") != node_->attr().end()) {
      for (const auto& pos : GetOutputPos()) {
        auto shape = node_->mutable_attr()
                         ->at("_output_shapes")
                         .mutable_list()
                         ->mutable_shape(pos);
        if (shape->dim_size() == 4) {
          int64 h = shape->dim(1).size();
          int64 w = shape->dim(2).size();
          int64 c = shape->dim(3).size();
          shape->mutable_dim(1)->set_size(c);
          shape->mutable_dim(2)->set_size(h);
          shape->mutable_dim(3)->set_size(w);
        }
      }
    }
  }

  Status UpdateAttrValueOfInput(int input_index, bool permute) {
    auto input_node = node_map_->GetNode(node_->input(input_index));
    // We created a copy of the node, so that we don't modify the original node,
    // which might be used elsewhere. Note that this copy also copies the
    // control dependency input in the case this node is inside a loop,
    // to ensure added_node is in the same frame with node_.
    NodeDef* added_node = graph_->add_node();
    *added_node = *input_node;
    string base_name = strings::StrCat(node_->name(), "-", input_index);
    string node_name = LayoutOptimizerNode(base_name);
    added_node->set_name(node_name);
    *node_->mutable_input(input_index) = node_name;
    node_map_->AddNode(node_name, added_node);
    node_map_->AddOutput(node_name, node_->name());
    return UpdateAttrValue(added_node, permute);
  }

  virtual std::vector<int> GetInputPos() const { return {0}; }

  virtual std::set<int> GetOutputPos() const {
    // For most nodes, no need to process control nodes or nodes that use an
    // output other than the first output: only the first output is of
    // 4D NCHW/NHWC format and thus relevant here.
    std::set<int> output_pos = {0};
    return output_pos;
  }

  virtual Status AddLayoutTransposeToInputs() {
    std::vector<int> input_pos = GetInputPos();
    for (const auto& pos : input_pos) {
      string node_name = LayoutOptimizerNode(
          strings::StrCat(node_->name(), "-", pos, "-", kTransposeNHWCToNCHW));
      DataType dtype =
          graph_properties_.GetInputProperties(node_->name())[pos].dtype();
      auto input_node = node_map_->GetNode(node_->input(pos));
      TF_RETURN_IF_ERROR(HasAttribute(*input_node, "_output_shapes"));
      string const_name = GetOrAddNodePermNHWCToNCHW(pos);
      int output_pos;
      ParseNodeName(node_->input(pos), &output_pos);
      AddNodeTranspose(
          node_name, node_->input(pos), const_name, dtype,
          input_node->attr().at("_output_shapes").list().shape(output_pos),
          true);
      node_map_->UpdateOutput(NodeName(node_->input(pos)), node_->name(),
                              node_name);
      node_map_->AddOutput(node_name, node_->name());
      *node_->mutable_input(pos) = node_name;
    }
    return Status::OK();
  }

  Status AddTransformToOutputs(const string& op) {
    auto outputs = node_map_->GetOutputs(node_->name());
    string const_name = GetOrAddNodePermNCHWToNHWC();
    int output_count = 0;
    for (const auto& output : outputs) {
      int connections = 0;
      int connections_removed = 0;
      for (int i = 0; i < output->input_size(); i++) {
        auto& input = *output->mutable_input(i);
        int input_port;
        string input_name = ParseNodeName(input, &input_port);
        auto output_pos = GetOutputPos();
        if (input_name == node_->name()) {
          connections++;
          if (output_pos.find(input_port) != output_pos.end()) {
            connections_removed++;
            string added_node_base_name =
                strings::StrCat(node_->name(), "-", output_count, "-", i);
            string added_node_name;
            DataType dtype =
                graph_properties_.GetOutputProperties(node_->name())[input_port]
                    .dtype();
            if (op == "Transpose") {
              added_node_name = LayoutOptimizerNode(strings::StrCat(
                  added_node_base_name, "-", kTransposeNCHWToNHWC));
              TF_RETURN_IF_ERROR(HasAttribute(*node_, "_output_shapes"));
              AddNodeTranspose(
                  added_node_name, input, const_name, dtype,
                  node_->attr().at("_output_shapes").list().shape(input_port),
                  false);
            } else if (op == "DataFormatVecPermute") {
              added_node_name = LayoutOptimizerNode(strings::StrCat(
                  added_node_base_name, "-", kVecPermuteNCHWToNHWC));
              AddNodeDataFormatOp(added_node_name, input, op, dtype, false);
            } else {
              return errors::InvalidArgument("Unsupported op type: ", op);
            }
            input = added_node_name;
            node_map_->AddOutput(node_->name(), added_node_name);
            node_map_->AddOutput(added_node_name, output->name());
          }
        }
      }
      if (connections == connections_removed) {
        node_map_->RemoveOutput(node_->name(), output->name());
      }
      output_count++;
    }
    return Status::OK();
  }

  virtual Status AddLayoutTransposeToOutputs() {
    return AddTransformToOutputs("Transpose");
  }

  virtual Status CustomizedProcessing() { return Status::OK(); }

  Status UpdateOrTransformParamInput(int param_index, const string& op,
                                     DataType dtype) {
    auto param_node = node_map_->GetNode(node_->input(param_index));
    bool permute = (op == "DataFormatVecPermute") ? true : false;
    if (IsConstant(*param_node)) {
      TF_RETURN_IF_ERROR(UpdateAttrValueOfInput(param_index, permute));
    } else {
      AddDataFormatTranformToParamInput(op, param_index, dtype);
    }
    return Status::OK();
  }

  NodeDef* node_;
  bool is_in_frame_;

 private:
  void UpdateAttrKSize() {
    if (node_->attr().find("ksize") != node_->attr().end()) {
      auto list = node_->mutable_attr()->at("ksize").mutable_list();
      UpdateTuple(list);
    }
  }

  void UpdateAttrStrides() {
    if (node_->attr().find("strides") != node_->attr().end()) {
      auto list = node_->mutable_attr()->at("strides").mutable_list();
      UpdateTuple(list);
    }
  }

  void UpdateAttrDilations() {
    if (node_->attr().find("dilations") != node_->attr().end()) {
      auto list = node_->mutable_attr()->at("dilations").mutable_list();
      UpdateTuple(list);
    }
  }

  void UpdateAttrDataFormat() {
    if (node_->attr().find("data_format") != node_->attr().end()) {
      if (node_->attr().at("data_format").s().compare("NHWC") == 0) {
        string* data_format =
            node_->mutable_attr()->at("data_format").mutable_s();
        *data_format = "NCHW";
      }
    }
  }

  Status UpdateAttrValue(NodeDef* node, bool permute) {
    TF_RETURN_IF_ERROR(HasAttribute(*node, "value"));
    Tensor tensor;
    auto success =
        tensor.FromProto(node->mutable_attr()->at({"value"}).tensor());
    if (!success) {
      LOG(ERROR) << "Failed to parse TensorProto.";
    }

    if (permute) {
      if (tensor.dims() == 1) {
        if (tensor.flat<int>().size() == 4) {
          int c = tensor.flat<int>()(3);
          tensor.flat<int>()(3) = tensor.flat<int>()(2);
          tensor.flat<int>()(2) = tensor.flat<int>()(1);
          tensor.flat<int>()(1) = c;
        } else {
          return Status(error::INVALID_ARGUMENT,
                        strings::StrCat("Unsupported tensor size: ",
                                        tensor.flat<int>().size()));
        }
      } else if (tensor.dims() == 2) {
        for (int i = 0; i < 2; i++) {
          int c = tensor.matrix<int>()(3, i);
          tensor.matrix<int>()(3, i) = tensor.matrix<int>()(2, i);
          tensor.matrix<int>()(2, i) = tensor.matrix<int>()(1, i);
          tensor.matrix<int>()(1, i) = c;
        }
      } else {
        return Status(
            error::INVALID_ARGUMENT,
            strings::StrCat("Unsupported dimension size: ", tensor.dims()));
      }
    } else {
      for (int i = 0; i < tensor.flat<int>().size(); i++) {
        int value = tensor.flat<int>()(i);
        value = (value >= 0) ? value : value + 4;
        if (value == 1 || value == 2) {
          value = value + 1;
        } else if (value == 3) {
          value = 1;
        }
        tensor.flat<int>()(i) = value;
      }
    }

    if (tensor.dims() == 0) {
      tensor.AsProtoField(node->mutable_attr()->at({"value"}).mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(
          node->mutable_attr()->at({"value"}).mutable_tensor());
    }
    return Status::OK();
  }

  NodeDef* AddNodeTranspose(const string& node_name, const string& input_name,
                            const string& const_name, DataType data_type,
                            const TensorShapeProto& input_shape,
                            bool NHWCToNCHW) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(node_name, node);
    node->set_name(node_name);
    *node->add_input() = input_name;
    *node->add_input() = const_name;
    node->set_op("Transpose");
    node->set_device(node_->device());
    AttrValue attr_data_type;
    attr_data_type.set_type(data_type);
    node->mutable_attr()->insert({"T", attr_data_type});
    AttrValue attr_data_type_perm;
    attr_data_type_perm.set_type(DT_INT32);
    node->mutable_attr()->insert({"Tperm", attr_data_type_perm});
    if (!input_shape.unknown_rank()) {
      AttrValue attr_output_shape;
      auto output_shape = attr_output_shape.mutable_list()->add_shape();
      if (NHWCToNCHW) {
        output_shape->add_dim()->set_size(input_shape.dim(0).size());
        output_shape->add_dim()->set_size(input_shape.dim(3).size());
        output_shape->add_dim()->set_size(input_shape.dim(1).size());
        output_shape->add_dim()->set_size(input_shape.dim(2).size());
      } else {
        output_shape->add_dim()->set_size(input_shape.dim(0).size());
        output_shape->add_dim()->set_size(input_shape.dim(2).size());
        output_shape->add_dim()->set_size(input_shape.dim(3).size());
        output_shape->add_dim()->set_size(input_shape.dim(1).size());
      }
      node->mutable_attr()->insert({"_output_shapes", attr_output_shape});
    }
    return node;
  }

  NodeDef* AddNodePermNHWCToNCHW(const string& base_name,
                                 const string& depended_node,
                                 const string& device) {
    string name =
        LayoutOptimizerNode(strings::StrCat(base_name, "-", kPermNHWCToNCHW));
    auto const_node = AddNodePermConst(name, device, {0, 3, 1, 2});
    // This is to ensure the transpose node and the const node are in the
    // same frame.
    *const_node->add_input() = AsControlDependency(depended_node);
    return const_node;
  }

  NodeDef* AddNodePermNCHWToNHWC(const string& base_name,
                                 const string& depended_node,
                                 const string& device) {
    auto const_node = AddNodePermConst(
        LayoutOptimizerNode(strings::StrCat(base_name, "-", kPermNCHWToNHWC)),
        device, {0, 2, 3, 1});
    // This is to ensure the transpose node and the const node are in the same
    // frame.
    *const_node->add_input() = AsControlDependency(depended_node);
    return const_node;
  }

  string GetOrAddNodePermNHWCToNCHW(int pos) {
    string const_name;
    if (is_in_frame_) {
      string base_name = strings::StrCat(node_->name(), "-", pos);
      string input = NodeName(node_->input(pos));
      string depended_node;
      if (!IsTransposeNCHWToNHWC(input)) {
        depended_node = input;
      } else {
        auto input_node = node_map_->GetNode(input);
        depended_node = NodeName(input_node->input(0));
      }
      auto const_node =
          AddNodePermNHWCToNCHW(base_name, depended_node, node_->device());
      const_name = const_node->name();
    } else {
      const_name = LayoutOptimizerNode(kPermNHWCToNCHW);
    }
    return const_name;
  }

  string GetOrAddNodePermNCHWToNHWC() {
    string const_name;
    if (is_in_frame_) {
      auto const_node =
          AddNodePermNCHWToNHWC(node_->name(), node_->name(), node_->device());
      const_name = const_node->name();
    } else {
      const_name = LayoutOptimizerNode(kPermNCHWToNHWC);
    }
    return const_name;
  }

  void UpdateTuple(AttrValue_ListValue* list) {
    int64 h = list->i(1);
    int64 w = list->i(2);
    int64 c = list->i(3);
    list->set_i(1, c);
    list->set_i(2, h);
    list->set_i(3, w);
  }

  bool IsInputOnHost(const string& input_name) const {
    string device = node_->device();
    DeviceNameUtils::ParsedName parsed_name;
    if (DeviceNameUtils::ParseFullName(device, &parsed_name)) {
      if (parsed_name.type != "CPU") {
        NodeDef* input = node_map_->GetNode(input_name);
        int port;
        ParseNodeName(input_name, &port);
        if (IsHostMemory(*input, port)) {
          return true;
        }
      }
    }
    return false;
  }

  NodeDef* AddNodeDataFormatOp(const string& name, const string& input_name,
                               const string& op, DataType dtype,
                               bool nhwc_to_nchw) {
    NodeDef* added_node = graph_->add_node();
    added_node->set_name(name);
    added_node->set_op(op);
    node_map_->AddNode(added_node->name(), added_node);
    added_node->set_device(node_->device());
    // The inputs of a DataFormat op could be in host memory for ops such as
    // Reshape. In such cases, run the kernel on the host too.
    if (IsInputOnHost(input_name)) {
      AttrValue attr_kernel;
      attr_kernel.set_s("host");
      added_node->mutable_attr()->insert({"_kernel", attr_kernel});
    }
    AttrValue attr_data_type;
    attr_data_type.set_type(dtype);
    added_node->mutable_attr()->insert({"T", attr_data_type});
    string src_format = (nhwc_to_nchw) ? "NHWC" : "NCHW";
    string dst_format = (nhwc_to_nchw) ? "NCHW" : "NHWC";
    AttrValue attr_format;
    attr_format.set_s(src_format);
    added_node->mutable_attr()->insert({"src_format", attr_format});
    attr_format.set_s(dst_format);
    added_node->mutable_attr()->insert({"dst_format", attr_format});
    *added_node->add_input() = input_name;
    return added_node;
  }

  void AddDataFormatTranformToParamInput(const string& op, int input_pos,
                                         DataType dtype) {
    string suffix = (op == "DataFormatVecPermute") ? kVecPermuteNHWCToNCHW
                                                   : kDimMapNHWCToNCHW;
    string name = LayoutOptimizerNode(
        strings::StrCat(node_->name(), "-", input_pos, "-", suffix));
    auto added_node =
        AddNodeDataFormatOp(name, node_->input(input_pos), op, dtype, true);
    *node_->mutable_input(input_pos) = added_node->name();
    node_map_->UpdateOutput(NodeName(added_node->input(0)), node_->name(),
                            added_node->name());
    node_map_->AddOutput(added_node->name(), node_->name());
  }
};

class AvgPoolGradProcessor : public NodeProcessor {
 public:
  explicit AvgPoolGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override { return {1}; }
  Status CustomizedProcessing() override {
    return UpdateOrTransformParamInput(0, "DataFormatVecPermute", DT_INT32);
  }
};

class BiasAddGradProcessor : public NodeProcessor {
 public:
  explicit BiasAddGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    if (MustPreserve()) {
      return false;
    }
    if (!IsOnGPU()) {
      return false;
    }
    auto input = node_map_->GetNode(node_->input(0));
    if (input) {
      int port;
      ParseNodeName(node_->input(0), &port);
      if (IsNHWC() && IsPortDimsFour(*input, port)) {
        return true;
      }
    }
    return false;
  }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }
};

class Conv2DProcessor : public NodeProcessor {
 public:
  Conv2DProcessor(const OptimizeContext& opt_cxt, bool no_gemm)
      : NodeProcessor(opt_cxt), no_gemm_(no_gemm) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsNHWC() && IsPortZeroDimsFour(*node_) &&
           HasOutputs() && (!IsGemmUsed() || no_gemm_) && IsOnGPU();
  }

  TensorShapeProto GetShape(const string& input_name) const {
    string node_name;
    int output_pos;
    node_name = ParseNodeName(input_name, &output_pos);
    NodeDef* node = node_map_->GetNode(node_name);
    if (node->attr().find("_output_shapes") != node->attr().end()) {
      return node->attr().at("_output_shapes").list().shape(output_pos);
    }
    TensorShapeProto shape;
    return shape;
  }

  bool IsStrideOne() const {
    if (node_->attr().find("strides") != node_->attr().end()) {
      auto list = node_->attr().at("strides").list();
      return list.i(1) == 1 && list.i(2) == 1;
    }
    return false;
  }

  bool IsValidPadding() const {
    if (node_->attr().find("padding") != node_->attr().end()) {
      auto padding = node_->attr().at("padding").s();
      return padding == "VALID";
    }
    return false;
  }

  // The logic inside this function is based on the internal implementation of
  // Conv2D, Conv2DBackpropInput, and Conv2DBackpropFilter ops, and thus
  // needs to be updated accordingly if the internal implementation changes.
  bool IsGemmUsed(const TensorShapeProto& filter_shape,
                  const TensorShapeProto& input_shape) const {
    if (filter_shape.dim_size() == 4) {
      if (filter_shape.dim(0).size() == 1 && filter_shape.dim(1).size() == 1 &&
          IsStrideOne()) {
        return true;
      }
    }
    if (input_shape.dim_size() == 4 && filter_shape.dim_size() == 4) {
      if (input_shape.dim(1).size() == filter_shape.dim(0).size() &&
          input_shape.dim(2).size() == filter_shape.dim(1).size() &&
          IsValidPadding()) {
        return true;
      }
    }
    return false;
  }

  virtual bool IsGemmUsed() const {
    auto filter_shape = GetShape(node_->input(1));
    auto input_shape = GetShape(node_->input(0));
    return IsGemmUsed(filter_shape, input_shape);
  }

  bool no_gemm_;
};

class Conv2DBackpropFilterProcessor : public Conv2DProcessor {
 public:
  Conv2DBackpropFilterProcessor(const OptimizeContext& opt_cxt, bool no_gemm)
      : Conv2DProcessor(opt_cxt, no_gemm) {}

 protected:
  bool IsGemmUsed() const override {
    auto filter_shape = GetShape(node_->name());
    auto input_shape = GetShape(node_->input(0));
    return Conv2DProcessor::IsGemmUsed(filter_shape, input_shape);
  }

  std::vector<int> GetInputPos() const override { return {0, 2}; }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }
  // No need to update output shape, as it is always of shape
  // [filter_height, filter_width, in_channels, out_channels], regardless of
  // whether NCHW or NHWC is used.
  void UpdateAttrShape() override {}
};

class Conv2DBackpropInputProcessor : public Conv2DProcessor {
 public:
  Conv2DBackpropInputProcessor(const OptimizeContext& opt_cxt, bool no_gemm)
      : Conv2DProcessor(opt_cxt, no_gemm) {}

 protected:
  bool IsGemmUsed() const override {
    auto filter_shape = GetShape(node_->input(1));
    auto input_shape = GetShape(node_->name());
    return Conv2DProcessor::IsGemmUsed(filter_shape, input_shape);
  }

  std::vector<int> GetInputPos() const override { return {2}; }

  Status CustomizedProcessing() override {
    return UpdateOrTransformParamInput(0, "DataFormatVecPermute", DT_INT32);
  }
};

class FusedBatchNormGradProcessor : public NodeProcessor {
 public:
  explicit FusedBatchNormGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return NodeProcessor::ShouldProcess() && IsTraining();
  }

  std::vector<int> GetInputPos() const override { return {0, 1}; }

 private:
  bool IsTraining() const {
    if (node_->attr().find("is_training") != node_->attr().end()) {
      if (node_->attr().at("is_training").b()) {
        return true;
      }
    }
    return false;
  }
};

class MaxPoolGradProcessor : public NodeProcessor {
 public:
  explicit MaxPoolGradProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override { return {0, 1, 2}; }
};

class MaxPoolGradV2Processor : public MaxPoolGradProcessor {
 public:
  explicit MaxPoolGradV2Processor(const OptimizeContext& opt_cxt)
      : MaxPoolGradProcessor(opt_cxt) {}

 protected:
  Status CustomizedProcessing() override {
    for (int i = 3; i <= 4; i++) {
      TF_RETURN_IF_ERROR(
          UpdateOrTransformParamInput(i, "DataFormatVecPermute", DT_INT32));
    }
    return Status::OK();
  }
};

class MaxPoolV2Processor : public NodeProcessor {
 public:
  explicit MaxPoolV2Processor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    // We check data_input's shape instead, because the shape inference of
    // MaxPoolV2 is not able to infer the shape when ksize or strides is not
    // constant.
    auto data_input = node_map_->GetNode(node_->input(0));
    int port;
    ParseNodeName(node_->input(0), &port);
    return !MustPreserve() && IsNHWC() && IsPortDimsFour(*data_input, port) &&
           HasOutputs() && IsOnGPU();
  }

  Status CustomizedProcessing() override {
    for (int i = 1; i <= 2; i++) {
      TF_RETURN_IF_ERROR(
          UpdateOrTransformParamInput(i, "DataFormatVecPermute", DT_INT32));
    }
    return Status::OK();
  }
};

class AgnosticNodeProcessor : public NodeProcessor {
 public:
  explicit AgnosticNodeProcessor(const OptimizeContext& opt_cxt)
      : NodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsPortZeroDimsFour(*node_) && HasOutputs() &&
           IsNodeAfterNCHWToNHWC() && IsOnGPU();
  }

  bool IsNodeAfterNCHWToNHWC(const NodeDef& node) const {
    std::set<string> ops_format_agnostic = GetOpsFormatAgnostic();
    std::deque<NodeDef*> queue;
    auto data_node_pos = DataInputPos(node);
    std::unordered_set<string> visited;
    for (const auto& pos : data_node_pos) {
      auto input_node = node_map_->GetNode(node.input(pos));
      queue.push_back(input_node);
      visited.insert(input_node->name());
    }
    // The code will exit this while loop in one iteration in most cases, as the
    // graph is already topologically sorted.
    while (!queue.empty()) {
      NodeDef* current_node = queue.front();
      queue.pop_front();
      if (IsTransposeNCHWToNHWC(current_node->name()) ||
          IsDimMapNCHWToNHWC(current_node->name()) ||
          IsVecPermuteNCHWToNHWC(current_node->name())) {
        return true;
      }
      // We only continue searching if the path is connected through
      // format-agnostic nodes.
      if (ops_format_agnostic.find(current_node->op()) !=
          ops_format_agnostic.end()) {
        auto current_node_pos = DataInputPos(*current_node);
        for (const auto& pos : current_node_pos) {
          auto input_node = node_map_->GetNode(current_node->input(pos));
          if (visited.find(input_node->name()) == visited.end()) {
            queue.push_back(input_node);
            visited.insert(input_node->name());
          }
        }
      }
    }
    return false;
  }

  bool IsNodeAfterNCHWToNHWC() const { return IsNodeAfterNCHWToNHWC(*node_); }
};

class AddNProcessor : public AgnosticNodeProcessor {
 public:
  explicit AddNProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override {
    return NonControlInputs(*node_);
  }
};

class BinaryOpProcessor : public AgnosticNodeProcessor {
 public:
  explicit BinaryOpProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsPortZeroDimsFour(*node_) && HasOutputs() &&
           IsNodeAfterNCHWToNHWC() &&
           (IsNDOperateWithMD(4, 0) || IsNDOperateWithMD(4, 1) ||
            IsNDOperateWithMD(4, 4) || IsNDOperateWithMD(0, 4) ||
            IsNDOperateWithMD(1, 4)) &&
           IsOnGPU();
  }

  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos;
    auto input0 = node_map_->GetNode(node_->input(0));
    auto input1 = node_map_->GetNode(node_->input(1));
    int input0_port;
    ParseNodeName(node_->input(0), &input0_port);
    int input1_port;
    ParseNodeName(node_->input(1), &input1_port);
    if (IsPortDimsFour(*input0, input0_port)) {
      input_pos.push_back(0);
    }
    if (IsPortDimsFour(*input1, input1_port)) {
      input_pos.push_back(1);
    }
    return input_pos;
  }

  bool IsNDOperateWithMD(int n, int m) const {
    auto input0 = node_map_->GetNode(node_->input(0));
    auto input1 = node_map_->GetNode(node_->input(1));
    int input0_port;
    ParseNodeName(node_->input(0), &input0_port);
    int input1_port;
    ParseNodeName(node_->input(1), &input1_port);

    if (input0 && input1) {
      bool input0_is_n = (n == 4) ? IsPortDimsFour(*input0, input0_port)
                                  : IsPortDimsN(*input0, input0_port, n);
      bool input1_is_m = (m == 4) ? IsPortDimsFour(*input1, input1_port)
                                  : IsPortDimsN(*input1, input1_port, m);
      return input0_is_n && input1_is_m;
    }
    return false;
  }

  NodeDef* AddNodeShapeConst(const string& name, int num_channels,
                             const string& depended_node) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(name, node);
    node->set_name(name);
    node->set_op("Const");
    node->set_device(node_->device());
    AttrValue attr_data_type;
    attr_data_type.set_type(DT_INT32);
    node->mutable_attr()->insert({"dtype", attr_data_type});

    AttrValue attr_tensor;
    Tensor tensor(DT_INT32, TensorShape({4}));
    std::vector<int> shape = {1, num_channels, 1, 1};
    for (int i = 0; i < static_cast<int>(shape.size()); i++) {
      tensor.flat<int>()(i) = shape[i];
    }
    tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
    node->mutable_attr()->insert({"value", attr_tensor});
    if (is_in_frame_) {
      // This is to ensure the transpose node and the const node are in the
      // same frame.
      *node->add_input() = AsControlDependency(depended_node);
    }
    return node;
  }

  NodeDef* AddNodeReshape(const string& node_name, const string& input_name,
                          const string& shape_const_node_name,
                          DataType data_type) {
    NodeDef* node = graph_->add_node();
    node_map_->AddNode(node_name, node);
    node->set_name(node_name);
    *node->add_input() = input_name;
    *node->add_input() = shape_const_node_name;
    node->set_op("Reshape");
    node->set_device(node_->device());

    AttrValue attr_type_indices;
    attr_type_indices.set_type(DT_INT32);
    node->mutable_attr()->insert({"Tshape", attr_type_indices});

    AttrValue attr_type_params;
    attr_type_params.set_type(data_type);
    node->mutable_attr()->insert({"T", attr_type_params});
    return node;
  }

  Status CustomizedProcessing() override {
    int vector_index = -1;
    if (IsNDOperateWithMD(4, 1)) {
      vector_index = 1;
    } else if (IsNDOperateWithMD(1, 4)) {
      vector_index = 0;
    }
    if (vector_index != -1) {
      string base_name = strings::StrCat(node_->name(), "-", vector_index);
      string reshape_node_name = LayoutOptimizerNode(
          strings::StrCat(base_name, "-", kReshapeNHWCToNCHW));
      string shape_const_node_name =
          LayoutOptimizerNode(strings::StrCat(base_name, "-", kReshapeConst));
      auto input_node = node_map_->GetNode(node_->input(vector_index));
      TF_RETURN_IF_ERROR(HasAttribute(*input_node, "_output_shapes"));
      int port;
      ParseNodeName(node_->input(vector_index), &port);
      int vector_size = input_node->attr()
                            .at("_output_shapes")
                            .list()
                            .shape(port)
                            .dim(0)
                            .size();
      AddNodeShapeConst(shape_const_node_name, vector_size,
                        NodeName(node_->input(vector_index)));
      TF_RETURN_IF_ERROR(HasAttribute(*node_, "T"));
      AddNodeReshape(reshape_node_name, node_->input(vector_index),
                     shape_const_node_name, node_->attr().at("T").type());
      node_map_->AddOutput(shape_const_node_name, reshape_node_name);
      node_map_->UpdateOutput(NodeName(node_->input(vector_index)),
                              node_->name(), reshape_node_name);
      node_map_->AddOutput(reshape_node_name, node_->name());
      *node_->mutable_input(vector_index) = reshape_node_name;
    }
    return Status::OK();
  }
};

class ConcatProcessor : public AgnosticNodeProcessor {
 public:
  explicit ConcatProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {
    // For Concat,  the concat axis is the first input; for ConcatV2,
    // the last input. Note that if with control inputs, the number of inputs
    // is larger than the integer attribute N.
    int n = node_->attr().at("N").i();
    axis_node_pos_ = (IsConcatV1(*node_)) ? 0 : n;
  }

 protected:
  std::vector<int> GetInputPos() const override {
    return DataInputPosConcat(*node_);
  }

  Status CustomizedProcessing() override {
    DataType dtype =
        (IsConcatV1(*node_)) ? DT_INT32 : node_->attr().at("Tidx").type();
    return UpdateOrTransformParamInput(axis_node_pos_, "DataFormatDimMap",
                                       dtype);
  }

  int axis_node_pos_;
};

class FillProcessor : public AgnosticNodeProcessor {
 public:
  explicit FillProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override { return {}; }

  Status CustomizedProcessing() override {
    DataType dtype = node_->attr().at("index_type").type();
    return UpdateOrTransformParamInput(0, "DataFormatVecPermute", dtype);
  }
};

class HistogramSummaryProcessor : public AgnosticNodeProcessor {
 public:
  explicit HistogramSummaryProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    auto input1 = node_map_->GetNode(node_->input(1));
    int port;
    ParseNodeName(node_->input(1), &port);
    return !MustPreserve() && HasOutputs() && IsNodeAfterNCHWToNHWC() &&
           IsPortDimsFour(*input1, port) && IsOnGPU();
  }

  std::vector<int> GetInputPos() const override { return {1}; }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }
};

class IdentityNProcessor : public AgnosticNodeProcessor {
 public:
  explicit IdentityNProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {
    std::set<string> ops_format_agnostic = GetOpsFormatAgnostic();
    for (int i = 0; i < node_->input_size(); i++) {
      auto input = node_map_->GetNode(node_->input(i));
      int port;
      ParseNodeName(node_->input(i), &port);
      // Skip control input.
      if (port != -1) {
        bool is_agnostic =
            ops_format_agnostic.find(input->op()) != ops_format_agnostic.end();
        if (IsPortDimsFour(*input, port) &&
            ((IsNodeAfterNCHWToNHWC(*input) && is_agnostic) ||
             IsTransposeNCHWToNHWC(input->name()))) {
          input_pos_.push_back(i);
        }
      }
    }
  }

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && HasOutputs() && IsNodeAfterNCHWToNHWC() &&
           IsOnGPU();
  }

  std::vector<int> GetInputPos() const override { return input_pos_; }

  std::set<int> GetOutputPos() const override {
    std::set<int> output_pos{};
    for (const auto& input_pos : input_pos_) {
      output_pos.insert(input_pos);
    }
    return output_pos;
  }

 private:
  std::vector<int> input_pos_;
};

class ShapeProcessor : public IdentityNProcessor {
 public:
  explicit ShapeProcessor(const OptimizeContext& opt_cxt)
      : IdentityNProcessor(opt_cxt) {}

 protected:
  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }

  Status CustomizedProcessing() override {
    return AddTransformToOutputs("DataFormatVecPermute");
  }
};

class MergeProcessor : public AgnosticNodeProcessor {
 public:
  explicit MergeProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    return !MustPreserve() && IsPortZeroDimsFour(*node_) && HasOutputs() &&
           IsEveryInputAfterNCHWToNHWC() && IsOnGPU();
  }

  std::vector<int> GetInputPos() const override {
    std::vector<int> input_pos;
    int n = node_->attr().at("N").i();
    input_pos.reserve(n);
    for (int i = 0; i < n; i++) {
      input_pos.push_back(i);
    }
    return input_pos;
  }

 private:
  bool IsEveryInputAfterNCHWToNHWC() const {
    std::set<string> ops_format_agnostic = GetOpsFormatAgnostic();
    for (const auto& input : node_->input()) {
      auto input_node = node_map_->GetNode(input);
      int port;
      ParseNodeName(input, &port);
      bool is_agnostic = ops_format_agnostic.find(input_node->op()) !=
                         ops_format_agnostic.end();
      if (IsPortDimsFour(*input_node, port) &&
          ((IsNodeAfterNCHWToNHWC(*input_node) && is_agnostic) ||
           IsTransposeNCHWToNHWC(input_node->name()))) {
        continue;
      }
      return false;
    }
    return true;
  }
};

class PadProcessor : public AgnosticNodeProcessor {
 public:
  explicit PadProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  Status CustomizedProcessing() override {
    DataType dtype = node_->attr().at("Tpaddings").type();
    return UpdateOrTransformParamInput(1, "DataFormatVecPermute", dtype);
  }
};

class ReverseProcessor : public AgnosticNodeProcessor {
 public:
  explicit ReverseProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  Status CustomizedProcessing() override {
    DataType dtype = node_->attr().at("Tidx").type();
    return UpdateOrTransformParamInput(1, "DataFormatDimMap", dtype);
  }
};

class SplitProcessor : public AgnosticNodeProcessor {
 public:
  explicit SplitProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {
    axis_node_pos_ = 0;
  }

 protected:
  std::vector<int> GetInputPos() const override { return {1}; }

  std::set<int> GetOutputPos() const override {
    std::set<int> output_pos{0};
    if (HasAttribute(*node_, "num_split").ok()) {
      for (int i = 1; i < node_->attr().at("num_split").i(); i++) {
        output_pos.insert(i);
      }
    }
    return output_pos;
  }

  Status CustomizedProcessing() override {
    return UpdateOrTransformParamInput(axis_node_pos_, "DataFormatDimMap",
                                       DT_INT32);
  }

  int axis_node_pos_;
};

class SplitVProcessor : public SplitProcessor {
 public:
  explicit SplitVProcessor(const OptimizeContext& opt_cxt)
      : SplitProcessor(opt_cxt) {
    axis_node_pos_ = 2;
  }

 protected:
  std::vector<int> GetInputPos() const override { return {0}; }
};

class TernaryOpProcessor : public AgnosticNodeProcessor {
 public:
  explicit TernaryOpProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override { return {0, 1, 2}; }
};

class SelectProcessor : public AgnosticNodeProcessor {
 public:
  explicit SelectProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    auto input0 = node_map_->GetNode(node_->input(0));
    int input0_port;
    ParseNodeName(node_->input(0), &input0_port);
    bool is_input0_scalar_vector_4d = IsPortDimsN(*input0, input0_port, 0) ||
                                      IsPortDimsN(*input0, input0_port, 1) ||
                                      IsPortDimsN(*input0, input0_port, 4);
    return AgnosticNodeProcessor::ShouldProcess() && is_input0_scalar_vector_4d;
  }

  std::vector<int> GetInputPos() const override {
    auto input0 = node_map_->GetNode(node_->input(0));
    int input0_port;
    ParseNodeName(node_->input(0), &input0_port);
    // Input 0 could be a scalar, a vector with size matching the first
    // dimension of input 1 and 2, or must have the same shape as input 1 and 2.
    if (IsPortDimsFour(*input0, input0_port)) {
      return {0, 1, 2};
    } else {
      return {1, 2};
    }
  }
};

class UnaryGradProcessor : public AgnosticNodeProcessor {
 public:
  explicit UnaryGradProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::vector<int> GetInputPos() const override { return {0, 1}; }
};

class SliceProcessor : public AgnosticNodeProcessor {
 public:
  explicit SliceProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {
    // Skip the first input, which is the data to be sliced.
    start_ = 1;
    // Note that we can't use node_->input_size() here because there
    // could be control inputs.
    end_ = 2;
  }

 protected:
  Status ProcessInputs() {
    for (int i = start_; i <= end_; i++) {
      DataType dtype = node_->attr().at("Index").type();
      TF_RETURN_IF_ERROR(
          UpdateOrTransformParamInput(i, "DataFormatVecPermute", dtype));
    }
    return Status::OK();
  }

  Status CustomizedProcessing() override { return ProcessInputs(); }

  int start_;
  int end_;
};

class StridedSliceProcessor : public SliceProcessor {
 public:
  explicit StridedSliceProcessor(const OptimizeContext& opt_cxt)
      : SliceProcessor(opt_cxt) {
    start_ = 1;
    end_ = 3;
  }

 protected:
  bool ShouldProcess() const override {
    return AgnosticNodeProcessor::ShouldProcess() && IsOnlyBeginEndMask();
  }

  Status CustomizedProcessing() override {
    TF_RETURN_IF_ERROR(UpdateMask("begin_mask"));
    TF_RETURN_IF_ERROR(UpdateMask("end_mask"));
    TF_RETURN_IF_ERROR(ProcessInputs());
    return Status::OK();
  }

 private:
  bool IsMaskZero(const string& mask) const {
    return node_->attr().at(mask).i() == 0;
  }

  bool IsOnlyBeginEndMask() const {
    return IsMaskZero("ellipsis_mask") && IsMaskZero("new_axis_mask") &&
           IsMaskZero("shrink_axis_mask");
  }

  Status UpdateMask(const string& mask) {
    int i = node_->attr().at(mask).i();
    if (i < 0 || i > 15) {
      return errors::InvalidArgument("invalid mask value: ", i);
    }
    if (i == 0 || i == 1 || i == 14 || i == 15) return Status::OK();
    switch (i) {
      case 2:
      case 3:
        i += 2;
        break;
      case 4:
      case 5:
        i += 4;
        break;
      case 6:
      case 7:
        i += 6;
        break;
      case 8:
      case 9:
        i -= 6;
        break;
      case 10:
      case 11:
        i -= 4;
        break;
      case 12:
      case 13:
        i -= 2;
        break;
    }
    node_->mutable_attr()->at(mask).set_i(i);
    return Status::OK();
  }
};

class StridedSliceGradProcessor : public StridedSliceProcessor {
 public:
  explicit StridedSliceGradProcessor(const OptimizeContext& opt_cxt)
      : StridedSliceProcessor(opt_cxt) {
    start_ = 0;
    end_ = 3;
  }

 protected:
  std::vector<int> GetInputPos() const override { return {4}; }
};

class SqueezeProcessor : public AgnosticNodeProcessor {
 public:
  explicit SqueezeProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    bool is_dims_supported = (IsPortZeroDimsN(*node_, 2) && IsAlongHW()) ||
                             (IsPortZeroDimsN(*node_, 1) && IsAlongNHW());
    return !MustPreserve() && HasOutputs() && IsNodeAfterNCHWToNHWC() &&
           IsInputConvertible() && is_dims_supported && IsOnGPU();
  }

  Status AddLayoutTransposeToOutputs() override { return Status::OK(); }

  Status CustomizedProcessing() override {
    TF_RETURN_IF_ERROR(HasAttribute(*node_, "squeeze_dims"));
    auto list = node_->mutable_attr()->at("squeeze_dims").mutable_list();
    if (list->i_size() == 2) {
      list->set_i(0, 2);
      list->set_i(1, 3);
    } else if (list->i_size() == 3) {
      list->set_i(1, 2);
      list->set_i(2, 3);
    }
    return Status::OK();
  }

 private:
  bool IsInputConvertible() const {
    int input_port;
    auto input = node_map_->GetNode(node_->input(0));
    ParseNodeName(node_->input(0), &input_port);
    if (input->attr().find("_output_shapes") != input->attr().end()) {
      auto shape = input->attr().at("_output_shapes").list().shape(input_port);
      if (shape.dim_size() != 4) {
        return false;
      }
      if (shape.dim(1).size() == 1 && shape.dim(2).size() == 1) {
        return true;
      }
      if (shape.dim(0).size() == 1 && shape.dim(1).size() == 1 &&
          shape.dim(2).size() == 1) {
        return true;
      }
    }
    return false;
  }

  bool IsAlongAxis(const std::vector<int>& axis) const {
    if (node_->attr().find("squeeze_dims") != node_->attr().end()) {
      auto list = node_->attr().at("squeeze_dims").list();
      // If list is empty, Squeeze op will squeeze all dimensions of size 1.
      if (list.i_size() == 0) return true;
      if (list.i_size() == axis.size()) {
        bool along_axis = true;
        for (int i = 0; i < axis.size(); i++) {
          along_axis = along_axis && (list.i(i) == axis[i]);
        }
        if (along_axis) return true;
      }
    }
    return false;
  }
  bool IsAlongHW() const { return IsAlongAxis({1, 2}); }
  bool IsAlongNHW() const { return IsAlongAxis({0, 1, 2}); }
};

class ReduceProcessor : public AgnosticNodeProcessor {
 public:
  explicit ReduceProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  bool ShouldProcess() const override {
    auto input0 = node_map_->GetNode(node_->input(0));
    int port;
    ParseNodeName(node_->input(0), &port);
    return !MustPreserve() && HasOutputs() && IsNodeAfterNCHWToNHWC() &&
           IsPortDimsFour(*input0, port) && IsReduceAxisSupported() &&
           IsOnGPU();
  }

  Status CustomizedProcessing() override {
    if (IsReduceAxisSupported()) {
      DataType dtype = node_->attr().at("Tidx").type();
      TF_RETURN_IF_ERROR(
          UpdateOrTransformParamInput(1, "DataFormatDimMap", dtype));
    }
    return Status::OK();
  }

  Status AddLayoutTransposeToOutputs() override {
    if (KeepDims()) {
      return AddTransformToOutputs("Transpose");
    }
    return Status::OK();
  }

 private:
  bool IsReduceAxisSupported() const {
    return KeepDims() || ((IsAlongAllFourDims() || IsAlongHWC() ||
                           IsAlongNHW() || IsAlongHW() || IsAlongC()) &&
                          !KeepDims());
  }

  bool IsAlongAxis(const std::vector<int>& axis) const {
    auto axis_node = node_map_->GetNode(node_->input(1));
    if (!IsConstant(*axis_node)) {
      return false;
    }
    if (HasAttribute(*axis_node, "value").ok()) {
      Tensor tensor;
      auto success = tensor.FromProto(axis_node->attr().at({"value"}).tensor());
      if (!success) {
        LOG(ERROR) << "Failed to parse TensorProto.";
      }
      if (tensor.dims() == 1 && tensor.dim_size(0) == axis.size()) {
        bool along_axis = true;
        for (int i = 0; i < axis.size(); i++) {
          along_axis = along_axis && (tensor.flat<int>()(i) == axis[i]);
        }
        if (along_axis) return true;
      }
    }
    return false;
  }

  bool IsAlongAllFourDims() const { return IsAlongAxis({0, 1, 2, 3}); }

  bool IsAlongHWC() const { return IsAlongAxis({1, 2, 3}); }

  bool IsAlongNHW() const { return IsAlongAxis({0, 1, 2}); }

  bool IsAlongHW() const { return IsAlongAxis({1, 2}); }

  bool IsAlongC() const { return IsAlongAxis({3}); }

  bool KeepDims() const { return node_->attr().at("keep_dims").b(); }
};

class SwitchProcessor : public AgnosticNodeProcessor {
 public:
  explicit SwitchProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  std::set<int> GetOutputPos() const override { return {0, 1}; }
};

class TileProcessor : public AgnosticNodeProcessor {
 public:
  explicit TileProcessor(const OptimizeContext& opt_cxt)
      : AgnosticNodeProcessor(opt_cxt) {}

 protected:
  Status CustomizedProcessing() override {
    DataType dtype = node_->attr().at("Tmultiples").type();
    return UpdateOrTransformParamInput(1, "DataFormatVecPermute", dtype);
  }
};

class DataLayoutOptimizer : GraphProcessor {
 public:
  explicit DataLayoutOptimizer(
      const GraphProperties& graph_properties,
      const VirtualPlacer& virtual_placer,
      const LayoutOptimizer::TuningConfig& config,
      const std::unordered_set<string>& nodes_to_preserve, GraphDef* graph,
      NodeMap* node_map)
      : GraphProcessor(graph_properties, virtual_placer, nodes_to_preserve,
                       graph, node_map),
        config_(config) {}

  Status Optimize() {
    VLOG(1) << "Number of nodes for original graph: " << graph_->node_size();
    TF_RETURN_IF_ERROR(Expand());
    VLOG(1) << "Number of nodes after Expand: " << graph_->node_size();
    TF_RETURN_IF_ERROR(Collapse());
    VLOG(1) << "Number of nodes after Collapse: " << graph_->node_size();
    return Status::OK();
  }

 private:
  NodeDef* AddNodePermNHWCToNCHW() {
    return AddNodePermConst(LayoutOptimizerNode(kPermNHWCToNCHW), "",
                            {0, 3, 1, 2});
  }

  NodeDef* AddNodePermNCHWToNHWC() {
    return AddNodePermConst(LayoutOptimizerNode(kPermNCHWToNHWC), "",
                            {0, 2, 3, 1});
  }

  // Expand all nodes which is in NHWC, but supports NCHW or is layout agnostic.
  Status Expand() {
    int node_size_original = graph_->node_size();
    std::unordered_map<const NodeDef*, std::vector<int>> frames;
    int num_frames;
    TF_RETURN_IF_ERROR(IdentifyFrames(*graph_, &frames, &num_frames));

    // This is the first pass where we expand the nodes which support NCHW.
    std::set<string> ops_format_supported = GetOpsFormatSupported();
    for (int i = 0; i < node_size_original; i++) {
      if (IsNodeByLayoutOptimizer(graph_->node(i).name())) {
        return Status(error::INVALID_ARGUMENT,
                      "The graph is already optimized by layout optimizer.");
      }
      if (ops_format_supported.find(graph_->node(i).op()) !=
          ops_format_supported.end()) {
        auto node = graph_->mutable_node(i);
        bool is_in_frame = !frames[node].empty();
        OptimizeContext opt_cxt(graph_, node, node_map_, graph_properties_,
                                virtual_placer_, nodes_to_preserve_,
                                is_in_frame);
        std::unique_ptr<NodeProcessor> node_processor;
        if (IsAvgPoolGrad(*node)) {
          node_processor.reset(new AvgPoolGradProcessor(opt_cxt));
        } else if (IsBiasAddGrad(*node)) {
          node_processor.reset(new BiasAddGradProcessor(opt_cxt));
        } else if (IsConv2D(*node)) {
          node_processor.reset(new Conv2DProcessor(opt_cxt, config_.no_gemm));
        } else if (IsConv2DBackpropFilter(*node)) {
          node_processor.reset(
              new Conv2DBackpropFilterProcessor(opt_cxt, config_.no_gemm));
        } else if (IsConv2DBackpropInput(*node)) {
          node_processor.reset(
              new Conv2DBackpropInputProcessor(opt_cxt, config_.no_gemm));
        } else if (IsDepthwiseConv2dNative(*node)) {
          node_processor.reset(new Conv2DProcessor(opt_cxt, true));
        } else if (IsDepthwiseConv2dNativeBackpropFilter(*node)) {
          node_processor.reset(
              new Conv2DBackpropFilterProcessor(opt_cxt, true));
        } else if (IsDepthwiseConv2dNativeBackpropInput(*node)) {
          node_processor.reset(new Conv2DBackpropInputProcessor(opt_cxt, true));
        } else if (IsFusedBatchNormGrad(*node)) {
          node_processor.reset(new FusedBatchNormGradProcessor(opt_cxt));
        } else if (IsMaxPoolV2(*node)) {
          node_processor.reset(new MaxPoolV2Processor(opt_cxt));
        } else if (IsMaxPoolGradV1(*node) || IsMaxPoolGradGradV1(*node)) {
          node_processor.reset(new MaxPoolGradProcessor(opt_cxt));
        } else if (IsMaxPoolGradV2(*node) || IsMaxPoolGradGradV2(*node)) {
          node_processor.reset(new MaxPoolGradV2Processor(opt_cxt));
        } else {
          node_processor.reset(new NodeProcessor(opt_cxt));
        }
        TF_RETURN_IF_ERROR(node_processor->ConvertNode());
      }
    }

    // This is the second pass where we expand layout-agnostic nodes. This pass
    // only needs to be performed if at least one node in the previous pass is
    // expanded.
    if (graph_->node_size() > node_size_original) {
      NodeDef* n = AddNodePermNHWCToNCHW();
      n = AddNodePermNCHWToNHWC();
      std::set<string> ops_format_agnostic = GetOpsFormatAgnostic();
      for (int i = 0; i < graph_->node_size(); i++) {
        if (ops_format_agnostic.find(graph_->node(i).op()) !=
            ops_format_agnostic.end()) {
          auto node = graph_->mutable_node(i);
          bool is_in_frame = !frames[node].empty();
          OptimizeContext opt_cxt(graph_, node, node_map_, graph_properties_,
                                  virtual_placer_, nodes_to_preserve_,
                                  is_in_frame);
          std::unique_ptr<NodeProcessor> node_processor;
          if (IsAddN(*node)) {
            node_processor.reset(new AddNProcessor(opt_cxt));
          } else if (IsBetainc(*node)) {
            node_processor.reset(new TernaryOpProcessor(opt_cxt));
          } else if (IsBinaryOp(*node)) {
            node_processor.reset(new BinaryOpProcessor(opt_cxt));
          } else if (IsConcat(*node)) {
            node_processor.reset(new ConcatProcessor(opt_cxt));
          } else if (IsFill(*node)) {
            node_processor.reset(new FillProcessor(opt_cxt));
          } else if (IsHistogramSummary(*node)) {
            node_processor.reset(new HistogramSummaryProcessor(opt_cxt));
          } else if (IsIdentityN(*node)) {
            node_processor.reset(new IdentityNProcessor(opt_cxt));
          } else if (IsMerge(*node)) {
            node_processor.reset(new MergeProcessor(opt_cxt));
          } else if (IsPad(*node) || IsMirrorPad(*node) ||
                     IsMirrorPadGrad(*node)) {
            node_processor.reset(new PadProcessor(opt_cxt));
          } else if (IsReduceOp(*node)) {
            node_processor.reset(new ReduceProcessor(opt_cxt));
          } else if (IsReverseV2(*node)) {
            node_processor.reset(new ReverseProcessor(opt_cxt));
          } else if (IsSelect(*node)) {
            node_processor.reset(new SelectProcessor(opt_cxt));
          } else if (IsSlice(*node)) {
            node_processor.reset(new SliceProcessor(opt_cxt));
          } else if (IsStridedSlice(*node)) {
            node_processor.reset(new StridedSliceProcessor(opt_cxt));
          } else if (IsShape(*node) || IsShapeN(*node)) {
            node_processor.reset(new ShapeProcessor(opt_cxt));
          } else if (IsSplit(*node)) {
            node_processor.reset(new SplitProcessor(opt_cxt));
          } else if (IsSplitV(*node)) {
            node_processor.reset(new SplitVProcessor(opt_cxt));
          } else if (IsSqueeze(*node)) {
            node_processor.reset(new SqueezeProcessor(opt_cxt));
          } else if (IsStridedSliceGrad(*node)) {
            node_processor.reset(new StridedSliceGradProcessor(opt_cxt));
          } else if (IsSwitch(*node)) {
            node_processor.reset(new SwitchProcessor(opt_cxt));
          } else if (IsTile(*node)) {
            node_processor.reset(new TileProcessor(opt_cxt));
          } else if (IsUnaryGrad(*node)) {
            node_processor.reset(new UnaryGradProcessor(opt_cxt));
          } else {
            node_processor.reset(new AgnosticNodeProcessor(opt_cxt));
          }
          TF_RETURN_IF_ERROR(node_processor->ConvertNode());
        }
      }
    }
    return Status::OK();
  }

  // Remove all node pairs, where a NCHW-to-NHWC node is followed by
  // a NHWC-to-NCHW node.
  Status Collapse() {
    std::unordered_set<string> nodes_removable;
    for (int i = 0; i < graph_->node_size(); i++) {
      auto node = graph_->mutable_node(i);
      node->mutable_attr()->erase("_output_shapes");
      if (IsTransposeNHWCToNCHW(node->name()) ||
          IsDimMapNHWCToNCHW(node->name()) ||
          IsVecPermuteNHWCToNCHW(node->name())) {
        bool transpose_pair = IsTransposeNHWCToNCHW(node->name()) &&
                              IsTransposeNCHWToNHWC(node->input(0));
        bool dim_map_pair = IsDimMapNHWCToNCHW(node->name()) &&
                            IsDimMapNCHWToNHWC(node->input(0));
        bool vec_permute_pair = IsVecPermuteNHWCToNCHW(node->name()) &&
                                IsVecPermuteNCHWToNHWC(node->input(0));
        if (transpose_pair || dim_map_pair || vec_permute_pair) {
          const string& trans_first = node->input(0);
          const string& trans_second = node->name();
          auto outputs = node_map_->GetOutputs(trans_second);
          CHECK(outputs.size() == 1)
              << "There is always only a single output for a Transpose node, "
              << "due to the way it is added by NodeProcessor.";
          NodeDef* output = *outputs.begin();
          string input = node_map_->GetNode(trans_first)->input(0);
          for (int i = 0; i < output->input_size(); i++) {
            if (output->input(i).compare(trans_second) == 0) {
              *output->mutable_input(i) = input;
              break;
            }
          }
          nodes_removable.insert(trans_first);
          nodes_removable.insert(trans_second);
        }
      }
    }
    graph_->mutable_node()->erase(
        std::remove_if(
            graph_->mutable_node()->begin(), graph_->mutable_node()->end(),
            [nodes_removable](const NodeDef& node) {
              return nodes_removable.find(node.name()) != nodes_removable.end();
            }),
        graph_->mutable_node()->end());
    return Status::OK();
  }

  const LayoutOptimizer::TuningConfig& config_;
};

int GetNumGPUs(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  for (const auto& device : devices) {
    if (device.second.type() == "GPU") {
      num_gpus++;
    }
  }
  return num_gpus;
}
}  // namespace

Status LayoutOptimizer::Tune(const GrapplerItem& item,
                             const GraphProperties& graph_properties,
                             const TuningConfig& config, GraphDef* output) {
  auto status = graph_properties.AnnotateOutputShapes(output);
  if (!status.ok()) {
    VLOG(1) << "Annotate shape return status: " << status.ToString();
    *output = item.graph;
    return status;
  }
  NodeMap node_map(output);
  DataLayoutOptimizer layout_optimizer(graph_properties, *virtual_placer_,
                                       config, nodes_to_preserve_, output,
                                       &node_map);
  status = layout_optimizer.Optimize();
  return status;
}

Status LayoutOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output) {
  if (cluster == nullptr) {
    return errors::InvalidArgument("cluster == nullptr");
  }

  if (GetNumGPUs(*cluster) < 1) {
    // LayoutOptimizer is currently only tuned for GPU.
    *output = item.graph;
    return Status::OK();
  }

  virtual_placer_.reset(new VirtualPlacer(cluster));
  nodes_to_preserve_ = item.NodesToPreserve();
  GraphProperties graph_properties(item);
  auto status = graph_properties.InferStatically(false);
  if (!status.ok()) {
    VLOG(1) << "Infer shape return status: " << status.ToString();
    *output = item.graph;
    return status;
  }
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  TuningConfig config;
  config.no_gemm = true;
  // TODO(yaozhang): Enable tuning with various TuningConfig choices with
  // the measurement-based estimator.
  status = Tune(item, graph_properties, config, output);
  if (!status.ok()) {
    *output = item.graph;
  }
  return status;
}

void LayoutOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for LayoutOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
