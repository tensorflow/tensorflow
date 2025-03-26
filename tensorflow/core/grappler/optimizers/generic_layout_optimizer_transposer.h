/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

constexpr char kAttrSrcFormat[] = "src_format";
constexpr char kAttrDstFormat[] = "dst_format";
constexpr char kAttrOutputShape[] = "_output_shapes";
constexpr char kGPU[] = "GPU";
constexpr char kCPU[] = "CPU";

// TransposeContext owns all data members. Must initialize GraphProperties,
// FrameView, GraphDef and MutableGraphView with the same graph. NodeDef
// pointers in FrameView, GraphDef and MutableGraphView must point to nodes in
// the same GraphDef instance.
struct TransposeContext {
  // Initializes TransposeContext with given GrapplerItem. Because initializing
  // FrameMap and GraphProperties may return error, we initialize
  // TransposeContext outside constructor.
  static absl::Status InitializeTransposeContext(bool assume_valid_feeds,
                                                 const GrapplerItem& item,
                                                 const Cluster* cluster,
                                                 TransposeContext* context);

  static absl::Status InitializeTransposeContext(const GrapplerItem& item,
                                                 const Cluster* cluster,
                                                 TransposeContext* context) {
    return InitializeTransposeContext(false, item, cluster, context);
  }

  // Sets data formats to convert from and to for specified device type.
  void AssignDeviceAndDataFormats(absl::string_view target_device,
                                  absl::string_view src_format,
                                  absl::string_view dst_format);

  FrameView frames;
  GraphDef graph;
  // Number of nodes in the original graph. As new nodes are appended to the end
  // of the graph, all new nodes should have a node index greater than or equal
  // to this.
  int num_nodes;
  absl::flat_hash_set<string> nodes_to_preserve;
  std::unique_ptr<GraphProperties> graph_properties;
  std::unique_ptr<utils::MutableGraphView> graph_view;

  string target_device;
  string src_format;
  string dst_format;
  absl::flat_hash_map<char, int> src_dim_indices;
  absl::flat_hash_map<char, int> dst_dim_indices;
  std::vector<int> src_to_dst;
  std::vector<int> dst_to_src;

  string enforced_layout;
};

class Transposer {
 public:
  explicit Transposer() {}

  Transposer(const Transposer&) = delete;
  Transposer& operator=(const Transposer&) = delete;

  virtual ~Transposer() {}

  // Returns true iff the node should be processed by this transposer.
  // NodeProcessors may perform additional oprand specific checks before
  // processing if necessary.
  // Following common conditions are checked:
  // * node's device matches target device
  // * node's source format matches config's source format
  // * node has output
  bool ShouldProcess(const TransposeContext& context,
                     const utils::MutableNodeView& node) const;

  // Transposes given node from src format to dst format. Also perform other
  // necessary operations to guarantee the graph produce the same result.
  // Eg. Add Transpose node sets before fanin ports and after fanout ports.
  virtual absl::Status TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) = 0;

  // Creates a Const node for permutation. If node with node_name already exits,
  // return and reuse it.
  absl::Status CreateConstPermNode(TransposeContext* context,
                                   absl::string_view node_name,
                                   absl::string_view device,
                                   absl::Span<const int> permutation,
                                   absl::string_view control_node_name,
                                   utils::MutationNewNode* added_node);

  // Creates a TransposeNode with given properties. If node with node_name
  // already exits, return and reuse it.
  // A const perm node is also created and connected to the 2nd fanin.
  // control_node_name is ignored if it is empty.
  absl::Status CreateTransposeNode(
      TransposeContext* context, absl::string_view name_format,
      const DataType& data_type, absl::string_view device,
      TensorShapeProto fanin_shape, absl::Span<const int> permutation,
      absl::string_view control_node_name, utils::MutationNewNode* added_node,
      string* transpose_node_name);

  // Update all edges between dst_node->fanin[dst_ports] and dst_node by
  // inserting an op node.
  absl::Status UpdateFaninEdgesWithOp(TransposeContext* context,
                                      absl::Span<const int> dst_ports,
                                      utils::MutableNodeView* dst_node,
                                      absl::string_view op);

  // Update all edges between src_node:src_ports and nodes take
  // src_node:src_ports as fanin. Also update attr _output_shape of src_node.
  absl::Status UpdateFanoutEdgesWithOp(TransposeContext* context,
                                       absl::Span<const int> src_ports,
                                       utils::MutableNodeView* src_node,
                                       absl::string_view op);

  // Creates a DataFromat node with given properties.
  // DataFromat op is either DataFormatVecPermute or DataFormatDimMap.
  absl::Status CreateDataFormatNode(
      TransposeContext* context, absl::string_view node_name,
      absl::string_view op, absl::string_view device, const DataType& data_type,
      bool is_fanin_on_host, bool is_src_format_to_dst_format,
      utils::MutationNewNode* added_node);

 protected:
  int GetFanoutPortRank(const utils::MutableNodeView& node, int port) const;
  bool IsFanoutPortRankN(const utils::MutableNodeView& node, int port,
                         int n) const;
  bool IsFanoutPortsRankN(const utils::MutableNodeView& node,
                          absl::Span<const int> ports, int n) const;
  int GetFaninPortRank(const utils::MutableNodeView& node, int port) const;
  bool IsFaninPortRankN(const utils::MutableNodeView& node, int port,
                        int n) const;

  // Checks if fanin at specified port(s) has dimensions `dims` iff fanin is a
  // Const. If fanin is not a Const, no dimensions will be checked and this will
  // return true.
  bool IsFaninPortDimsNIfConst(const utils::MutableNodeView& node, int port,
                               absl::Span<const int> dims) const;
  bool IsFaninPortsDimsNIfConst(const utils::MutableNodeView& node,
                                absl::Span<const int> ports,
                                absl::Span<const int> dims) const;
  bool CanProcessNode(const TransposeContext& context,
                      const utils::MutableNodeView& node) const;
  // Update all edges between dst_node->fanin[dst_ports] and dst_node.
  // A node with op is created and inserted between all edges.
  // op is one of Transpose, DataFormatVecPermute or DataFormatDimMap.
  absl::Status UpdateEdge(TransposeContext* context,
                          absl::string_view name_format, absl::string_view op,
                          const AttrValue* input_shape, bool is_in_frame,
                          bool is_src_format_to_dst_format, const int src_port,
                          const int dst_port, utils::MutableNodeView* src_node,
                          utils::MutableNodeView* dst_node);
  string GetFaninNameFormat(absl::string_view node_name, int port,
                            absl::string_view src_format,
                            absl::string_view dst_format);
  string GetFanoutNameFormat(absl::string_view node_name, int port, int index,
                             absl::string_view src_format,
                             absl::string_view dst_format);
  string LayoutOptimizerNode(absl::string_view node_name);
  string GetReshapeNodeNameFormat(absl::string_view node_name, int index,
                                  absl::string_view src_format,
                                  absl::string_view dst_format);
  string GetShapeConstNodeNameFormat(absl::string_view node_name, int index);
};

class LayoutSensitiveOpTransposer : public Transposer {
 public:
  explicit LayoutSensitiveOpTransposer() : Transposer() {}

  // Updates attrs data_format, ksize, strides of the given node to dst_format.
  // _output_shape is updated during UpdateOutputEdges.
  absl::Status UpdateNode(TransposeContext* context,
                          utils::MutableNodeView* node);
};

// Layout sensitive op transposers.

class DefaultLayoutSensitiveOpTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit DefaultLayoutSensitiveOpTransposer()
      : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class BiasAddTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit BiasAddTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class AvgPoolGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit AvgPoolGradTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class BiasAddGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit BiasAddGradTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class Conv2DBackpropFilterTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv2DBackpropFilterTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class Conv2DBackpropInputTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv2DBackpropInputTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class Conv3DTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv3DTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class Conv3DBackpropFilterTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv3DBackpropFilterTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class Conv3DBackpropInputTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit Conv3DBackpropInputTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class FusedBatchNormExTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit FusedBatchNormExTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class FusedBatchNormGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit FusedBatchNormGradTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 private:
  bool IsTraining(const utils::MutableNodeView& node) const;
};

class MaxPoolV2Transposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPoolV2Transposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class MaxPool3DTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPool3DTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class MaxPoolGradTransposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPoolGradTransposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class MaxPoolGradV2Transposer : public LayoutSensitiveOpTransposer {
 public:
  explicit MaxPoolGradV2Transposer() : LayoutSensitiveOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

// Layout agnostic op transposers.

class LayoutAgnosticOpTransposer : public Transposer {
 public:
  explicit LayoutAgnosticOpTransposer() : Transposer() {}

 protected:
  bool IsAfterDstToSrcTransform(const TransposeContext& context,
                                const utils::MutableNodeView& node) const;

  std::vector<int> GetVariadicNDFaninPorts(const TransposeContext& context,
                                           const utils::MutableNodeView& node,
                                           int rank) const;
};

class DefaultLayoutAgnosticOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit DefaultLayoutAgnosticOpTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class AddNTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit AddNTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class BinaryOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit BinaryOpTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 private:
  bool IsNDOperateWithMD(const utils::MutableNodeView& node, int n, int m);
  bool IsFaninShapeSupported(const utils::MutableNodeView& node, int rank);
  std::vector<int> GetNDDataFaninPorts(const utils::MutableNodeView& node,
                                       int rank);
  absl::Status AddNodeShapeConst(utils::Mutation* mutation,
                                 absl::string_view node_name,
                                 absl::string_view node_device,
                                 bool node_in_frame, int num_channels,
                                 absl::string_view depended_node, int rank);
  absl::Status AddNodeReshape(utils::Mutation* mutation,
                              absl::string_view node_name,
                              absl::string_view node_device,
                              absl::string_view input_name,
                              absl::string_view shape_const_node_name,
                              const DataType& data_type);
  absl::Status MaybeReshapeVectorFanin(TransposeContext* context,
                                       utils::MutableNodeView* node, int rank);
};

class ConcatOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ConcatOpTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class FillOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit FillOpTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class IdentityNTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit IdentityNTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class MergeTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit MergeTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 private:
  bool IsEveryFaninAfterDstToSrcTransform(
      const TransposeContext& context,
      const utils::MutableNodeView& node) const;
};

class PadTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit PadTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class ReduceTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ReduceTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 private:
  bool KeepDims(const utils::MutableNodeView& node);
  bool IsAlongAxis(const Tensor& tensor, absl::Span<const int> axis, int rank);
  bool IsReduceAxisSupported(const TransposeContext& context,
                             const utils::MutableNodeView& node, int rank);
};

class ReverseV2Transposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ReverseV2Transposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class SelectTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SelectTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 protected:
  bool IsFaninScalarVector4D(const utils::MutableNodeView& fanin, int port);
  std::vector<int> GetFaninPorts(const utils::MutableNodeView& fanin, int port);
};

class ShapeTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ShapeTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class ShapeNTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit ShapeNTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class SliceTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SliceTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class SplitTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SplitTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class SplitVTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SplitVTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class SqueezeTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SqueezeTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 private:
  bool IsInputConvertible(const TransposeContext& context,
                          const utils::MutableNodeView& node) const;
  bool IsAlongAxis(const AttrValue& attr, absl::Span<const int> axis,
                   int rank) const;
  bool IsDimsSupported(const TransposeContext& context,
                       const utils::MutableNodeView& node) const;
  absl::Status UpdateSqueezeDims(TransposeContext* context,
                                 utils::MutableNodeView* node);
};

class StridedSliceTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit StridedSliceTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;

 private:
  bool IsMaskZero(const utils::MutableNodeView& node, absl::string_view mask);
  bool HasOnlyBeginEndMask(const utils::MutableNodeView& node);
  absl::Status PermuteMask(TransposeContext* context,
                           utils::MutableNodeView* node,
                           absl::string_view mask);
};

class SwitchTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit SwitchTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class TernaryOpTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit TernaryOpTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class TileTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit TileTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

class UnaryGradTransposer : public LayoutAgnosticOpTransposer {
 public:
  explicit UnaryGradTransposer() : LayoutAgnosticOpTransposer() {}

  absl::Status TransposeNode(TransposeContext* context,
                             utils::MutableNodeView* node) override;
};

// Utils.

// Permutes elements according to permutation and replaces the original values.
// Permutation and values must have same size.
template <typename T>
absl::Status PermuteSingle(absl::string_view location,
                           absl::Span<const int> permutation, T* values) {
  DCHECK(values != nullptr);
  int permutation_size = permutation.size();
  if (values->size() != permutation_size) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrCat("Size of values ", values->size(),
                                     " does not match size of permutation ",
                                     permutation_size, " @ ", location));
  }
  typedef typename T::value_type V;
  std::vector<V> elements(values->begin(), values->end());
  int index = 0;
  for (V& element : *values) {
    element = elements[permutation[index++]];
  }
  return absl::OkStatus();
}

// Permutes two elements at a time according to permutation and replaces the
// original values. Values must be twice the size of permutation.
template <typename T>
absl::Status PermuteDouble(absl::string_view location,
                           absl::Span<const int> permutation, T* values) {
  DCHECK(values != nullptr);
  int permutation_size = permutation.size();
  if (values->size() != permutation_size * 2) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Size of values ", values->size(),
                     " does not match twice the size of permutation ",
                     permutation_size, " @ ", location));
  }
  typedef typename T::value_type V;
  std::vector<V> elements(values->begin(), values->end());
  for (int i = 0; i < values->size(); i = i + 2) {
    const int permutation_index = permutation[i / 2];
    (*values)[i] = elements[permutation_index * 2];
    (*values)[i + 1] = elements[permutation_index * 2 + 1];
  }
  return absl::OkStatus();
}

string GetDeviceName(const NodeDef& node);

bool IsDefaultLayoutSensitiveOp(const NodeDef& node);

bool IsLayoutSensitiveOp(const NodeDef& node);

bool IsDefaultLayoutAgnosticOp(const NodeDef& node);

bool IsLayoutAgnosticOp(const NodeDef& node);

bool IsTernaryOp(const NodeDef& node);

bool IsUnaryGrad(const NodeDef& node);

bool IsMaxPoolV2(const NodeDef& node);

bool IsMaxPool3D(const NodeDef& node);

bool IsMaxPoolGradV2(const NodeDef& node);

bool IsMaxPoolGradGradV1(const NodeDef& node);

bool IsMaxPoolGradGradV2(const NodeDef& node);

bool IsBinaryOp(const NodeDef& node);

bool IsReduceOp(const NodeDef& node);

std::vector<int> GetDataFaninPorts(const utils::MutableNodeView& node);

std::vector<int> GetDataFanoutPorts(const utils::MutableNodeView& node);

// Returns a value of constant input to the `node` at `index`, iff `predicate`
// evaluated to true. Returns true if `tensor` was populated with data.
bool GetValueAttrFromConstInputNode(
    const utils::MutableNodeView& node,
    const std::function<bool(const NodeDef&)>& predicate, int index,
    Tensor* tensor);

bool IsDataFormatOp(const utils::MutableNodeView& node);

absl::flat_hash_map<char, int> GetDimensionIndices(
    absl::string_view data_format);

std::vector<int> GetPermutation(
    const absl::flat_hash_map<char, int>& src_dim_indices,
    absl::string_view dst_format);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_TRANSPOSER_H_
