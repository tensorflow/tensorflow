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

#define EIGEN_USE_THREADS

#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {
namespace grappler {
using TensorVector = gtl::InlinedVector<TensorValue, 4>;

namespace {
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(thread::ThreadPool* pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}
  void Schedule(std::function<void()> fn) override {
    auto wrapped = [=]() {
      // TensorFlow flushes denormals to zero and rounds to nearest, so we do
      // the same here.
      port::ScopedFlushDenormal flush;
      port::ScopedSetRound round(FE_TONEAREST);
      fn();
    };
    pool_->Schedule(std::move(wrapped));
  }
  int NumThreads() const override { return pool_->NumThreads(); }
  int CurrentThreadId() const override { return pool_->CurrentThreadId(); }

 private:
  thread::ThreadPool* pool_ = nullptr;
};

class DeviceSimple : public DeviceBase {
 public:
  DeviceSimple() : DeviceBase(Env::Default()) {
    eigen_worker_threads_.num_threads = port::NumSchedulableCPUs();
    eigen_worker_threads_.workers = new thread::ThreadPool(
        Env::Default(), "constant_folding", eigen_worker_threads_.num_threads);
    eigen_threadpool_wrapper_.reset(
        new EigenThreadPoolWrapper(eigen_worker_threads_.workers));
    eigen_device_.reset(new Eigen::ThreadPoolDevice(
        eigen_threadpool_wrapper_.get(), eigen_worker_threads_.num_threads));
    set_tensorflow_cpu_worker_threads(&eigen_worker_threads_);
    set_eigen_cpu_device(eigen_device_.get());
  }
  ~DeviceSimple() override {
    eigen_threadpool_wrapper_.reset();
    eigen_device_.reset();
    delete eigen_worker_threads_.workers;
  }
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override {
    Tensor parsed(tensor_proto.dtype());
    if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
      return errors::InvalidArgument("Cannot parse tensor from tensor_proto.");
    }
    *tensor = parsed;
    return Status::OK();
  }
  Allocator* GetAllocator(AllocatorAttributes attr) override {
    return cpu_allocator();
  }

 private:
  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolInterface> eigen_threadpool_wrapper_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
};

template <typename T>
bool AllValuesAre(const TensorProto& proto, const T& value) {
  Tensor tensor;
  if (!tensor.FromProto(proto)) {
    return false;
  }
  auto values = tensor.flat<T>();
  for (int i = 0; i < tensor.NumElements(); ++i) {
    if (values(i) != value) {
      return false;
    }
  }
  return true;
}

// Add new_input as a control input to node if it does not already depend on it.
// TODO(rmlarsen): Move the following two utility functions to utils.{h,cc} and
// clean up code that should be using them.
bool MaybeAddControlInput(const string& ctrl_input, NodeDef* node,
                          GraphDef* graph, NodeMap* node_map) {
  bool already_exists = false;
  for (const string& input : node->input()) {
    if (input == ctrl_input || AsControlDependency(input) == ctrl_input) {
      already_exists = true;
      break;
    }
  }
  if (!already_exists) {
    const string ctrl_dep =
        ConstantFolding::AddControlDependency(ctrl_input, graph, node_map);
    node->add_input(ctrl_dep);
    node_map->AddOutput(NodeName(ctrl_input), node->name());
  }
  return !already_exists;
}

// Remove old_input as a control input to node.
bool MaybeRemoveControlInput(const string& old_input, NodeDef* node,
                             GraphDef* graph, NodeMap* node_map) {
  bool removed_input = false;
  bool update_node_map = true;
  const string old_input_ctrl_dep = AsControlDependency(NodeName(old_input));
  for (int i = 0; i < node->input_size(); ++i) {
    const string& input = node->input(i);
    if (old_input_ctrl_dep == input) {
      if (IsControlInput(input)) {
        node->mutable_input()->SwapElements(i, node->input_size() - 1);
        node->mutable_input()->RemoveLast();
        removed_input = true;
      } else {
        // There is a non-control input from the same node.
        // Don't remove the output from the NodeMap.
        update_node_map = false;
      }
    }
  }
  if (update_node_map) {
    node_map->RemoveOutput(NodeName(old_input), node->name());
  }
  return removed_input;
}

}  // namespace

ConstantFolding::ConstantFolding(RewriterConfig::Toggle opt_level,
                                 DeviceBase* cpu_device)
    : opt_level_(opt_level), cpu_device_(cpu_device) {
  resource_mgr_.reset(new ResourceMgr());
}

ConstantFolding::ConstantFolding(DeviceBase* cpu_device)
    : ConstantFolding(RewriterConfig::ON, cpu_device) {}

// static
string ConstantFolding::AddControlDependency(const string& input_name,
                                             GraphDef* graph,
                                             NodeMap* node_map) {
  if (IsControlInput(input_name)) {
    return input_name;
  }
  const NodeDef* node = node_map->GetNode(input_name);
  if (!IsSwitch(*node)) {
    return AsControlDependency(*node);
  } else {
    // We can't anchor control dependencies directly on the switch node: unlike
    // other nodes only one of the outputs of the switch node will be generated
    // when the switch node is executed, and we need to make sure the control
    // dependency is only triggered when the corresponding output is triggered.
    // We start by looking for an identity node connected to the output of the
    // switch node, and use it to anchor the control dependency.
    auto outputs = node_map->GetOutputs(node->name());
    for (const NodeDef* output : outputs) {
      if (IsIdentity(*output)) {
        if (IsSameInput(node->input(0), input_name)) {
          return AsControlDependency(*output);
        }
      }
    }
    // We haven't found an existing node where we can anchor the control
    // dependency: add a new identity node.
    int port = 0;
    string ctrl_dep_name = ParseNodeName(input_name, &port);
    strings::StrAppend(&ctrl_dep_name, "_", port);
    ctrl_dep_name = AddPrefixToNodeName(ctrl_dep_name, kConstantFoldingCtrl);
    const DataType output_type = node->attr().at("T").type();

    NodeDef* added_node = node_map->GetNode(ctrl_dep_name);
    if (added_node == nullptr) {
      added_node = graph->add_node();
      added_node->set_name(ctrl_dep_name);
      added_node->set_op("Identity");
      added_node->set_device(node->device());

      (*added_node->mutable_attr())["T"].set_type(output_type);
      *added_node->add_input() = input_name;
      node_map->AddNode(added_node->name(), added_node);
      node_map->AddOutput(node->name(), added_node->name());
    }
    return AsControlDependency(*added_node);
  }
}

// Puts the given value into the tensor at the given "flat" index.
static Status PutValueIntoTensor(const int64 value, const DataType& type,
                                 const int index, Tensor* tensor) {
  if (type == DT_INT32) {
    if (value >= INT_MAX) {
      return Status(error::INVALID_ARGUMENT, "int32 overflow");
    }
    tensor->flat<int32>()(index) = static_cast<int32>(value);
  } else {
    tensor->flat<int64>()(index) = value;
  }
  return Status::OK();
}

// Writes the given tensor shape into the given tensor.
// Op is assumed to be Shape, ShapeN, Size or Rank.
static Status ConvertShapeToConstant(const string& op, const DataType& type,
                                     const PartialTensorShape& shp,
                                     Tensor* tensor) {
  if (op == "Shape" || op == "ShapeN") {
    *tensor = Tensor(type, TensorShape({shp.dims()}));
    for (int i = 0; i < shp.dims(); ++i) {
      TF_RETURN_IF_ERROR(PutValueIntoTensor(shp.dim_size(i), type, i, tensor));
    }
  } else if (op == "Size") {
    int64 size = 1;
    for (int i = 0; i < shp.dims(); ++i) {
      size *= shp.dim_size(i);
    }
    *tensor = Tensor(type, TensorShape({}));
    TF_RETURN_IF_ERROR(PutValueIntoTensor(size, type, 0, tensor));
  } else {
    CHECK_EQ(op, "Rank");
    *tensor = Tensor(type, TensorShape({}));
    TF_RETURN_IF_ERROR(PutValueIntoTensor(shp.dims(), type, 0, tensor));
  }
  return Status::OK();
}

// TODO(rmlarsen): Perhaps we should move this to the GraphOptimizer base class.
bool ConstantFolding::OptimizedNodeExists(const NodeDef& node,
                                          StringPiece suffix) const {
  return node_map_->NodeExists(OptimizedNodeName(node, suffix));
}

string ConstantFolding::OptimizedNodeName(const NodeDef& node,
                                          StringPiece suffix) const {
  return AddPrefixToNodeName(strings::StrCat(node.name(), suffix),
                             kConstantFoldingConst);
}

bool ConstantFolding::IsReallyConstant(const NodeDef& node) const {
  if (!IsConstant(node)) {
    return false;
  }
  // If the node is fed it's not constant anymore.
  return feed_nodes_.find(node.name()) == feed_nodes_.end();
}

// Materialize the shapes using constants whenever possible.
Status ConstantFolding::MaterializeShapes(const GraphProperties& properties) {
  // We may add some nodes to the graph to encode control dependencies and hold
  // the materialized shapes: there is no need to process these added nodes, so
  // only iterate over the nodes of the input graph.
  const int node_count = graph_->node_size();
  for (int node_idx = 0; node_idx < node_count; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    const string op = node->op();
    if (op != "Shape" && op != "Size" && op != "Rank" && op != "ShapeN" &&
        op != "TensorArraySizeV3") {
      continue;
    }

    const std::vector<OpInfo::TensorProperties>& output =
        properties.GetOutputProperties(node->name());
    const std::vector<OpInfo::TensorProperties>& input =
        properties.GetInputProperties(node->name());
    if (input.empty() || output.empty()) {
      continue;
    }

    if (op == "Shape" || op == "Size" || op == "Rank") {
      CHECK_EQ(1, output.size());
      CHECK_EQ(1, input.size());

      const DataType type = output[0].dtype();
      CHECK(type == DT_INT32 || type == DT_INT64);
      const PartialTensorShape shape(input[0].shape());

      if ((op != "Rank" && !shape.IsFullyDefined()) ||
          (op == "Rank" && shape.unknown_rank())) {
        continue;
      }

      Tensor constant_value(type);
      if (!ConvertShapeToConstant(op, type, shape, &constant_value).ok()) {
        continue;
      }

      // Repurpose the existing node to be the constant.
      // Device placement is preserved.
      node->set_op("Const");
      node->clear_attr();
      (*node->mutable_attr())["dtype"].set_type(type);
      constant_value.AsProtoTensorContent(
          (*node->mutable_attr())["value"].mutable_tensor());

      // Turn the data input into a control dependency: this is needed to
      // ensure that the constant value will only be run in the
      // cases where the shape/rank/size would have been run in
      // the original graph.
      string ctrl_dep =
          AddControlDependency(node->input(0), graph_, node_map_.get());
      node->set_input(0, ctrl_dep);
      node_map_->AddOutput(NodeName(ctrl_dep), node->name());

      // Done with the Shape/Size/Rank node, move to the next node.
      continue;
    }

    if (op == "TensorArraySizeV3") {
      const NodeDef* array = node_map_->GetNode(node->input(0));
      if (array->attr().count("dynamic_size") != 0 &&
          array->attr().at("dynamic_size").b()) {
        continue;
      }
      const NodeDef* array_size = node_map_->GetNode(array->input(0));
      if (IsReallyConstant(*array_size)) {
        // Don't materialize 0 sizes to avoid triggering incorrect static
        // checks. A 0 sized array that can't grow isn't useful anyway.
        const TensorProto& raw_val = array_size->attr().at("value").tensor();
        if (raw_val.dtype() != DT_INT32) {
          continue;
        }
        Tensor value(raw_val.dtype(), raw_val.tensor_shape());
        if (!value.FromProto(raw_val)) {
          continue;
        }
        if (value.flat<int32>()(0) == 0) {
          continue;
        }
        node->set_op("Const");
        *node->mutable_attr() = array_size->attr();
        node->set_input(0, AsControlDependency(NodeName(node->input(0))));
        node->set_input(1, AddControlDependency(NodeName(node->input(1)),
                                                graph_, node_map_.get()));
      }
      continue;
    }

    // Handle ShapeN materialization case.
    // It's possible that not all input tensors have known shapes.
    CHECK_EQ(op, "ShapeN");
    CHECK_EQ(input.size(), output.size());
    const NodeDef* const shape_n_node = node;
    for (int port_idx = 0; port_idx < output.size(); ++port_idx) {
      const DataType type = output[port_idx].dtype();
      CHECK(type == DT_INT32 || type == DT_INT64);
      const PartialTensorShape shape(input[port_idx].shape());
      if (!shape.IsFullyDefined()) {
        continue;
      }
      Tensor constant_value(type);
      auto status = ConvertShapeToConstant(op, type, shape, &constant_value);
      if (!status.ok()) {
        continue;
      }

      // Find all nodes consuming this shape and connect them through the new
      // constant node instead.
      auto outputs = node_map_->GetOutputs(shape_n_node->name());
      for (NodeDef* output : outputs) {
        // Track whether there are any direct edges left between shape_n_node
        // and this output node after the transformation.
        bool direct_edges_exist = false;
        for (int k = 0; k < output->input_size(); ++k) {
          int port;
          const string node_name = ParseNodeName(output->input(k), &port);
          if (node_name == shape_n_node->name() && port == port_idx) {
            // Create a const node as ShapeN's output if not already.
            const string const_name = OptimizedNodeName(
                *shape_n_node, strings::StrCat("-matshapes-", port_idx));
            if (node_map_->GetNode(const_name) == nullptr) {
              NodeDef* added_node = graph_->add_node();
              added_node->set_name(const_name);
              added_node->set_op("Const");
              added_node->set_device(shape_n_node->device());
              node_map_->AddNode(added_node->name(), added_node);
              (*added_node->mutable_attr())["dtype"].set_type(type);
              constant_value.AsProtoTensorContent(
                  (*added_node->mutable_attr())["value"].mutable_tensor());
              // We add a control dependency to the original ShapeN node,
              // so that the node will only be run if all inputs of the
              // original ShapeN node are run.
              string ctrl_dep = AddControlDependency(shape_n_node->name(),
                                                     graph_, node_map_.get());
              *added_node->add_input() = ctrl_dep;
              node_map_->AddOutput(NodeName(ctrl_dep), added_node->name());
            }
            *output->mutable_input(k) = const_name;
            node_map_->AddOutput(const_name, output->name());
          }
          if (node_name == shape_n_node->name() && port != port_idx) {
            direct_edges_exist = true;
          }
        }
        if (!direct_edges_exist) {
          node_map_->RemoveOutput(node->name(), output->name());
        }
      }
    }
  }

  return Status::OK();
}

namespace {
bool ShapesEqual(const TensorShapeProto& shape1,
                 const TensorShapeProto& shape2) {
  if (shape1.unknown_rank() || shape2.unknown_rank()) {
    return false;
  }
  if (shape1.dim_size() != shape2.dim_size()) {
    return false;
  }
  for (int i = 0; i < shape1.dim_size(); ++i) {
    if (shape1.dim(i).size() != shape2.dim(i).size()) {
      return false;
    }
    if (shape1.dim(i).size() == -1 || shape2.dim(i).size() == -1) {
      return false;
    }
  }
  return true;
}

bool ExtractShape(const NodeDef& shape_node, const GraphProperties& properties,
                  BCast::Vec* shape, int64* min_id) {
  if (shape_node.op() == "Shape") {
    const std::vector<OpInfo::TensorProperties>& prop1 =
        properties.GetInputProperties(shape_node.name());
    if (prop1.size() != 1) {
      return false;
    }
    const TensorShapeProto& shp = prop1[0].shape();
    if (shp.unknown_rank()) {
      return false;
    }
    for (const auto& dim : shp.dim()) {
      shape->push_back(dim.size());
      *min_id = std::min<int64>(*min_id, dim.size());
    }
  } else {
    const TensorProto& raw_val = shape_node.attr().at("value").tensor();
    if (raw_val.dtype() != DT_INT64 && raw_val.dtype() != DT_INT32) {
      return false;
    }
    Tensor value(raw_val.dtype(), raw_val.tensor_shape());
    if (!value.FromProto(raw_val)) {
      return false;
    }
    for (int j = 0; j < value.NumElements(); ++j) {
      if (raw_val.dtype() == DT_INT64) {
        shape->push_back(value.vec<int64>()(j));
      } else {
        shape->push_back(value.vec<int>()(j));
      }
    }
  }
  return true;
}
}  // namespace

Status ConstantFolding::MaterializeBroadcastGradientArgs(
    const NodeDef& node, const GraphProperties& properties) {
  const NodeDef* shape_node1 = node_map_->GetNode(node.input(0));
  const NodeDef* shape_node2 = node_map_->GetNode(node.input(1));
  if (shape_node1 == nullptr ||
      (shape_node1->op() != "Shape" && !IsReallyConstant(*shape_node1)) ||
      shape_node2 == nullptr ||
      (shape_node2->op() != "Shape" && !IsReallyConstant(*shape_node2))) {
    return Status::OK();
  }

  // Don't optimize this again if it was already optimized and folded.
  if (OptimizedNodeExists(node, "-folded-1") ||
      OptimizedNodeExists(node, "-folded-2")) {
    return Status::OK();
  }
  int64 min_id = 0;
  BCast::Vec shape1;
  if (!ExtractShape(*shape_node1, properties, &shape1, &min_id)) {
    return Status::OK();
  }
  BCast::Vec shape2;
  if (!ExtractShape(*shape_node2, properties, &shape2, &min_id)) {
    return Status::OK();
  }
  // A value of -1 means we don't known anything about the dimension. Replace
  // the -1 values with unique dimension ids since we don't want two '-1'
  // dimensions to be considered equal.
  for (auto& id : shape1) {
    if (id == -1) {
      id = --min_id;
    }
  }
  for (auto& id : shape2) {
    if (id == -1) {
      id = --min_id;
    }
  }

  // Beware: the reduction dimensions computed by the BCast class are valid iff
  // we assume that two distinct symbolic dimensions can't be equal and a
  // symbolic dimension can't be equal to 1. This is often but not always true,
  // so to make this optimization safe we filter out these cases.
  const int common_dims = std::min(shape1.size(), shape2.size());
  for (int i = 0; i < common_dims; ++i) {
    if (shape1[i] >= 0 && shape2[i] >= 0) {
      continue;
    }
    if (shape1[i] != shape2[i]) {
      // We're either dealing with 2 different symbolic dimensions or a symbolic
      // and a know dimensions. We can't be sure whether both are equal or not,
      // so we can't be sure whether we'll be broadcasting or not.
      return Status::OK();
    }
  }
  // These extra dims could be equal to 1, in which case there is no
  // broadcasting. It could also be greater than 1, in which case there would
  // be broadcasting. Since we don't know, we'll just punt.
  for (int i = common_dims; i < shape1.size(); ++i) {
    if (shape1[i] < 0) {
      return Status::OK();
    }
  }
  for (int i = common_dims; i < shape2.size(); ++i) {
    if (shape2[i] < 0) {
      return Status::OK();
    }
  }

  BCast bcast(shape1, shape2);
  if (!bcast.IsValid()) {
    return Status::OK();
  }

  BCast::Vec reduce_dims[2];
  reduce_dims[0] = bcast.grad_x_reduce_idx();
  reduce_dims[1] = bcast.grad_y_reduce_idx();

  const DataType type = node.attr().at("T").type();
  NodeDef* out[2];
  for (int j = 0; j < 2; ++j) {
    int reduction_indices = reduce_dims[j].size();
    Tensor value(type, TensorShape({reduction_indices}));
    for (int i = 0; i < reduction_indices; ++i) {
      if (type == DT_INT32) {
        value.vec<int32>()(i) = reduce_dims[j][i];
      } else {
        value.vec<int64>()(i) = reduce_dims[j][i];
      }
    }
    string const_name =
        OptimizedNodeName(node, strings::StrCat("-bcastargs-", j));
    out[j] = node_map_->GetNode(const_name);
    if (out[j] == nullptr) {
      out[j] = graph_->add_node();
      TF_RETURN_IF_ERROR(
          CreateNodeDef(const_name, TensorValue(&value), out[j]));
      out[j]->set_device(node.device());
      node_map_->AddNode(const_name, out[j]);
      string ctrl_dep =
          AddControlDependency(node.name(), graph_, node_map_.get());
      *out[j]->add_input() = ctrl_dep;
      node_map_->AddOutput(NodeName(ctrl_dep), const_name);
    }
  }

  const std::set<NodeDef*> outputs = node_map_->GetOutputs(node.name());
  for (NodeDef* output : outputs) {
    for (int k = 0; k < output->input_size(); ++k) {
      int port;
      string node_name = ParseNodeName(output->input(k), &port);
      if (node_name == node.name() && port >= 0 && port < 2 && out[port]) {
        *output->mutable_input(k) = out[port]->name();
        node_map_->UpdateInput(output->name(), node_name, out[port]->name());
      }
    }
  }

  return Status::OK();
}

Status ConstantFolding::MaterializeReductionIndices(
    NodeDef* node, const GraphProperties& properties) {
  if (node->input_size() < 2) {
    return Status::OK();
  }
  const NodeDef* indices = node_map_->GetNode(node->input(1));
  if (!indices || IsReallyConstant(*indices)) {
    // The reduction indices are already constant, there's nothing to do.
    return Status::OK();
  }

  const std::vector<OpInfo::TensorProperties>& input_props =
      properties.GetInputProperties(node->name());
  if (input_props.size() != 2) {
    return Status::OK();
  }
  const OpInfo::TensorProperties& input_prop = input_props[0];
  if (input_prop.shape().unknown_rank()) {
    // We can't do anything if we don't know the rank of the input.
    return Status::OK();
  }
  const int rank = input_prop.shape().dim_size();
  if (rank == 0) {
    // Unexpected graph, don't try to change it.
    return Status::OK();
  }
  const std::vector<OpInfo::TensorProperties>& output_props =
      properties.GetOutputProperties(node->name());
  if (output_props.size() != 1) {
    return Status::OK();
  }
  const bool keep_dims =
      node->attr().count("keep_dims") && node->attr().at("keep_dims").b();
  const OpInfo::TensorProperties& output_prop = output_props[0];
  PartialTensorShape output_shape(output_prop.shape());
  if (output_shape.num_elements() != 1) {
    bool full_reduction = false;
    for (const NodeDef* fanout : node_map_->GetOutputs(node->name())) {
      if (!IsReshape(*fanout) && !keep_dims) {
        // Depending on how it's setup, a full reduction will generate a tensor
        // of shape [], [1], [1, 1], [1, 1, ...]. If keep_dims isn't true, we
        // rely on the existence of a reshape node following the reduction to
        // ensure that the fanout is fed a scalar of the right shape.
        return Status::OK();
      }
      const std::vector<OpInfo::TensorProperties>& reshape_props =
          properties.GetOutputProperties(fanout->name());
      if (reshape_props.size() != 1) {
        return Status::OK();
      }
      const OpInfo::TensorProperties& reshape_prop = reshape_props[0];
      PartialTensorShape shape(reshape_prop.shape());
      if (shape.num_elements() != 1) {
        return Status::OK();
      } else {
        full_reduction = true;
      }
    }
    if (!full_reduction) {
      return Status::OK();
    }
  }

  const OpInfo::TensorProperties& reduction_prop = input_props[1];
  DataType dtype = reduction_prop.dtype();
  if (dtype != DT_INT32 && dtype != DT_INT64) {
    return Status::OK();
  }
  // We know it's a full reduction. We can generate the set of indices to
  // reduce.
  string const_name = OptimizedNodeName(*node, "-reduction_indices");
  if (node_map_->GetNode(const_name)) {
    return Status::OK();
  }
  NodeDef* reduction_indices = graph_->add_node();
  Tensor value(dtype, TensorShape({rank}));
  for (int i = 0; i < rank; ++i) {
    if (dtype == DT_INT32) {
      value.vec<int32>()(i) = i;
    } else {
      value.vec<int64>()(i) = i;
    }
  }
  TF_RETURN_IF_ERROR(
      CreateNodeDef(const_name, TensorValue(&value), reduction_indices));
  reduction_indices->set_device(node->device());
  string ctrl_dep =
      AddControlDependency(node->input(1), graph_, node_map_.get());
  *reduction_indices->add_input() = ctrl_dep;
  node_map_->AddNode(const_name, reduction_indices);
  node_map_->AddOutput(NodeName(ctrl_dep), const_name);

  node->set_input(1, reduction_indices->name());
  node_map_->UpdateInput(node->name(), indices->name(),
                         reduction_indices->name());

  return Status::OK();
}

Status ConstantFolding::MaterializeConstants(
    const GraphProperties& properties) {
  const int node_count = graph_->node_size();
  for (int i = 0; i < node_count; ++i) {
    NodeDef& node = *graph_->mutable_node(i);
    const string& op = node.op();
    if (op == "BroadcastGradientArgs") {
      TF_RETURN_IF_ERROR(MaterializeBroadcastGradientArgs(node, properties));
    } else if (IsReduction(node)) {
      TF_RETURN_IF_ERROR(MaterializeReductionIndices(&node, properties));
    }
  }
  return Status::OK();
}

bool ConstantFolding::IsFoldable(const NodeDef& node) const {
  // Folding not applicable to ops with no inputs.
  if (node.input().empty()) {
    return false;
  }
  // Skips nodes that must be preserved except whitelisted nodes.
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end() &&
      nodes_whitelist_.find(node.name()) == nodes_whitelist_.end()) {
    return false;
  }
  // Skip control flow nodes, they can't be folded.
  if (ModifiesFrameInfo(node)) {
    return false;
  }
  // Skip constants, they're already folded
  if (IsConstant(node)) {
    return false;
  }

  // Don't fold stateful ops such as TruncatedNormal.
  if (!IsFreeOfSideEffect(node)) {
    return false;
  }

  // Skips ops that don't benefit from folding.
  if (IsPlaceholder(node)) {
    return false;
  }
  const string& op = node.op();
  if (op.find("Save") != string::npos || op.find("Restore") != string::npos ||
      op.find("Reader") != string::npos) {
    return false;
  }
  if (op.find("Quantized") != string::npos || op.find("Sparse") == 0) {
    return false;
  }

  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  // Don't fold ops without outputs.
  if (op_def->output_arg_size() == 0) {
    return false;
  }

  // No need to (and don't) fold nodes that have no outgoing edges except
  // whitelisted nodes. Such nodes could be introduced by an earlier constant
  // folding pass and are preserved in case users want to fetch their values;
  // re-processing them would lead to an error of adding a duplicated node
  // to graph.
  auto outputs = node_map_->GetOutputs(node.name());
  if (outputs.empty() &&
      nodes_whitelist_.find(node.name()) == nodes_whitelist_.end()) {
    return false;
  }

  // We can only fold nodes if all their inputs are known statically, except in
  // the case of a merge node that propagate the first inputs that becomes
  // available, and therefore only requires a single constant input to be
  // foldable.
  bool merge_has_constant_input = false;
  const bool is_merge = IsMerge(node);
  for (const auto& input : node.input()) {
    if (IsControlInput(input)) {
      continue;
    }
    const NodeDef* input_node = node_map_->GetNode(input);
    if (!input_node) {
      return false;
    }
    bool is_const = IsReallyConstant(*input_node);
    if (is_const) {
      // Don't fold strings constants for now since this causes problems with
      // checkpointing.
      if (input_node->attr().at("dtype").type() == DT_STRING) {
        return false;
      }
      // Special case: If a Merge node has at least one constant input that
      // does not depend on a control input, we can fold it.
      merge_has_constant_input |= !HasControlInputs(*input_node);
    } else if (!is_merge) {
      return false;
    }
  }
  return !is_merge || merge_has_constant_input;
}

namespace {

#define SET_TENSOR_VAL_CASE(DTYPE, TYPE, NAME)     \
  case DTYPE:                                      \
    t->add_##NAME##_val(static_cast<TYPE>(value)); \
    break;

Status CreateConstantTensorAttrValue(DataType type, double value,
                                     const TensorShapeProto& shape,
                                     AttrValue* attr_tensor) {
  TensorProto* t = attr_tensor->mutable_tensor();
  t->set_dtype(type);
  *t->mutable_tensor_shape() = shape;
  switch (type) {
    case DT_HALF:
      t->add_half_val(static_cast<Eigen::half>(value).x);
      break;
    case DT_BFLOAT16:
      t->add_half_val(static_cast<bfloat16>(value).value);
      break;
      SET_TENSOR_VAL_CASE(DT_FLOAT, float, float);
      SET_TENSOR_VAL_CASE(DT_DOUBLE, double, double);
      SET_TENSOR_VAL_CASE(DT_INT64, int64, int64);
      SET_TENSOR_VAL_CASE(DT_UINT64, int64, int64);
      SET_TENSOR_VAL_CASE(DT_INT32, int32, int);
      SET_TENSOR_VAL_CASE(DT_UINT32, int32, int);
      SET_TENSOR_VAL_CASE(DT_INT16, int32, int);
      SET_TENSOR_VAL_CASE(DT_UINT16, int32, int);
      SET_TENSOR_VAL_CASE(DT_INT8, int32, int);
      SET_TENSOR_VAL_CASE(DT_UINT8, int32, int);
      SET_TENSOR_VAL_CASE(DT_BOOL, bool, bool);
    default:
      return errors::InvalidArgument("Unsupported type: ", type);
  }
  return Status::OK();
}

#undef SET_TENSOR_CAL_CASE

DataType GetDataTypeFromNodeOrProps(const NodeDef& node,
                                    const GraphProperties& properties) {
  DataType dtype = DT_INVALID;
  if (node.attr().count("T") == 1) {
    dtype = node.attr().at("T").type();
  } else if (node.attr().count("dtype") == 1) {
    dtype = node.attr().at("dtype").type();
  } else if (IsLogicalOr(node) || IsLogicalAnd(node)) {
    dtype = DT_BOOL;
  } else {
    auto output_props = properties.GetOutputProperties(node.name());
    if (!output_props.empty()) {
      dtype = output_props[0].dtype();
    }
  }
  return dtype;
}

}  // namespace

// static
Status ConstantFolding::CreateNodeDef(const string& name,
                                      const TensorValue& tensor,
                                      NodeDef* node) {
  node->set_name(name);
  node->set_op("Const");

  AttrValue attr_type;
  attr_type.set_type(tensor->dtype());
  node->mutable_attr()->insert({"dtype", attr_type});

  AttrValue attr_tensor;
  TensorProto* t = attr_tensor.mutable_tensor();
  bool optimized = false;
  size_t encoded_size;
  // Use the packed representation whenever possible to avoid generating large
  // graphdefs. Moreover, avoid repeating the last values if they're equal.
  if (tensor->NumElements() > 4) {
#define POPULATE_TENSOR_PROTO(tensor, t, TYPE, NAME)                  \
  {                                                                   \
    const TYPE* val_ptr = tensor->flat<TYPE>().data();                \
    TYPE last = *val_ptr;                                             \
    int64 last_index = 0;                                             \
    for (int64 i = 0; i < tensor->NumElements(); ++i) {               \
      TYPE cur = *val_ptr++;                                          \
      if (cur != last) {                                              \
        last = cur;                                                   \
        last_index = i;                                               \
      }                                                               \
    }                                                                 \
    if (last_index < kint32max) {                                     \
      optimized = true;                                               \
      encoded_size = (last_index + 1) * sizeof(NAME);                 \
      t->mutable_##NAME##_val()->Reserve(last_index + 1);             \
      t->mutable_##NAME##_val()->AddNAlreadyReserved(last_index + 1); \
      val_ptr = tensor->flat<TYPE>().data();                          \
      for (int64 i = 0; i <= last_index; ++i) {                       \
        t->set_##NAME##_val(i, *val_ptr++);                           \
      }                                                               \
    }                                                                 \
  }                                                                   \
  break

    switch (tensor->dtype()) {
      case DT_FLOAT:
        POPULATE_TENSOR_PROTO(tensor, t, float, float);
      case DT_DOUBLE:
        POPULATE_TENSOR_PROTO(tensor, t, double, double);
      case DT_INT64:
        POPULATE_TENSOR_PROTO(tensor, t, int64, int64);
      case DT_UINT64:
        POPULATE_TENSOR_PROTO(tensor, t, uint64, int64);
      case DT_INT32:
        POPULATE_TENSOR_PROTO(tensor, t, int32, int);
      case DT_UINT32:
        POPULATE_TENSOR_PROTO(tensor, t, uint32, int);
      case DT_INT16:
        POPULATE_TENSOR_PROTO(tensor, t, int16, int);
      case DT_UINT16:
        POPULATE_TENSOR_PROTO(tensor, t, uint16, int);
      case DT_INT8:
        POPULATE_TENSOR_PROTO(tensor, t, int8, int);
      case DT_UINT8:
        POPULATE_TENSOR_PROTO(tensor, t, uint8, int);
      case DT_BOOL:
        POPULATE_TENSOR_PROTO(tensor, t, bool, bool);
      default:
        /* Do nothing. */
        break;
    }
  }
  if (optimized) {
    // Also specify type and shape.
    t->set_dtype(tensor->dtype());
    tensor->shape().AsProto(t->mutable_tensor_shape());
  } else {
    tensor->AsProtoTensorContent(t);
    encoded_size = t->tensor_content().size();
  }
  node->mutable_attr()->insert({"value", attr_tensor});

  if (encoded_size < 10 * 1024 * 1024) {
    return Status::OK();
  }
  return errors::InvalidArgument(
      strings::StrCat("Can't fold ", name, ", its size would be too large"));
}

Status ConstantFolding::EvaluateNode(const NodeDef& node,
                                     const TensorVector& inputs,
                                     TensorVector* output) const {
  Status status;
  auto op_kernel =
      CreateOpKernel("CPU", cpu_device_, cpu_device_->GetAllocator({}), node,
                     TF_GRAPH_DEF_VERSION, &status);
  TF_RETURN_IF_ERROR(status);
  OpKernelContext::Params params;
  params.device = cpu_device_;
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op_kernel.get();
  params.resource_manager = resource_mgr_.get();

  gtl::InlinedVector<AllocatorAttributes, 4> output_attrs;
  const int num_outputs = op_kernel->num_outputs();
  for (int i = 0; i < num_outputs; i++) {
    AllocatorAttributes attr;
    attr.set_on_host(true);
    output_attrs.push_back(attr);
  }
  params.output_attr_array = output_attrs.data();

  OpKernelContext op_context(&params);
  op_kernel->Compute(&op_context);
  for (int i = 0; i < num_outputs; i++) {
    output->push_back(op_context.release_output(i));
  }
  return op_context.status();
}

Status ConstantFolding::EvaluateOneFoldable(const NodeDef& node,
                                            std::vector<NodeDef>* outputs) {
  TensorVector inputs;
  TensorVector output_tensors;
  auto inputs_cleanup = gtl::MakeCleanup([&inputs, &output_tensors] {
    for (const auto& input : inputs) {
      delete input.tensor;
    }
    for (const auto& output : output_tensors) {
      if (output.tensor) {
        delete output.tensor;
      }
    }
  });

  for (const auto& input : node.input()) {
    int port = 0;
    ParseNodeNameAsStringPiece(input, &port);
    if (port < 0) {
      // Control dependency
      break;
    }
    const NodeDef* input_node = node_map_->GetNode(input);
    if (!IsReallyConstant(*input_node)) {
      return Status(error::INVALID_ARGUMENT,
                    strings::StrCat("Can't fold ", node.name(), ", its ", input,
                                    " isn't constant"));
    }
    const TensorProto& raw_val = input_node->attr().at("value").tensor();
    Tensor* value = new Tensor(raw_val.dtype(), raw_val.tensor_shape());
    CHECK(value->FromProto(raw_val));
    inputs.emplace_back(value);
  }

  TF_RETURN_IF_ERROR(EvaluateNode(node, inputs, &output_tensors));
  if (output_tensors.empty()) {
    return Status(error::INVALID_ARGUMENT, "Expected at least one output.");
  }

  outputs->resize(output_tensors.size());
  for (size_t i = 0; i < output_tensors.size(); i++) {
    string node_name = OptimizedNodeName(node, "-folded");
    if (output_tensors.size() > 1) {
      node_name = strings::StrCat(node_name, "-", i);
    }
    if (output_tensors[i].tensor) {
      TF_RETURN_IF_ERROR(
          CreateNodeDef(node_name, output_tensors[i], &outputs->at(i)));
    } else {
      // Create an empty NodeDef to identify dead outputs (e.g. the output of a
      // switch that's not selected by the switch predicate).
      outputs->at(i) = NodeDef();
    }
  }
  return Status::OK();
}

Status ConstantFolding::FoldNode(NodeDef* node, GraphDef* output_graph) {
  if (IsMerge(*node)) {
    // Merge nodes are special, in the sense that they execute as soon as one of
    // their input is ready. We can therefore fold a merge node iff it has at
    // least one constant input without control dependency.
    // We still need to ensure that the nodes in the fanin of the merge node are
    // scheduled. We'll therefore add a control dependency from the merge node
    // to the folded constant. We end up with:
    //  * the merge node and its inputs are preserved as is
    //  * a new constant node C1, driven by the merge node through a control
    //  dependency, initialized to the value of the folded input
    //  * a new constant node C2, driven by the merge node through a control
    //  dependency, initialized to the index of the folded input
    //  * the fanout of the merge nodes is rewired to be driven by either C1 or
    //  C2.
    for (int input_index = 0; input_index < node->input_size(); ++input_index) {
      const auto& input = node->input(input_index);
      if (IsControlInput(input)) {
        // Try the next input.
        continue;
      }
      NodeDef* input_node = node_map_->GetNode(input);
      if (!IsReallyConstant(*input_node)) {
        continue;
      }
      bool valid_input = true;
      for (const string& fanin_of_input : input_node->input()) {
        if (IsControlInput(fanin_of_input)) {
          valid_input = false;
          break;
        }
      }
      if (!valid_input) {
        // Try the next input
        continue;
      }

      string const_out_name = OptimizedNodeName(*node, "_const");
      string const_index_name = OptimizedNodeName(*node, "_index");
      if (node_map_->GetNode(const_out_name) ||
          node_map_->GetNode(const_index_name)) {
        // Intended name already exists.
        return errors::AlreadyExists(
            strings::StrCat(const_out_name, " or ", const_index_name,
                            " already present in the graph"));
      }

      NodeDef* const_out = output_graph->add_node();
      *const_out = *input_node;
      const_out->set_name(const_out_name);
      const_out->set_device(node->device());
      *const_out->add_input() = AsControlDependency(*node);
      node_map_->AddNode(const_out->name(), const_out);
      node_map_->AddOutput(node->name(), const_out->name());

      NodeDef* const_index = output_graph->add_node();
      const_index->set_op("Const");
      Tensor index(DT_INT32, TensorShape({}));
      index.flat<int32>()(0) = input_index;
      (*const_index->mutable_attr())["dtype"].set_type(DT_INT32);
      index.AsProtoTensorContent(
          (*const_index->mutable_attr())["value"].mutable_tensor());
      const_index->set_name(const_index_name);
      const_index->set_device(node->device());
      *const_index->add_input() = AsControlDependency(*node);
      node_map_->AddNode(const_index->name(), const_index);
      node_map_->AddOutput(node->name(), const_index->name());

      auto outputs = node_map_->GetOutputs(node->name());
      for (NodeDef* output : outputs) {
        for (int i = 0; i < output->input_size(); i++) {
          int port;
          string node_name = ParseNodeName(output->input(i), &port);
          if (node_name == node->name()) {
            if (port == 0) {
              *output->mutable_input(i) = const_out->name();
              node_map_->AddOutput(const_out->name(), output->name());
            } else if (port == 1) {
              *output->mutable_input(i) = const_index->name();
              node_map_->AddOutput(const_index->name(), output->name());
            } else {
              // This is a control dependency (or an invalid edge since the
              // merge node has only 2 inputs): preserve them.
            }
          }
        }
      }
      return Status::OK();
    }
    return Status::OK();
  }

  std::vector<NodeDef> const_nodes;
  TF_RETURN_IF_ERROR(EvaluateOneFoldable(*node, &const_nodes));
  NodeDef* constant_output = nullptr;
  for (int i = 0; i < const_nodes.size(); i++) {
    NodeDef* const_node = &const_nodes[i];
    if (const_node->name().empty()) {
      // Dead output: we can't create a constant to encode its value, so we'll
      // just skip it. We'll preserve the edges that originate from that
      // output below to preserve the overall behavior of the graph wrt dead
      // edges.
      continue;
    }

    // Forward control dependencies.
    for (const auto& input : node->input()) {
      if (IsControlInput(input) &&
          std::find(const_node->input().begin(), const_node->input().end(),
                    input) == const_node->input().end()) {
        *const_node->add_input() = input;
      } else {
        NodeDef* input_node = node_map_->GetNode(input);
        for (const auto& fanin_of_input : input_node->input()) {
          if (IsControlInput(fanin_of_input) &&
              std::find(const_node->input().begin(), const_node->input().end(),
                        fanin_of_input) == const_node->input().end()) {
            *const_node->add_input() = fanin_of_input;
          }
        }
      }
    }

    // We rewrite the existing node if it only has a single output, and
    // create new nodes otherwise.
    if (const_nodes.size() == 1) {
      node->set_op("Const");
      // Note we need to clear the inputs in NodeMap before we clear the inputs
      // in the node, otherwise NodeMap would see empty inputs and effectively
      // does nothing.
      node_map_->RemoveInputs(node->name());
      node->clear_input();
      *node->mutable_input() = const_node->input();
      for (const auto& input : node->input()) {
        node_map_->AddOutput(NodeName(input), node->name());
      }
      *node->mutable_attr() = const_node->attr();
      break;
    } else {
      if (node_map_->GetNode(const_node->name())) {
        // Intended name already exists.
        return errors::AlreadyExists(strings::StrCat(
            const_node->name(), " already present in the graph"));
      }
      NodeDef* added_node = output_graph->add_node();
      *added_node = *const_node;
      added_node->set_device(node->device());
      node_map_->AddNode(added_node->name(), added_node);
      for (const auto& input : added_node->input()) {
        node_map_->AddOutput(NodeName(input), added_node->name());
      }
      // All the constant nodes encoding output values have the same control
      // dependencies (since these are the control dependencies of the node
      // we're trying to fold). Record one such constant node.
      constant_output = added_node;
    }
  }

  if (const_nodes.size() > 1) {
    auto outputs = node_map_->GetOutputs(node->name());
    for (NodeDef* output : outputs) {
      for (int i = 0; i < output->input_size(); i++) {
        int port;
        string node_name = ParseNodeName(output->input(i), &port);
        if (node_name == node->name()) {
          if (port < 0) {
            // Propagate control dependencies if possible. If not, we'll just
            // preserve the existing control dependencies.
            if (constant_output != nullptr) {
              node_map_->UpdateInput(node_name, NodeName(output->input(i)),
                                     constant_output->name());
              *output->mutable_input(i) = AsControlDependency(*constant_output);
            }
          } else if (port < const_nodes.size() &&
                     !const_nodes[port].name().empty()) {
            // Replace alive outputs with the corresponding constant.
            node_map_->UpdateInput(output->name(), NodeName(output->input(i)),
                                   const_nodes[port].name());
            *output->mutable_input(i) = const_nodes[port].name();
          } else {
            // Leave this edge alone.
            VLOG(1) << "Preserving edge from " << node->name() << ":" << port
                    << "[" << node->op() << "] to " << output->name() << ":"
                    << i << "[" << output->op() << "]";
          }
        }
      }
    }
    outputs = node_map_->GetOutputs(node->name());
    if (outputs.empty() && has_fetch_ &&
        nodes_to_preserve_.find(node->name()) == nodes_to_preserve_.end()) {
      node_map_->RemoveInputs(node->name());
      node->clear_input();
    }
  }
  return Status::OK();
}

Status ConstantFolding::FoldGraph(GraphDef* output) {
  std::unordered_set<string> processed_nodes;
  std::deque<NodeDef*> queue;
  for (int i = 0; i < graph_->node_size(); i++) {
    if (IsFoldable(graph_->node(i))) {
      queue.push_back(graph_->mutable_node(i));
    }
  }
  while (!queue.empty()) {
    NodeDef* node = queue.front();
    queue.pop_front();
    if (processed_nodes.count(node->name())) {
      continue;
    }
    // We need to record a copy of output nodes before FoldNode() modifies it.
    // We also need to ensure that the fanout is sorted deterministically.
    const std::set<NodeDef*>& outputs = node_map_->GetOutputs(node->name());
    std::vector<NodeDef*> fanout(outputs.begin(), outputs.end());
    std::sort(fanout.begin(), fanout.end(),
              [](const NodeDef* n1, const NodeDef* n2) {
                return n1->name() < n2->name();
              });

    Status s = FoldNode(node, output);
    processed_nodes.insert(node->name());
    if (!s.ok()) {
      VLOG(1) << "Failed to fold node " << node->DebugString()
              << "\nError message: " << s;
    } else {
      for (auto& output : fanout) {
        if (IsFoldable(*output)) {
          queue.push_back(output);
        }
      }
    }
  }

  // Delete the newly created nodes that don't feed anything.
  int last = output->node_size() - 1;
  for (int i = output->node_size() - 1; i >= 0; --i) {
    const NodeDef& node = output->node(i);
    auto fanout = node_map_->GetOutputs(node.name());
    if (fanout.empty()) {
      output->mutable_node()->SwapElements(i, last);
      last--;
    }
  }
  output->mutable_node()->DeleteSubrange(last + 1,
                                         output->node_size() - last - 1);

  for (const auto& node : graph_->node()) {
    // If no fetch nodes is provided, we conservatively
    // keep all nodes in the original graph in case users need to fetch
    // their values.
    auto fanout = node_map_->GetOutputs(node.name());
    if (!fanout.empty() || !has_fetch_ ||
        nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
      auto added_node = output->add_node();
      *added_node = node;
    }
  }
  return Status::OK();
}

// Returns true iff this reduction can be reduced to an identity (i.e if the set
// of dimensions to reduce along is empty). This happens often in the gradient
// graphs.
bool ConstantFolding::IsSimplifiableReduction(const NodeDef& node) const {
  if (IsReduction(node)) {
    CHECK_LE(2, node.input_size());
    const NodeDef* reductions_indices = node_map_->GetNode(node.input(1));
    if (IsReallyConstant(*reductions_indices)) {
      TensorVector output;
      Status s = EvaluateNode(*reductions_indices, TensorVector(), &output);
      if (!s.ok()) {
        return false;
      }
      CHECK_EQ(1, output.size());
      int output_size = output[0]->NumElements();
      delete output[0].tensor;
      if (output_size == 0) {
        return true;
      }
    }
  }
  return false;
}

bool ConstantFolding::IsSimplifiableReshape(
    const NodeDef& node, const GraphProperties& properties) const {
  if (!IsReshape(node)) {
    return false;
  }
  CHECK_LE(2, node.input_size());
  const NodeDef* new_shape = node_map_->GetNode(node.input(1));
  if (!IsReallyConstant(*new_shape)) {
    return false;
  }
  TensorVector outputs;
  auto outputs_cleanup = gtl::MakeCleanup([&outputs] {
    for (const auto& output : outputs) {
      delete output.tensor;
    }
  });

  Status s = EvaluateNode(*new_shape, TensorVector(), &outputs);
  if (!s.ok()) {
    return false;
  }
  CHECK_EQ(1, outputs.size());

  const std::vector<OpInfo::TensorProperties>& props =
      properties.GetInputProperties(node.name());
  if (props.empty()) {
    return false;
  }
  const OpInfo::TensorProperties& prop = props[0];
  if (prop.dtype() == DT_INVALID) {
    return false;
  }
  const PartialTensorShape shape(prop.shape());
  if (!shape.IsFullyDefined()) {
    return false;
  }

  PartialTensorShape new_dims;
  if (outputs[0]->dtype() == DT_INT32) {
    std::vector<int32> shp;
    for (int i = 0; i < outputs[0]->NumElements(); ++i) {
      int32 dim = outputs[0]->flat<int32>()(i);
      shp.push_back(dim);
    }
    TF_CHECK_OK(TensorShapeUtils::MakeShape(shp, &new_dims));
  } else {
    std::vector<int64> shp;
    for (int i = 0; i < outputs[0]->NumElements(); ++i) {
      int64 dim = outputs[0]->flat<int64>()(i);
      shp.push_back(dim);
    }
    TF_CHECK_OK(TensorShapeUtils::MakeShape(shp, &new_dims));
  }

  return shape.IsCompatibleWith(new_dims);
}

#define IS_VALUE_CASE(DTYPE, VALUE)                   \
  case DTYPE:                                         \
    return AllValuesAre<EnumToDataType<DTYPE>::Type>( \
        node.attr().at("value").tensor(), EnumToDataType<DTYPE>::Type(VALUE))

#define IS_ONES_CASE(TYPE) IS_VALUE_CASE(TYPE, 1)
#define IS_ZEROS_CASE(TYPE) IS_VALUE_CASE(TYPE, 0)

bool ConstantFolding::IsOnes(const NodeDef& node) const {
  if (feed_nodes_.find(node.name()) != feed_nodes_.end()) {
    return false;
  }
  if (node.op() == "OnesLike") {
    return true;
  }
  if (node.op() == "Fill") {
    NodeDef* values = node_map_->GetNode(NodeName(node.input(1)));
    return values != nullptr && IsOnes(*values);
  }
  if (node.op() != "Const") {
    return false;
  }
  const auto dtype = node.attr().at("dtype").type();
  switch (dtype) {
    IS_ONES_CASE(DT_BOOL);
    IS_ONES_CASE(DT_HALF);
    IS_ONES_CASE(DT_BFLOAT16);
    IS_ONES_CASE(DT_FLOAT);
    IS_ONES_CASE(DT_DOUBLE);
    IS_ONES_CASE(DT_COMPLEX64);
    IS_ONES_CASE(DT_COMPLEX128);
    IS_ONES_CASE(DT_UINT8);
    IS_ONES_CASE(DT_INT8);
    IS_ONES_CASE(DT_UINT16);
    IS_ONES_CASE(DT_INT16);
    IS_ONES_CASE(DT_INT32);
    IS_ONES_CASE(DT_INT64);
    default:
      VLOG(1) << "Unsupported type " << DataTypeString(dtype);
      return false;
  }
  return false;
}

bool ConstantFolding::IsZeros(const NodeDef& node) const {
  if (feed_nodes_.find(node.name()) != feed_nodes_.end()) {
    return false;
  }
  if (node.op() == "ZerosLike") {
    return true;
  }
  if (node.op() == "Fill") {
    NodeDef* values = node_map_->GetNode(NodeName(node.input(1)));
    return values != nullptr && IsZeros(*values);
  }
  if (!IsConstant(node)) {
    return false;
  }
  const auto dtype = node.attr().at("dtype").type();
  switch (dtype) {
    IS_ZEROS_CASE(DT_BOOL);
    IS_ZEROS_CASE(DT_HALF);
    IS_ZEROS_CASE(DT_BFLOAT16);
    IS_ZEROS_CASE(DT_FLOAT);
    IS_ZEROS_CASE(DT_DOUBLE);
    IS_ZEROS_CASE(DT_COMPLEX64);
    IS_ZEROS_CASE(DT_COMPLEX128);
    IS_ZEROS_CASE(DT_UINT8);
    IS_ZEROS_CASE(DT_INT8);
    IS_ZEROS_CASE(DT_UINT16);
    IS_ZEROS_CASE(DT_INT16);
    IS_ZEROS_CASE(DT_INT32);
    IS_ZEROS_CASE(DT_INT64);
    default:
      VLOG(1) << "Unsupported type " << DataTypeString(dtype);
      return false;
  }
  return false;
}

void ConstantFolding::ReplaceOperationWithIdentity(
    int input_to_forward, const GraphProperties& properties, NodeDef* node,
    GraphDef* graph) {
  const DataType dtype = GetDataTypeFromNodeOrProps(*node, properties);
  if (dtype == DT_INVALID) return;

  node->set_op("Identity");
  node->clear_attr();
  (*node->mutable_attr())["T"].set_type(dtype);
  // Propagate the designated input through the identity.
  node->mutable_input()->SwapElements(0, input_to_forward);
  // Add all other inputs as control dependencies.
  for (int i = 1; i < node->input_size(); ++i) {
    if (IsControlInput(node->input(i))) {
      break;
    }
    const string ctrl_dep =
        AddControlDependency(node->input(i), graph, node_map_.get());
    node_map_->UpdateInput(node->name(), node->input(i), ctrl_dep);
    node->set_input(i, ctrl_dep);
  }
  graph_modified_ = true;
}

void ConstantFolding::ReplaceOperationWithSnapshot(
    int input_to_forward, const GraphProperties& properties, NodeDef* node,
    GraphDef* graph) {
  const DataType dtype = GetDataTypeFromNodeOrProps(*node, properties);
  if (dtype == DT_INVALID) return;

  node->set_op("Snapshot");
  node->clear_attr();
  (*node->mutable_attr())["T"].set_type(dtype);
  // Propagate the designated input through the Snapshot.
  node->mutable_input()->SwapElements(0, input_to_forward);
  // Add all other inputs as control dependencies.
  for (int i = 1; i < node->input_size(); ++i) {
    if (IsControlInput(node->input(i))) {
      break;
    }
    const string ctrl_dep =
        AddControlDependency(node->input(i), graph, node_map_.get());
    node_map_->UpdateInput(node->name(), node->input(i), ctrl_dep);
    node->set_input(i, ctrl_dep);
  }
  graph_modified_ = true;
}

void ConstantFolding::ReplaceDivisionOfOnesByReciprocal(NodeDef* node,
                                                        GraphDef* graph) {
  node->set_op("Reciprocal");
  node->mutable_input()->SwapElements(0, 1);
  const string ctrl_dep =
      AddControlDependency(node->input(1), graph, node_map_.get());
  node_map_->UpdateInput(node->name(), node->input(1), ctrl_dep);
  node->set_input(1, ctrl_dep);
  graph_modified_ = true;
}

void ConstantFolding::ReplaceSubtractionFromZeroByNegation(NodeDef* node,
                                                           GraphDef* graph) {
  node->set_op("Neg");
  node->mutable_input()->SwapElements(0, 1);
  const string ctrl_dep =
      AddControlDependency(node->input(1), graph, node_map_.get());
  node_map_->UpdateInput(node->name(), node->input(1), ctrl_dep);
  node->set_input(1, ctrl_dep);
  graph_modified_ = true;
}

Status ConstantFolding::ReplaceOperationWithConstant(
    double value, const GraphProperties& properties,
    const TensorShapeProto& shape, NodeDef* node, GraphDef* graph) {
  const DataType dtype = GetDataTypeFromNodeOrProps(*node, properties);
  if (dtype == DT_INVALID) return Status::OK();

  AttrValue tensor_attr;
  TF_RETURN_IF_ERROR(
      CreateConstantTensorAttrValue(dtype, value, shape, &tensor_attr));
  node->set_op("Const");
  node->clear_attr();
  (*node->mutable_attr())["dtype"].set_type(dtype);
  node->mutable_attr()->insert({"value", tensor_attr});
  // Convert all inputs to control dependencies.
  for (int i = 0; i < node->input_size(); ++i) {
    if (IsControlInput(node->input(i))) {
      break;
    }
    const string ctrl_dep =
        AddControlDependency(node->input(i), graph, node_map_.get());
    node_map_->UpdateInput(node->name(), node->input(i), ctrl_dep);
    node->set_input(i, ctrl_dep);
  }
  graph_modified_ = true;
  return Status::OK();
}

Status ConstantFolding::SimplifyGraph(GraphDef* optimized_graph,
                                      GraphProperties* properties,
                                      bool use_shape_info) {
  const bool is_aggressive = opt_level_ == RewriterConfig::AGGRESSIVE;
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    NodeDef* node = optimized_graph->mutable_node(i);

    if (IsSplit(*node) && node->attr().at("num_split").i() == 1) {
      ReplaceOperationWithIdentity(1, *properties, node, optimized_graph);
      continue;
    }

    if (IsSplitV(*node) && node->attr().at("num_split").i() == 1) {
      ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
      continue;
    }

    // Remove Shuffle or Transpose op over dimensions of size 1.
    if (use_shape_info && (IsShuffle(*node) || IsTranspose(*node)) &&
        properties->GetInputProperties(node->name()).size() >= 2) {
      const auto& shape =
          properties->GetInputProperties(node->name())[0].shape();
      if (shape.unknown_rank()) {
        // Not optimizable.
        continue;
      }
      const auto& p = properties->GetInputProperties(node->name())[1];
      if (TensorShape::IsValid(p.shape()) && p.has_value()) {
        Tensor perm(p.dtype(), p.shape());
        if (!perm.FromProto(p.value())) {
          return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                         p.value().DebugString());
        }
        std::vector<int> permutation;
        for (int j = 0; j < perm.NumElements(); ++j) {
          if (perm.dtype() == DT_INT64) {
            permutation.push_back(perm.vec<int64>()(j));
          } else {
            permutation.push_back(perm.vec<int>()(j));
          }
        }
        if (permutation.size() != shape.dim_size()) {
          // Number of elements in perm should be same as dim_size. Skip if not.
          continue;
        }
        // The node is replaceable iff
        // dim_size == 0 || all dims have size 1 ||
        // all dims with > 1 size are not permuted.
        bool replaceable = true;
        for (int j = 0; replaceable && j < shape.dim_size(); ++j) {
          replaceable &= shape.dim(j).size() == 1 || j == permutation[j];
        }
        if (replaceable) {
          ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
          continue;
        }
      }
    }

    // Remove RandomShuffle op if it is scalar or first dimension is of size 1.
    if (use_shape_info && IsRandomShuffle(*node) &&
        !properties->GetInputProperties(node->name()).empty()) {
      const auto& shape =
          properties->GetInputProperties(node->name())[0].shape();
      // The node is replaceable iff
      // unknown_rank == false && (dim_size == 0 || first dim is of size 1)
      if (!shape.unknown_rank() &&
          (shape.dim_size() == 0 || shape.dim(0).size() == 1)) {
        ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
        continue;
      }
    }

    // Remove Reverse op over dimensions with size 1.
    if (use_shape_info && node->op() == "ReverseV2" &&
        properties->GetInputProperties(node->name()).size() >= 2) {
      const auto& shape =
          properties->GetInputProperties(node->name())[0].shape();
      if (shape.unknown_rank()) {
        // Not optimizable.
        continue;
      }
      const auto& a = properties->GetInputProperties(node->name())[1];
      if (TensorShape::IsValid(a.shape()) && a.has_value()) {
        Tensor axis(a.dtype(), a.shape());
        if (!axis.FromProto(a.value())) {
          return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                         a.value().DebugString());
        }
        std::set<int> target_axes;
        for (int j = 0; j < axis.NumElements(); ++j) {
          // value of axis can be negative.
          if (axis.dtype() == DT_INT64) {
            target_axes.insert((axis.vec<int64>()(j) + shape.dim_size()) %
                               shape.dim_size());
          } else {
            target_axes.insert((axis.vec<int>()(j) + shape.dim_size()) %
                               shape.dim_size());
          }
        }

        // The node is replaceable iff
        // unknown_rank == false &&
        // (dim_size == 0 || all dims have size 1 ||
        //  all dims with > 1 size are not in target_axes)
        bool replaceable = !shape.unknown_rank();
        for (int j = 0; replaceable && j < shape.dim_size(); ++j) {
          replaceable &= shape.dim(j).size() == 1 ||
                         target_axes.find(j) == target_axes.end();
        }
        if (replaceable) {
          ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
          continue;
        }
      }
    }

    if (use_shape_info && IsSlice(*node) &&
        properties->GetInputProperties(node->name()).size() == 3) {
      const auto& input = properties->GetInputProperties(node->name())[0];
      const auto& b = properties->GetInputProperties(node->name())[1];
      const auto& s = properties->GetInputProperties(node->name())[2];
      if (TensorShape::IsValid(b.shape()) && b.has_value() &&
          TensorShape::IsValid(s.shape()) && s.has_value()) {
        Tensor begin(b.dtype(), b.shape());
        if (!begin.FromProto(b.value())) {
          return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                         b.value().DebugString());
        }
        Tensor size(s.dtype(), s.shape());
        if (!size.FromProto(s.value())) {
          return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                         s.value().DebugString());
        }
        // The node is replaceable iff unknown_rank == false &&
        // begin == 0 && (size == -1 || size == input_shape) for all dimensions
        bool replaceable = !input.shape().unknown_rank();
        for (int j = 0; replaceable && j < input.shape().dim_size(); ++j) {
          if (begin.dtype() == DT_INT32) {
            replaceable &= begin.vec<int>()(j) == 0;
          } else {
            replaceable &= begin.vec<int64>()(j) == 0;
          }
          if (size.dtype() == DT_INT32) {
            replaceable &= (size.vec<int>()(j) == -1 ||
                            size.vec<int>()(j) == input.shape().dim(j).size());
          } else {
            replaceable &=
                (size.vec<int64>()(j) == -1 ||
                 size.vec<int64>()(j) == input.shape().dim(j).size());
          }
        }
        if (replaceable) {
          ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
          continue;
        }
      }
    }

    if (use_shape_info && IsTile(*node) &&
        properties->GetInputProperties(node->name()).size() == 2) {
      const auto& m = properties->GetInputProperties(node->name())[1];
      if (TensorShape::IsValid(m.shape()) && m.has_value()) {
        Tensor multiplies(m.dtype(), m.shape());
        if (!multiplies.FromProto(m.value())) {
          return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                         m.value().DebugString());
        }
        // The node is replaceable iff all values in multiplies are 1.
        bool replaceable = true;
        if (multiplies.dtype() == DT_INT32) {
          for (int j = 0; replaceable && j < multiplies.vec<int>().size();
               ++j) {
            replaceable &= multiplies.vec<int>()(j) == 1;
          }
        } else {
          for (int j = 0; replaceable && j < multiplies.vec<int64>().size();
               ++j) {
            replaceable &= multiplies.vec<int64>()(j) == 1;
          }
        }
        if (replaceable) {
          ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
          continue;
        }
      }
    }

    if (use_shape_info && IsPad(*node) &&
        properties->GetInputProperties(node->name()).size() >= 2) {
      const auto& p = properties->GetInputProperties(node->name())[1];
      if (TensorShape::IsValid(p.shape()) && p.has_value()) {
        Tensor paddings(p.dtype(), p.shape());
        if (!paddings.FromProto(p.value())) {
          return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                         p.value().DebugString());
        }
        // The node is replaceable iff all values in paddings are 0.
        bool replaceable = true;
        // The operation requires it to be int32 value so we don't check for
        // 1nt64.
        const auto flatten = paddings.flat<int32>();
        for (int j = 0; replaceable && j < flatten.size(); ++j) {
          replaceable &= flatten(j) == 0;
        }
        if (replaceable) {
          ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
          continue;
        }
      }
    }

    if (use_shape_info && IsSqueeze(*node) &&
        !properties->GetInputProperties(node->name()).empty()) {
      // https://www.tensorflow.org/api_docs/python/tf/squeeze mentions it's
      // error to squeeze a dimension that is not 1, so we only need to check
      // whether the input has > 1 size for each dimension.
      const auto& shape =
          properties->GetInputProperties(node->name())[0].shape();
      // The node is replaceable iff
      // unknown_rank == false && (dim_size == 0 || all dims have size > 1)
      bool replaceable = !shape.unknown_rank();
      for (int j = 0; replaceable && j < shape.dim_size(); ++j) {
        replaceable &= shape.dim(j).size() > 1;
      }
      if (replaceable) {
        ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
        continue;
      }
    }

    if (IsPack(*node) && NumNonControlInputs(*node) == 1 &&
        !OptimizedNodeExists(*node, "_const_axis")) {
      // Create constant axis node.
      Tensor axis_t(DT_INT32, TensorShape({}));
      NodeDef* axis_node = optimized_graph->add_node();
      axis_node->set_name(OptimizedNodeName(*node, "_const_axis"));
      const int axis = node->attr().at("axis").i();
      if (!SetTensorValue(DT_INT32, axis, &axis_t).ok() ||
          !CreateNodeDef(axis_node->name(), TensorValue(&axis_t), axis_node)
               .ok()) {
        continue;
      }
      // Add a control dependency to make sure axis_node is in the right frame.
      const string ctrl_dep = ConstantFolding::AddControlDependency(
          node->input(0), graph_, node_map_.get());
      axis_node->add_input(ctrl_dep);
      axis_node->set_device(node->device());
      node->set_op("ExpandDims");
      if (node->attr().count("axis") != 0) {
        node->mutable_attr()->erase("axis");
      }
      if (node->attr().count("N") != 0) {
        node->mutable_attr()->erase("N");
      }
      (*node->mutable_attr())["Tdim"].set_type(DT_INT32);
      node->add_input(axis_node->name());
      if (node->input_size() > 2) {
        node->mutable_input()->SwapElements(1, node->input_size() - 1);
      }
      graph_modified_ = true;
      continue;
    }

    // Move constants past Enter.
    if (IsEnter(*node) && node->input_size() > 0) {
      if (node->attr().count("is_constant") == 0 ||
          !node->attr().at("is_constant").b()) {
        continue;
      }
      const string& node_name = node->name();
      const NodeDef* input = node_map_->GetNode(node->input(0));
      if (input != nullptr && IsReallyConstant(*input) &&
          !OptimizedNodeExists(*input, "_enter")) {
        auto fanouts = node_map_->GetOutputs(node_name);
        // Find non-constant nodes that consume the output of *node.
        std::vector<NodeDef*> consumers;
        for (NodeDef* fanout : fanouts) {
          if (!IsConstant(*fanout)) {
            for (int i = 0; i < fanout->input_size(); ++i) {
              if (fanout->input(i) == node_name) {
                consumers.push_back(fanout);
                break;
              }
            }
          }
        }
        if (!consumers.empty()) {
          NodeDef* new_node = optimized_graph->add_node();
          *new_node = *input;
          new_node->set_name(OptimizedNodeName(*input, "_enter"));
          new_node->set_device(node->device());
          new_node->clear_input();
          new_node->add_input(AsControlDependency(node_name));
          node_map_->AddNode(new_node->name(), new_node);
          node_map_->AddOutput(node_name, new_node->name());
          for (NodeDef* consumer : consumers) {
            for (int i = 0; i < consumer->input_size(); ++i) {
              if (NodeName(consumer->input(i)) == node_name) {
                node_map_->UpdateInput(consumer->name(), node_name,
                                       new_node->name());
                consumer->set_input(i, new_node->name());
              }
            }
          }
          graph_modified_ = true;
          continue;
        }
      }
    }

    // Switch(x, x) will always feed false to its false branch and true to
    // its true branch. By rewriting the graph a bit, we can propagate these
    // constants down the two output branches, and just use control dependencies
    // to trigger the selected one at runtime. For example,
    //
    //     +------+
    // x-->|Switch|-->a  (in practice there may be multiple consumers of each
    // x-->|      |-->b   output branch.)
    //     +------+
    //
    // Is rewritten as
    //
    //     +------+
    // x-->|Switch|-->Identity--^>Const(false)-->a
    // x-->|      |-->Identity--^>Const(true)-->b
    //     +------+
    if (node->op() == "Switch" && node->input(0) == node->input(1) &&
        !OptimizedNodeExists(*node, "_const_false") &&
        !OptimizedNodeExists(*node, "_const_true")) {
      bool already_optimized = true;
      // If the optimization was already applied, the switch would have exactly
      // one Identity node consuming each of its outputs, each without any
      // non-control outputs.
      auto fanouts = node_map_->GetOutputs(node->name());
      if (fanouts.size() == 2) {
        for (NodeDef* fanout : fanouts) {
          if (!IsIdentity(*fanout) ||
              NumNonControlOutputs(*fanout, *node_map_) > 0) {
            already_optimized = false;
            break;
          }
        }
      }
      Tensor false_t(DT_BOOL, TensorShape({}));
      Tensor true_t(DT_BOOL, TensorShape({}));
      // Make sure we don't proceed if this switch node was already optimized.
      if (!already_optimized && SetTensorValue(DT_BOOL, true, &true_t).ok() &&
          SetTensorValue(DT_BOOL, false, &false_t).ok()) {
        // Copy the set of consumers of the switch as they will be manipulated
        // below.
        const std::set<NodeDef*>& consumer_set =
            node_map_->GetOutputs(node->name());
        std::vector<NodeDef*> consumers(consumer_set.begin(),
                                        consumer_set.end());
        std::sort(consumers.begin(), consumers.end(),
                  [](const NodeDef* n1, const NodeDef* n2) {
                    return n1->name() < n2->name();
                  });
        // Create constant false & true nodes.
        NodeDef* false_node = optimized_graph->add_node();
        false_node->set_name(OptimizedNodeName(*node, "_const_false"));
        if (!CreateNodeDef(false_node->name(), TensorValue(&false_t),
                           false_node)
                 .ok()) {
          continue;
        }
        false_node->set_device(node->device());

        NodeDef* true_node = optimized_graph->add_node();
        true_node->set_name(OptimizedNodeName(*node, "_const_true"));
        if (!CreateNodeDef(true_node->name(), TensorValue(&true_t), true_node)
                 .ok()) {
          continue;
        }
        true_node->set_device(node->device());

        // Add controls from the switch ports to the constants, and connect the
        // constants to the original switch outputs.
        const string false_port = node->name();
        const string true_port = strings::StrCat(node->name(), ":1");
        const string false_ctrl_dep =
            AddControlDependency(false_port, optimized_graph, node_map_.get());
        false_node->add_input(false_ctrl_dep);
        const string true_ctrl_dep =
            AddControlDependency(true_port, optimized_graph, node_map_.get());
        true_node->add_input(true_ctrl_dep);

        node_map_->AddNode(false_node->name(), false_node);
        node_map_->AddNode(true_node->name(), true_node);
        node_map_->AddOutput(NodeName(false_ctrl_dep), false_node->name());
        node_map_->AddOutput(NodeName(true_ctrl_dep), true_node->name());

        for (NodeDef* consumer : consumers) {
          for (int i = 0; i < consumer->input_size(); ++i) {
            const string& input = consumer->input(i);
            if (input == false_port) {
              consumer->set_input(i, false_node->name());
              node_map_->UpdateInput(consumer->name(), false_port,
                                     false_node->name());
            } else if (input == true_port) {
              consumer->set_input(i, true_node->name());
              node_map_->UpdateInput(consumer->name(), true_port,
                                     true_node->name());
            }
          }
        }
        graph_modified_ = true;
        continue;
      }
    }
    if (IsSimplifiableReduction(*node)) {
      // Replace the reduction node with an identity node, that can be further
      // optimized by the model pruner.
      DataType output_type;
      if (node->attr().count("T") > 0) {
        output_type = node->attr().at("T").type();
      } else {
        // This is an 'any' or 'all' reduction. The output is always boolean.
        output_type = DT_BOOL;
      }
      node->set_op("Identity");
      node->clear_attr();
      (*node->mutable_attr())["T"].set_type(output_type);
      *node->mutable_input(1) = AsControlDependency(node->input(1));
      graph_modified_ = true;
      continue;
    }
    if (use_shape_info && IsSimplifiableReshape(*node, *properties)) {
      DataType output_type = node->attr().at("T").type();
      node->set_op("Identity");
      node->clear_attr();
      (*node->mutable_attr())["T"].set_type(output_type);
      *node->mutable_input(1) = AsControlDependency(node->input(1));
      graph_modified_ = true;
      continue;
    }

    const bool is_mul = IsMul(*node) || IsLogicalAnd(*node);
    const bool is_matmul = IsMatMul(*node);
    const bool is_add = IsAdd(*node) || IsBiasAdd(*node) || IsLogicalOr(*node);
    const bool is_sub = IsSub(*node);
    const bool is_any_div = IsAnyDiv(*node);
    // Simplify arithmetic operations with ones or zeros.
    if (use_shape_info &&
        (is_mul || is_matmul || is_add || is_sub || is_any_div) &&
        properties->HasInputProperties(node->name()) &&
        properties->HasOutputProperties(node->name())) {
      const NodeDef* x = node_map_->GetNode(node->input(0));
      const NodeDef* y = node_map_->GetNode(node->input(1));
      if (x == nullptr || y == nullptr) {
        return errors::InvalidArgument("Invalid inputs to node: ",
                                       node->DebugString());
      }
      const TensorShapeProto& output_shape =
          properties->GetOutputProperties(node->name())[0].shape();

      // Simplify element-wise multiplication by ones or addition/subtraction
      // of zeros.
      const TensorShapeProto& y_shape =
          properties->GetInputProperties(node->name())[1].shape();
      const bool x_is_zero = IsZeros(*x);
      const bool x_is_one = x_is_zero ? false : IsOnes(*x);
      const bool y_matches_output_shape = ShapesEqual(output_shape, y_shape);
      if (y_matches_output_shape &&
          ((is_mul && x_is_one) || (is_add && x_is_zero))) {
        // 1 * y = y or 0 + y = y.
        ReplaceOperationWithSnapshot(1, *properties, node, optimized_graph);
        continue;
      }

      if (y_matches_output_shape && (is_sub && x_is_zero)) {
        // Replace 0 - y with Neg(y).
        ReplaceSubtractionFromZeroByNegation(node, optimized_graph);
        continue;
      }

      // Replace 1 / y with Reciprocal op.
      if (y_matches_output_shape && is_any_div && x_is_one) {
        DataType type = node->attr().at("T").type();
        if (DataTypeIsFloating(type) || DataTypeIsComplex(type)) {
          ReplaceDivisionOfOnesByReciprocal(node, optimized_graph);
          continue;
        }
      }

      const TensorShapeProto& x_shape =
          properties->GetInputProperties(node->name())[0].shape();
      const bool y_is_zero = IsZeros(*y);
      const bool y_is_one = y_is_zero ? false : IsOnes(*y);
      const bool x_matches_output_shape = ShapesEqual(output_shape, x_shape);
      if (x_matches_output_shape && (((is_mul || is_any_div) && y_is_one) ||
                                     ((is_add || is_sub) && y_is_zero))) {
        // x * 1 = x or x / 1 = x or x +/- 0 = x
        ReplaceOperationWithSnapshot(0, *properties, node, optimized_graph);
        continue;
      }

      // x OR true = true OR y = true.
      const PartialTensorShape shp(output_shape);
      if (shp.IsFullyDefined() && IsLogicalOr(*node) &&
          (y_is_one || x_is_one)) {
        TF_RETURN_IF_ERROR(ReplaceOperationWithConstant(
            1, *properties, output_shape, node, optimized_graph));
      }

      // Simplify multiplication and matmul by zeros.
      // Also optimize zeros divided by a tensor, but only if we are in
      // aggressive mode, since we might get rid of divisions by zero.
      bool optimize_zeros_divided_by_y =
          is_any_div && x_is_zero && is_aggressive;
      if ((x_is_zero || y_is_zero) &&
          (is_mul || is_matmul || optimize_zeros_divided_by_y)) {
        if (shp.IsFullyDefined()) {
          TF_RETURN_IF_ERROR(ReplaceOperationWithConstant(
              0, *properties, output_shape, node, optimized_graph));
          continue;
        }
        // Even if an input shape is only partially known, we may known that it
        // matches the output shape and thus forward the corresponding zero
        // input.
        if ((is_mul || is_any_div) && x_is_zero && x_matches_output_shape) {
          ReplaceOperationWithIdentity(0, *properties, node, optimized_graph);
          continue;
        } else if (is_mul && y_is_zero && y_matches_output_shape) {
          ReplaceOperationWithIdentity(1, *properties, node, optimized_graph);
          continue;
        }
      }
    }

    // Strength reduce floating point division by a constant Div(x, const) to
    // multiplication by the reciprocal Mul(x, Reciprocal(const)). This in turn
    // will be constant folded to Mul(x, 1.0/const).
    if (node->input_size() >= 2 && (IsRealDiv(*node) || IsDiv(*node))) {
      const string& const_input = node->input(1);
      const NodeDef* denom = node_map_->GetNode(const_input);
      CHECK(denom != nullptr);
      if (!IsReallyConstant(*denom)) {
        continue;
      }
      if (node->attr().count("T") == 0) {
        continue;
      }
      DataType type = node->attr().at("T").type();
      if (IsDiv(*node) &&
          !(DataTypeIsFloating(type) || DataTypeIsComplex(type))) {
        continue;
      }
      // Insert new reciprocal op and change node from Div to Mul.
      NodeDef* reciprocal_node = optimized_graph->add_node();
      reciprocal_node->set_name(OptimizedNodeName(*node, "_recip"));
      reciprocal_node->set_op("Reciprocal");
      reciprocal_node->set_device(node->device());
      node->set_op("Mul");
      // Re-wire inputs and outputs.
      reciprocal_node->add_input(const_input);
      (*reciprocal_node->mutable_attr())["T"].set_type(type);
      node->set_input(1, reciprocal_node->name());
      node_map_->AddNode(reciprocal_node->name(), reciprocal_node);
      node_map_->UpdateOutput(node->name(), const_input,
                              reciprocal_node->name());
      graph_modified_ = true;
      continue;
    }

    // Consider the transformation
    //
    //                      +                +       = parent
    //                     / \              / \
    //                    C   +    -- >    X   +     = children
    //                       / \              / \
    //                      X   Y            C   Y   = leaves
    //
    // where C is constant and X is non-constant, and '+' denotes an
    // associative and commutative operator like addition or multiplication.
    // This optimization pushes constants down in the tree to canonicalize it.
    // Moreoever, in cases where the child node has a second constant input Y
    // we will create a leaf node that can be folded, e.g.
    //
    //    Add(C1, Add(C2, X)) -> Add(X, Add(C1, C2)) -> Add(X, C1 + C2)
    //
    // TODO(rmlarsen): Handle non-associative/non-commutative operators like
    // subtraction and division, as well as mixed subtraction/addition,
    // division/multiplication.
    // Don't touch BiasAdd since they can't handle vectors as their first
    // inputs.
    if (has_fetch_ && (IsAdd(*node) || is_mul) &&
        NumNonControlInputs(*node) == 2) {
      NodeDef* left_child = node_map_->GetNode(node->input(0));
      NodeDef* right_child = node_map_->GetNode(node->input(1));
      // One child must be constant, and the other the same op as the parent.
      if (node->op() != left_child->op() && node->op() != right_child->op()) {
        continue;
      }
      const bool left_child_is_constant = IsReallyConstant(*left_child);
      const bool right_child_is_constant = IsReallyConstant(*right_child);
      if (!left_child_is_constant && !right_child_is_constant) {
        continue;
      }
      if (node->device() != left_child->device() ||
          node->device() != right_child->device()) {
        continue;
      }
      NodeDef* op_child_node =
          left_child_is_constant ? right_child : left_child;
      NodeDef* const_child_node =
          left_child_is_constant ? left_child : right_child;
      // Make sure that it is safe to change the value of the child node->
      if (op_child_node->input_size() < 2 ||
          nodes_to_preserve_.find(op_child_node->name()) !=
              nodes_to_preserve_.end() ||
          NumNonControlOutputs(*op_child_node, *node_map_) > 1) {
        continue;
      }

      // Identify the nodes to swap.
      NodeDef* left_leaf = node_map_->GetNode(op_child_node->input(0));
      NodeDef* right_leaf = node_map_->GetNode(op_child_node->input(1));
      const bool left_leaf_is_constant = IsReallyConstant(*left_leaf);
      const bool right_leaf_is_constant = IsReallyConstant(*right_leaf);
      if (left_leaf_is_constant && right_leaf_is_constant) {
        // Child is already foldable, leave it alone.
        continue;
      }
      const int non_const_leaf_input = left_leaf_is_constant ? 1 : 0;
      const int parent_const_input = left_child_is_constant ? 0 : 1;
      const auto& child_output = node_map_->GetOutputs(op_child_node->name());
      if (child_output.find(const_child_node) != child_output.end()) {
        // If there is a control edge from the child op to C, the transformation
        // would create a cycle in the graph. We know that it must be a control
        // edge. We can replace such a control edge with a control edge from A
        // to C.
        CHECK(MaybeRemoveControlInput(op_child_node->name(), const_child_node,
                                      graph_, node_map_.get()));
        NodeDef* other_leaf = left_leaf_is_constant ? left_leaf : right_leaf;
        MaybeAddControlInput(other_leaf->name(), const_child_node, graph_,
                             node_map_.get());
      }

      // Swap the constant child with a non-constant leaf node.
      node_map_->UpdateInput(node->name(), node->input(parent_const_input),
                             op_child_node->input(non_const_leaf_input));
      node_map_->UpdateInput(op_child_node->name(),
                             op_child_node->input(non_const_leaf_input),
                             node->input(parent_const_input));
      std::swap(*node->mutable_input(parent_const_input),
                *op_child_node->mutable_input(non_const_leaf_input));
      graph_modified_ = true;
      continue;
    }

    // Partial constant propagation through IdentityN.
    if (IsIdentityN(*node) && NumNonControlInputs(*node) > 0) {
      const std::set<NodeDef*>& tmp = node_map_->GetOutputs(node->name());
      const std::vector<NodeDef*> consumers(tmp.begin(), tmp.end());
      bool updated_graph = false;
      for (int input_idx = 0; input_idx < node->input_size(); ++input_idx) {
        const string& input = node->input(input_idx);
        if (IsControlInput(input)) {
          break;
        }
        const NodeDef* input_node = node_map_->GetNode(NodeName(input));
        if (input_node == nullptr) {
          LOG(ERROR) << "Bad input: " << input;
          break;
        }
        // Forward constant inputs to outputs and add a control dependency on
        // the IdentityN node.
        if (IsReallyConstant(*input_node)) {
          // Update each consumer.
          for (NodeDef* consumer : consumers) {
            bool add_dep = false;
            for (int consumer_input_idx = 0;
                 consumer_input_idx < consumer->input_size();
                 ++consumer_input_idx) {
              const string& consumer_input =
                  consumer->input(consumer_input_idx);
              if (IsControlInput(consumer_input)) {
                break;
              }
              int output_idx;
              const string input_node_name =
                  ParseNodeName(consumer_input, &output_idx);
              if (input_node_name == node->name() && output_idx == input_idx) {
                consumer->set_input(consumer_input_idx, input);
                // We will keep the input from IdentityN through a control
                // dependency, so we only need to add the consumer as an output
                // for the constant input node.
                node_map_->AddOutput(NodeName(input), consumer->name());
                add_dep = true;
              }
            }
            if (add_dep) {
              consumer->add_input(AsControlDependency(node->name()));
              updated_graph = true;
            }
          }
        }
      }

      if (updated_graph) {
        for (NodeDef* consumer : consumers) {
          DedupControlInputs(consumer);
        }
        graph_modified_ = true;
        continue;
      }
    }

    // Partial constant folding for associative operators:
    // Split AddN/AccumulateNV2 to enable partial
    // folding of ops when more than one but not all inputs are constant.
    // For AddN and AccumulateNV2, we may furthermore reorder inputs, since
    // addition is commutative.
    const int num_non_control_inputs = NumNonControlInputs(*node);
    if (IsAggregate(*node) && IsCommutative(*node) &&
        num_non_control_inputs > 2) {
      const int num_control_inputs =
          node->input_size() - num_non_control_inputs;
      std::vector<int> const_inputs;
      std::vector<int> nonconst_inputs;
      for (int i = 0; i < node->input_size(); ++i) {
        const string& input = node->input(i);
        const NodeDef* input_node = node_map_->GetNode(NodeName(input));
        CHECK(input_node != nullptr) << input;
        if (!IsControlInput(input) && IsReallyConstant(*input_node)) {
          const_inputs.push_back(i);
        } else {
          // Non-const and control inputs.
          nonconst_inputs.push_back(i);
        }
      }
      // Promote AccumulateNV2 with all constant inputs to AddN, since it is
      // a fake node that cannot be constant folded by itself.
      if (const_inputs.size() == num_non_control_inputs &&
          node->op() == "AccumulateNV2") {
        node->set_op("AddN");
        node->mutable_attr()->erase("shape");
        graph_modified_ = true;
        continue;
      }
      const string new_node_name = OptimizedNodeName(
          *node, strings::StrCat("_partial_split_", const_inputs.size()));
      if (1 < const_inputs.size() &&
          const_inputs.size() < num_non_control_inputs &&
          !node_map_->NodeExists(new_node_name)) {
        NodeDef* added_node = optimized_graph->add_node();
        *added_node = *node;
        // Always use AddN for the constant node, since AccumulateNV2 is a fake
        // node that cannot be constant folded, since it does not have a kernel.
        added_node->set_op("AddN");
        added_node->mutable_attr()->erase("shape");
        added_node->set_name(new_node_name);
        node_map_->AddNode(added_node->name(), added_node);
        added_node->clear_input();
        for (int i : const_inputs) {
          added_node->add_input(node->input(i));
          node_map_->UpdateOutput(NodeName(node->input(i)), node->name(),
                                  added_node->name());
        }

        // Overwrite the first const input with the added node.
        node->set_input(const_inputs[0], added_node->name());
        node_map_->AddOutput(added_node->name(), node->name());
        nonconst_inputs.push_back(const_inputs[0]);
        // Compact the remaining inputs to the original node.
        std::sort(nonconst_inputs.begin(), nonconst_inputs.end());
        int idx = 0;
        for (int i : nonconst_inputs) {
          if (idx != i) {
            node->set_input(idx, node->input(i));
          }
          ++idx;
        }
        node->mutable_input()->DeleteSubrange(nonconst_inputs.size(),
                                              const_inputs.size() - 1);
        (*node->mutable_attr())["N"].set_i(node->input_size() -
                                           num_control_inputs);
        properties->ClearInputProperties(node->name());
        (*added_node->mutable_attr())["N"].set_i(const_inputs.size());
        graph_modified_ = true;
        continue;
      }
    }

    if (PartialConcatConstFolding(optimized_graph, properties, node)) {
      graph_modified_ = true;
      continue;
    }
  }

  return Status::OK();
}

bool ConstantFolding::PartialConcatConstFolding(GraphDef* optimized_graph,
                                                GraphProperties* properties,
                                                NodeDef* node) {
  // Partial constant folding for Concat which is not commutative, so
  // we have to preserve order and can only push consecutive runs of constant
  // inputs into sub-nodes.
  const int num_non_control_inputs = NumNonControlInputs(*node);
  if (IsConcat(*node) && num_non_control_inputs > 3 &&
      node->name().rfind("_partial_split_") == string::npos) {
    int axis_arg = -1;
    int begin = 0;
    int end = num_non_control_inputs;
    if (node->op() == "Concat") {
      begin = 1;
      axis_arg = 0;
    } else if (node->op() == "ConcatV2") {
      end = num_non_control_inputs - 1;
      axis_arg = num_non_control_inputs - 1;
    } else {
      return false;
    }

    const NodeDef* axis_arg_node =
        node_map_->GetNode(NodeName(node->input(axis_arg)));
    if (axis_arg_node == nullptr || !IsReallyConstant(*axis_arg_node)) {
      // We cannot constant fold Concat unless we the axis argument is
      // constant. Skip node.
      return false;
    }

    // We search for consecutive runs of constant inputs in the range
    // [begin:end[ and push then down into child nodes.
    std::vector<std::pair<int, int>> constant_input_runs;
    int first = begin;
    int last = begin;
    while (last < end) {
      while (first < end && !IsReallyConstant(*node_map_->GetNode(
                                NodeName(node->input(first))))) {
        ++first;
      }
      // Invariant: node[first] is constant || first >= end.
      last = first + 1;
      while (last < end && IsReallyConstant(*node_map_->GetNode(
                               NodeName(node->input(last))))) {
        ++last;
      }
      // Invariant: node[last] is not constant || last >= end
      // Discard intervals shorter than 2 elements.
      if (first < end && (last - first) > 1) {
        constant_input_runs.emplace_back(first, last);
      }
      first = last;
    }

    // Skip if all inputs are constant, and let constant folding take over.
    if (constant_input_runs.size() == 1 &&
        constant_input_runs[0].first == begin &&
        constant_input_runs[0].second == end) {
      return false;
    }
    std::set<int> inputs_to_delete;
    for (auto interval : constant_input_runs) {
      // Push the constant inputs in the interval to a child node than can be
      // constant folded.
      const string new_node_name = OptimizedNodeName(
          *node, strings::StrCat("_partial_split_", interval.first));
      if (node_map_->NodeExists(new_node_name)) {
        break;
      }
      NodeDef* added_node = optimized_graph->add_node();
      *added_node = *node;
      added_node->set_name(new_node_name);
      node_map_->AddNode(added_node->name(), added_node);
      added_node->clear_input();
      for (int i = interval.first; i < interval.second; ++i) {
        added_node->add_input(node->input(i));
        node_map_->UpdateOutput(NodeName(node->input(i)), node->name(),
                                added_node->name());
        if (i != interval.first) {
          inputs_to_delete.insert(i);
        }
      }
      added_node->add_input(node->input(axis_arg));
      (*added_node->mutable_attr())["N"].set_i(interval.second -
                                               interval.first);
      node_map_->AddOutput(NodeName(node->input(axis_arg)), added_node->name());

      // Overwrite the first constant input with the result of the added
      // child node.
      node->set_input(interval.first, added_node->name());
      node_map_->AddOutput(added_node->name(), node->name());
    }
    if (!constant_input_runs.empty()) {
      if (!inputs_to_delete.empty()) {
        // Fix up the inputs to the original node.
        std::vector<string> tmp(node->input().begin(), node->input().end());
        node->clear_input();
        for (int i = 0; i < tmp.size(); ++i) {
          if (inputs_to_delete.find(i) == inputs_to_delete.end()) {
            node->add_input(tmp[i]);
          }
        }
        (*node->mutable_attr())["N"].set_i(node->input_size() - 1);
        properties->ClearInputProperties(node->name());
      }
      return true;
    }
  }
  return false;
}

Status ConstantFolding::RunOptimizationPass(Cluster* cluster,
                                            const GrapplerItem& item,
                                            GraphDef* optimized_graph) {
  node_map_.reset(new NodeMap(graph_));
  nodes_whitelist_.clear();
  // Fold fetch nodes iff it has a single fanout. Note that if a fetch node
  // has a single fanout, it would be rewritten as a constant with the same
  // node name, and therefore users are still able to fetch it. This is not
  // the case if the node has multiple fanouts, and constant folding would
  // replace the node with multiple constants (each for one fanout) with
  // new names, and as a result users would not be able to fetch the node any
  // more with the original node name.
  for (const auto& fetch : item.fetch) {
    const NodeDef* fetch_node = node_map_->GetNode(fetch);
    if (fetch_node && NumOutputs(*fetch_node, graph_) == 1) {
      nodes_whitelist_.insert(fetch_node->name());
    }
  }

  GraphProperties properties(item);
  // It's possible to feed a placeholder with a tensor of any shape: make sure
  // that the shape inference deals with this conservatively unless we're in
  // aggressive mode.
  const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
  Status s = properties.InferStatically(assume_valid_feeds);
  const bool can_use_shape_info = s.ok();

  if (can_use_shape_info) {
    TF_RETURN_IF_ERROR(MaterializeShapes(properties));
    TF_RETURN_IF_ERROR(MaterializeConstants(properties));
  }
  TF_RETURN_IF_ERROR(FoldGraph(optimized_graph));
  node_map_.reset(new NodeMap(optimized_graph));
  TF_RETURN_IF_ERROR(
      SimplifyGraph(optimized_graph, &properties, can_use_shape_info));

  return Status::OK();
}

Status ConstantFolding::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  // TensorFlow flushes denormals to zero and rounds to nearest, so we do
  // the same here.
  port::ScopedFlushDenormal flush;
  port::ScopedSetRound round(FE_TONEAREST);
  nodes_to_preserve_ = item.NodesToPreserve();
  for (const auto& feed : item.feed) {
    feed_nodes_.insert(NodeName(feed.first));
  }

  if (cpu_device_ == nullptr) {
    owned_device_.reset(new DeviceSimple());
    cpu_device_ = owned_device_.get();
  }

  has_fetch_ = !item.fetch.empty();
  GrapplerItem item_to_optimize = item;
  *optimized_graph = item.graph;
  int64 node_count;
  do {
    graph_modified_ = false;
    item_to_optimize.graph.Swap(optimized_graph);
    graph_ = &item_to_optimize.graph;
    *optimized_graph = GraphDef();
    node_count = graph_->node_size();
    TF_RETURN_IF_ERROR(
        RunOptimizationPass(cluster, item_to_optimize, optimized_graph));
  } while (graph_modified_ || optimized_graph->node_size() != node_count);
  *optimized_graph->mutable_library() = item.graph.library();
  *optimized_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void ConstantFolding::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for ConstantFolding.
}

}  // namespace grappler
}  // namespace tensorflow
