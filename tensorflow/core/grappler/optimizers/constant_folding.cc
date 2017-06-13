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
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace grappler {
using TensorVector = gtl::InlinedVector<TensorValue, 4>;

namespace {
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(thread::ThreadPool* pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}
  void Schedule(std::function<void()> fn) override {
    pool_->Schedule(std::move(fn));
  }
  int NumThreads() const override { return pool_->NumThreads(); }
  int CurrentThreadId() const override { return pool_->CurrentThreadId(); }

 private:
  thread::ThreadPool* pool_ = nullptr;
};

class DeviceSimple : public DeviceBase {
 public:
  DeviceSimple() : DeviceBase(Env::Default()) {
    eigen_worker_threads_.num_threads = 1;
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

string AsControlDependency(const NodeDef& node) {
  return strings::StrCat("^", node.name());
}

}  // namespace

ConstantFolding::ConstantFolding() {
  ops_to_preserve_ = std::regex(
      "Placeholder.*|Const|.*Save.*|.*Restore.*|.*Reader|Enter|Exit|"
      "NextIteration");
}

string ConstantFolding::AddControlDependency(const string& input_name) {
  const NodeDef* node = node_map_->GetNode(input_name);
  if (!IsSwitch(*node)) {
    return AsControlDependency(*node);
  } else {
    // We can't anchor control dependencies directly on the switch node: unlike
    // other nodes only one of the outputs of the switch node will be generated
    // when the switch node is executed, and we need to make sure the control
    // dependency is only triggered when the corresponding output is triggered.
    // We start by looking for an identity node connected to the output of the
    // switch node, and use it to anchor the control dependency.
    auto outputs = node_map_->GetOutputs(node->name());
    for (const NodeDef* node : outputs) {
      if (IsIdentity(*node)) {
        CHECK_EQ(1, node->input_size());
        if (IsSameInput(node->input(0), input_name)) {
          return AsControlDependency(*node);
        }
      }
    }
    // We haven't found an existing node where we can anchor the control
    // dependency: add a new identity node.
    int position = 0;
    string ctrl_dep_name = ParseNodeName(input_name, &position);
    strings::StrAppend(&ctrl_dep_name, "_", position);
    ctrl_dep_name = AddPrefixToNodeName(ctrl_dep_name, kConstantFoldingCtrl);
    const DataType output_type = node->attr().at("T").type();

    NodeDef* added_node = graph_.add_node();
    added_node->set_name(ctrl_dep_name);
    added_node->set_op("Identity");
    (*added_node->mutable_attr())["T"].set_type(output_type);
    *added_node->add_input() = input_name;
    node_map_->AddNode(added_node->name(), added_node);
    node_map_->AddOutput(node->name(), added_node->name());
    return AsControlDependency(*added_node);
  }
}

Status ConstantFolding::MaterializeShapes(const GrapplerItem& item) {
  GraphProperties properties(item);
  TF_RETURN_IF_ERROR(properties.InferStatically());
  // We may add some nodes to the graph to encode control dependencies: there is
  // no need to process these, so only iterate over the nodes of the input
  // graph.
  const int node_count = graph_.node_size();
  for (int i = 0; i < node_count; ++i) {
    NodeDef& node = *graph_.mutable_node(i);
    const string op = node.op();
    if (op != "Shape" && op != "Size" && op != "Rank") {
      continue;
    }
    std::vector<OpInfo::TensorProperties> output =
        properties.GetOutputProperties(node.name());
    CHECK_EQ(1, output.size());
    const DataType type = output[0].dtype();
    CHECK(type == DT_INT32 || type == DT_INT64);

    std::vector<OpInfo::TensorProperties> input =
        properties.GetInputProperties(node.name());
    CHECK_EQ(1, input.size());

    const TensorShapeProto shape = input[0].shape();
    // Materialize the shapes using constants whenever possible.
    PartialTensorShape shp(shape);
    if (shp.IsFullyDefined() || (!shp.unknown_rank() && op == "Rank")) {
      bool valid = true;
      Tensor value(type);
      if (op == "Shape") {
        value = Tensor(type, TensorShape({shp.dims()}));
        for (int i = 0; i < shp.dims(); ++i) {
          if (type == DT_INT32) {
            if (shp.dim_size(i) >= INT_MAX) {
              valid = false;
              break;
            }
            value.flat<int32>()(i) = shp.dim_size(i);
          } else {
            value.flat<int64>()(i) = shp.dim_size(i);
          }
        }
      } else if (op == "Size") {
        int64 size = 1;
        for (int i = 0; i < shp.dims(); ++i) {
          size *= shp.dim_size(i);
        }
        value = Tensor(type, TensorShape({}));
        if (type == DT_INT32) {
          if (size >= INT_MAX) {
            valid = false;
          } else {
            value.flat<int32>()(0) = size;
          }
        } else {
          value.flat<int64>()(0) = size;
        }
      } else {
        value = Tensor(type, TensorShape({}));
        if (type == DT_INT32) {
          if (shp.dims() >= INT_MAX) {
            valid = false;
          } else {
            value.flat<int32>()(0) = shp.dims();
          }
        } else {
          value.flat<int64>()(0) = shp.dims();
        }
      }

      if (valid) {
        // Replace the node with the corresponding constant.
        node.set_op("Const");
        node.clear_attr();
        (*node.mutable_attr())["dtype"].set_type(type);
        value.AsProtoTensorContent(
            (*node.mutable_attr())["value"].mutable_tensor());

        // Turn the data input into a control dependency: this is needed to
        // ensure that the constant value will only be generated in the cases
        // where the shape/rank/size would have been generated in the original
        // graph. Additional inputs are extra control dependencies that we
        // preserve.
        CHECK_LE(1, node.input_size());
        string ctrl_dep = AddControlDependency(node.input(0));
        node.set_input(0, ctrl_dep);
      }
    }
  }
  return Status::OK();
}

bool ConstantFolding::IsFoldable(const NodeDef& node) const {
  // Skips nodes that must be preserved, and op_types that don't benefit from
  // folding
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  std::cmatch match;
  if (std::regex_match(node.op().c_str(), match, ops_to_preserve_)) {
    return false;
  }

  // Don't fold stateful ops such as TruncatedNormal.
  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  if (op_def->is_stateful()) {
    return false;
  }

  if (op_def->output_arg_size() == 0) {
    return false;
  }

  DeviceTypeVector device_types;
  status = SupportedDeviceTypesForNode({DeviceType(DEVICE_CPU)}, node,
                                       &device_types);
  if (!status.ok()) {
    return false;
  }
  // Only fold ops with a CPU implementation available.
  if (device_types[0] != DeviceType(DEVICE_CPU)) {
    return false;
  }

  // Folding not applicable to ops with no inputs.
  if (node.input().empty()) {
    return false;
  }

  // No need to (and don't) fold nodes that have no outgoing edges. Such nodes
  // could be introduced by an earlier constant folding pass and are preserved
  // in case users want to fetch their values; re-processing them would
  // lead to an error of adding a duplicated node to graph.
  auto outputs = node_map_->GetOutputs(node.name());
  if (outputs.empty()) {
    return false;
  }

  for (const auto& input : node.input()) {
    if (IsControlInput(input)) {
      continue;
    }
    bool is_const = IsConstant(*node_map_->GetNode(input));
    if (!is_const) {
      return false;
    }
  }
  return true;
}

NodeDef ConstantFolding::CreateNodeDef(const string& name,
                                       const TensorValue& tensor) {
  NodeDef node;
  node.set_name(name);
  node.set_op("Const");
  AttrValue attr_output_shape;
  auto output_shape = attr_output_shape.mutable_list()->add_shape();
  TensorShapeProto shape;
  tensor->shape().AsProto(&shape);
  *output_shape = shape;
  node.mutable_attr()->insert({"_output_shapes", attr_output_shape});

  AttrValue attr_type;
  attr_type.set_type(tensor->dtype());
  node.mutable_attr()->insert({"dtype", attr_type});

  AttrValue attr_tensor;
  tensor->AsProtoTensorContent(attr_tensor.mutable_tensor());
  node.mutable_attr()->insert({"value", attr_tensor});
  return node;
}

Status ConstantFolding::EvaluateNode(const NodeDef& node,
                                     const TensorVector& inputs,
                                     TensorVector* output) const {
  Status status;
  auto op_kernel =
      CreateOpKernel("CPU", device_.get(), device_->GetAllocator({}), node,
                     TF_GRAPH_DEF_VERSION, &status);
  TF_RETURN_IF_ERROR(status);
  OpKernelContext::Params params;
  params.device = device_.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op_kernel.get();

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
  return Status::OK();
}

Status ConstantFolding::EvaluateOneFoldable(const NodeDef& node,
                                            std::vector<NodeDef>* outputs) {
  TensorVector inputs;
  for (const auto& input : node.input()) {
    if (IsControlInput(input)) {
      break;
    }
    TensorVector output;
    TF_RETURN_IF_ERROR(
        EvaluateNode(*node_map_->GetNode(input), TensorVector(), &output));
    inputs.push_back(output[0]);
  }

  TensorVector output_tensors;
  TF_RETURN_IF_ERROR(EvaluateNode(node, inputs, &output_tensors));
  for (const auto& input : inputs) {
    delete input.tensor;
  }
  if (output_tensors.empty()) {
    Status(error::INVALID_ARGUMENT, "Expected at least one output.");
  }
  for (int i = 0; i < output_tensors.size(); i++) {
    string node_name = AddPrefixToNodeName(node.name(), kConstantFoldingConst);
    if (output_tensors.size() > 1) {
      node_name = strings::StrCat(node_name, "-", i);
    }
    if (output_tensors[i].tensor) {
      outputs->push_back(CreateNodeDef(node_name, output_tensors[i]));
      delete output_tensors[i].tensor;
    } else {
      // Create an empty NodeDef to identify dead outputs (e.g. the output of a
      // switch that's not selected by the switch predicate).
      outputs->push_back(NodeDef());
    }
  }
  return Status::OK();
}

Status ConstantFolding::FoldNode(const NodeDef& node, GraphDef* output) {
  std::vector<NodeDef> const_nodes;
  TF_RETURN_IF_ERROR(EvaluateOneFoldable(node, &const_nodes));

  NodeDef* constant_output = nullptr;
  for (const auto& const_node : const_nodes) {
    if (const_node.name().empty()) {
      // Dead output: we can't create a constant to encode its value, so we'll
      // just skip it. We'll preserve the edges that originate from that output
      // below to preserve the overall behavior of the graph wrt dead edges.
      continue;
    }
    NodeDef* added_node = output->add_node();
    *added_node = const_node;
    node_map_->AddNode(added_node->name(), added_node);

    for (const auto& input : node.input()) {
      if (IsControlInput(input)) {
        *added_node->add_input() = input;
      } else {
        NodeDef* input_node = node_map_->GetNode(input);
        for (const auto& fanin_of_input : input_node->input()) {
          if (IsControlInput(fanin_of_input)) {
            *added_node->add_input() = fanin_of_input;
          }
        }
      }
    }

    // All the constant nodes encoding output values have the same control
    // dependencies (since these are the control dependencies of the node we're
    // trying to fold). Record one such constant node.
    constant_output = added_node;
  }

  auto outputs = node_map_->GetOutputs(node.name());
  for (const auto& output : outputs) {
    for (int i = 0; i < output->input_size(); i++) {
      int position;
      string node_name = ParseNodeName(output->input(i), &position);
      if (node_name == node.name()) {
        if (position < 0) {
          // Propagate control dependencies if possible. If not, we'll just
          // preserve the existing control dependencies.
          if (constant_output != nullptr) {
            *output->mutable_input(i) = AsControlDependency(*constant_output);
          }

        } else if (position < const_nodes.size() &&
                   !const_nodes[position].name().empty()) {
          // Replace alive outputs with the corresponding constant.
          *output->mutable_input(i) = const_nodes[position].name();
        } else {
          // Leave this edge alone.
          VLOG(1) << "Preserving edge from " << node.name() << ":" << position
                  << "[" << node.op() << "] to " << output->name() << ":" << i
                  << "[" << output->op() << "]";
        }
      }
    }
  }
  return Status::OK();
}

Status ConstantFolding::FoldGraph(GraphDef* output) {
  std::set<string> processed_nodes;
  while (1) {
    int previous_processed = processed_nodes.size();
    for (const auto& node : graph_.node()) {
      if (IsFoldable(node) &&
          processed_nodes.find(node.name()) == processed_nodes.end()) {
        TF_RETURN_IF_ERROR(FoldNode(node, output));
        processed_nodes.insert(node.name());
      }
    }
    int current_processed = processed_nodes.size();
    LOG(INFO) << "Previous number of processed nodes: " << previous_processed
              << "; Current number of processed nodes: " << current_processed;
    if (current_processed == previous_processed) {
      break;
    }
  }

  // Build the graph after constant folding. Note that we keep all processed
  // nodes in the graph in case users need to fetch their values.
  for (const auto& node : graph_.node()) {
    auto added_node = output->add_node();
    *added_node = node;
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
    if (IsConstant(*reductions_indices)) {
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

Status ConstantFolding::SimplifyGraph(GraphDef* output) {
  for (auto& node : *output->mutable_node()) {
    if (IsSimplifiableReduction(node)) {
      // Replace the reduction node with an identity node, that can be further
      // optimized by the model pruner.
      const NodeDef* reductions_indices = node_map_->GetNode(node.input(1));
      DataType output_type;
      if (node.attr().count("T") > 0) {
        output_type = node.attr().at("T").type();
      } else {
        // This is an 'any' or 'all' reduction. The output is always boolean.
        output_type = DT_BOOL;
      }
      node.set_op("Identity");
      node.clear_attr();
      (*node.mutable_attr())["T"].set_type(output_type);
      if (node.input_size() > 2) {
        node.mutable_input()->SwapElements(1, node.input_size() - 1);
      }
      node.mutable_input()->RemoveLast();
      for (const auto& input : reductions_indices->input()) {
        if (IsControlInput(input)) {
          *node.add_input() = input;
        }
      }
    }
  }
  return Status::OK();
}

Status ConstantFolding::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output) {
  graph_ = item.graph;
  LOG(INFO) << "Initial graph size: " << item.graph.node_size();
  node_map_.reset(new NodeMap(&graph_));
  for (const auto& node : item.fetch) {
    nodes_to_preserve_.insert(NodeName(node));
  }
  for (const auto& node : item.feed) {
    nodes_to_preserve_.insert(NodeName(node.first));
  }
  device_.reset(new DeviceSimple());
  *output = GraphDef();
  TF_RETURN_IF_ERROR(MaterializeShapes(item));
  TF_RETURN_IF_ERROR(FoldGraph(output));
  TF_RETURN_IF_ERROR(SimplifyGraph(output));
  LOG(INFO) << "Optimized graph size: " << output->node_size();
  return Status::OK();
}

void ConstantFolding::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for ConstantFolding.
}

}  // end namespace grappler
}  // end namespace tensorflow
