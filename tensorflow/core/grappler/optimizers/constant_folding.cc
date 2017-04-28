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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
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
  DeviceSimple() : DeviceBase(nullptr) {
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

Status NumOutputs(const NodeDef& node, int* num_outputs) {
  const OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node.op(), &op_def));
  if (node.op() == "ConcatOffset") {
    *num_outputs = node.attr().at("N").i();
  } else {
    *num_outputs = op_def->output_arg_size();
  }
  return Status::OK();
}
}  // namespace

bool ConstantFolding::IsConst(const NodeDef& node) const {
  return node.op() == "Const";
}

bool ConstantFolding::IsFoldable(const NodeDef& node) const {
  DeviceTypeVector device_types;
  auto status = SupportedDeviceTypesForNode({DeviceType(DEVICE_CPU)}, node,
                                            &device_types);
  if (!status.ok()) {
    return false;
  }
  // Only fold ops with a CPU implementation available.
  if (device_types[0] != DeviceType(DEVICE_CPU)) {
    return false;
  }

  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }

  if (ops_to_preserve_.find(node.op()) != ops_to_preserve_.end()) {
    return false;
  }

  // Don't fold stateful ops such as TruncatedNormal.
  const OpDef* op_def = nullptr;
  status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  if (op_def->is_stateful()) {
    return false;
  }

  if (op_def->output_arg_size() == 0) {
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
    bool is_const = IsConst(*node_map_->GetNode(input));
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
                                     TensorVector* output) {
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
  int num_outputs;
  TF_RETURN_IF_ERROR(NumOutputs(node, &num_outputs));
  gtl::InlinedVector<AllocatorAttributes, 4> output_attrs;
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
    outputs->push_back(CreateNodeDef(node_name, output_tensors[i]));
    delete output_tensors[i].tensor;
  }
  return Status::OK();
}

Status ConstantFolding::FoldNode(const NodeDef& node, GraphDef* output) {
  std::vector<NodeDef> const_nodes;
  TF_RETURN_IF_ERROR(EvaluateOneFoldable(node, &const_nodes));

  auto outputs = node_map_->GetOutputs(node.name());
  for (const auto& const_node : const_nodes) {
    NodeDef* added_node = output->add_node();
    *added_node = const_node;
    node_map_->AddNode(added_node->name(), added_node);
  }
  for (const auto& output : outputs) {
    for (int i = 0; i < output->input_size(); i++) {
      int position;
      string node_name = ParseNodeName(output->input(i), &position);
      if (node_name == node.name()) {
        if (position < 0) {
          *output->mutable_input(i) =
              strings::StrCat("^", const_nodes[0].name());
        } else {
          *output->mutable_input(i) = const_nodes[position].name();
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

Status ConstantFolding::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output) {
  graph_ = item.graph;
  LOG(INFO) << "Initial graph size: " << item.graph.node_size();
  node_map_.reset(new NodeMap(&graph_));
  for (const auto& node : item.fetch) {
    nodes_to_preserve_.insert(NodeName(node));
  }
  device_.reset(new DeviceSimple());
  *output = GraphDef();
  TF_RETURN_IF_ERROR(FoldGraph(output));
  LOG(INFO) << "Optimized graph size: " << output->node_size();
  return Status::OK();
}

void ConstantFolding::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for ConstantFolding.
}

}  // end namespace grappler
}  // end namespace tensorflow
