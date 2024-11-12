/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM

#include <algorithm>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tsl/platform/status.h"

#include "tensorflow/core/common_runtime/gpu_fusion_pass.h"

namespace tensorflow {
namespace gpu_fusion_pass {

//----------------------------------------------------------------------

// declaration of all the constants that are used in the gpu_fusion_pass
// namespace
const int kVlogLevel = 2;

// an adapter on the util version of the ReadBoolFromEnvVar routine
// this version better suits the usage of the routine in this file.
// main modifications are
// 1. default value is hard-coded to false
// 2. return value is the bool
//    - true if env-var value is set to 1/true
//    - false if env-var value is either not set or set to any other value
bool ReadBoolFromEnvVar(const char* env_var_name) {
  bool value = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar(env_var_name, false, &value));
  return value;
}

// graph pass grouping for this fusion pass
//
// We want tf.function(x1*y1+x2*y2) to be fused into FMA (at least in
// cwise_ops_test.py), and that seems to require POST_PARTITIONING
//
const OptimizationPassRegistry::Grouping kROCmFMAPassGrouping =
    OptimizationPassRegistry::POST_PARTITIONING;

const OptimizationPassRegistry::Grouping kROCmFusionPassGrouping =
    ReadBoolFromEnvVar("TF_ROCM_FUSION_PASS_POST_PARTITIONING")
        ? OptimizationPassRegistry::POST_PARTITIONING
        : OptimizationPassRegistry::POST_PLACEMENT;

// attribute name strings
const char* kAttr_N = "N";
const char* kAttr_T = "T";
const char* kAttr_U = "U";

const char* kAttr_activation_mode = "activation_mode";
const char* kAttr_bop_type = "bop_type";
const char* kAttr_data_format = "data_format";
const char* kAttr_dilations = "dilations";
const char* kAttr_epsilon = "epsilon";
const char* kAttr_is_training = "is_training";
const char* kAttr_padding = "padding";
const char* kAttr_strides = "strides";

const char* kGPUDeviceStr = "GPU";

//----------------------------------------------------------------------

// Util routines - Start

std::ostream& operator<<(std::ostream& out, const Node* n) {
  out << "id : " << n->id() << ", name : " << n->name()
      << ", type_string : " << n->type_string()
      << ", num_inputs : " << n->num_inputs()
      << ", num_input_edges : " << n->in_edges().size()
      << ", num_outputs : " << n->num_outputs()
      << ", num_output_edges : " << n->out_edges().size()
      << ", attributes : " << n->attrs().DebugString()
      // << ", requested_device : " << n->requested_device()
      // << ", assigned_device : "
      // << (n->has_assigned_device_name() ? n->assigned_device_name() : "None")
      ;

  return out;
}

void DumpNodeList(int lvl, string message, std::list<const Node*> nodes) {
  VLOG(lvl) << message;
  for (auto n : nodes) {
    VLOG(lvl) << "\t" << n;
  }
}

void DumpOutputEdges(int lvl, const Node* node) {
  for (const Edge* e : node->out_edges()) {
    VLOG(lvl) << "\t" << e->src_output() << " : " << e->dst()->name()
              << e->dst()->type_string();
  }
}

void DumpGraph(int lvl, StringPiece label, const Graph* g) {
  VLOG(lvl) << "Graph " << label << " #nodes " << g->num_nodes() << " #edges "
            << g->num_edges();

  for (const auto& line : str_util::Split(DebugString(g), '\n')) {
    VLOG(lvl) << "|| " << line;
  }
}

bool isGpuDevice(StringPiece fullname) {
  DeviceNameUtils::ParsedName p;
  return (DeviceNameUtils::ParseFullName(fullname, &p) &&
          (p.type == kGPUDeviceStr));
}

bool areAssignedToSameGpu(std::list<const Node*> nodes) {
  int device_id = -1;
  bool on_same_gpu = true;

  DeviceNameUtils::ParsedName p;
  bool first_node = true;
  for (auto n : nodes) {
    if (!n->has_assigned_device_name()) {
      on_same_gpu = false;
      break;
    }

    DeviceNameUtils::ParseFullName(n->assigned_device_name(), &p);
    if (p.type != kGPUDeviceStr) {
      on_same_gpu = false;
      break;
    }

    if (first_node) {
      device_id = p.id;
    } else if (device_id != p.id) {
      on_same_gpu = false;
      break;
    }
  }

  return on_same_gpu;
}

// is this node an instance of a convolution op for which we support fusion for?
inline bool isOpConvolution(const Node* n) {
  return (n->type_string() == "Conv2D");
}

// is this node an instance of a bias op for which we support fusion for?
inline bool isOpBias(const Node* n) { return (n->type_string() == "BiasAdd"); }

// is this node an instance of a activation op for which we support fusion for?
inline bool isOpActivation(const Node* n) {
  return ((n->type_string() == "Sigmoid") || (n->type_string() == "Relu") ||
          (n->type_string() == "Relu6") || (n->type_string() == "Tanh"));
}

// is this node an instance of a activation gradient op for which we support
// fusion for?
inline bool isOpActivationGrad(const Node* n) {
  return (
      (n->type_string() == "SigmoidGrad") || (n->type_string() == "ReluGrad") ||
      (n->type_string() == "Relu6Grad") || (n->type_string() == "TanhGrad"));
}

// is this node an instance of a batchnorm op for which we support fusion for?
inline bool isOpBatchNorm(const Node* n) {
  return ((n->type_string() == "FusedBatchNorm") ||
          (n->type_string() == "FusedBatchNormV2") ||
          (n->type_string() == "FusedBatchNormV3"));
}

// is this node an instance of a batchnorm gradient op for which we support
// fusion for?
inline bool isOpBatchNormGrad(const Node* n) {
  return ((n->type_string() == "FusedBatchNormGrad") ||
          (n->type_string() == "FusedBatchNormGradV2") ||
          (n->type_string() == "FusedBatchNormGradV3"));
}

// is this node an instance of the "Add" op?
inline bool isOpAdd(const Node* n) { return (n->type_string() == "Add"); }

inline bool isOpAddX(const Node* n) {
  return (n->type_string() == "Add") || (n->type_string() == "AddV2");
}

// is this node an instance of the "AddN" op?
inline bool isOpAddN(const Node* n) { return (n->type_string() == "AddN"); }

// is this node an instance of the "Sub" op?
inline bool isOpSub(const Node* n) { return (n->type_string() == "Sub"); }

// is this node an instance of the "Relu" op?
inline bool isOpRelu(const Node* n) { return (n->type_string() == "Relu"); }

// is this node an instance of the "Relu" op or the "_ROCmFusedAddRelu" op?
inline bool isOpReluOrFusedAddRelu(const Node* n) {
  return ((n->type_string() == "Relu") ||
          (n->type_string() == "_ROCmFusedAddRelu"));
}

// is this node an instance of the "ReluGrad" op?
inline bool isOpReluGrad(const Node* n) {
  return (n->type_string() == "ReluGrad");
}

// return the activation op type string from the given activation gradient node
inline string getActivationOpType(const Node* n) {
  string activation_type = "None";

  if (n->type_string() == "SigmoidGrad") {
    activation_type = "Sigmoid";
  } else if (n->type_string() == "ReluGrad") {
    activation_type = "Relu";
  } else if (n->type_string() == "Relu6Grad") {
    activation_type = "Relu6";
  } else if (n->type_string() == "TanhGrad") {
    activation_type = "Tanh";
  }

  return activation_type;
}

// Util routines - End

//----------------------------------------------------------------------

class ROCmFusionOpBase;

class ROCmFusionPassBase : public GraphOptimizationPass {
 public:
  // optimization pass entry point,
  // application code will call this routine to run the pass
  virtual Status Run(const GraphOptimizationPassOptions& options, int grouping);
  virtual void InitializeFusions(
      std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions, Graph* g) = 0;

 private:
  // helper function that does all the work for this pass
  bool RunPass(Graph* g);
};

class ROCmFusionPass : public ROCmFusionPassBase {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
  void InitializeFusions(
      std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions,
      Graph* g) override;
};

class ROCmFMAPass : public ROCmFusionPassBase {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
  void InitializeFusions(
      std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions,
      Graph* g) override;
};

// Register the ROCmFusionPass with the registry.
// The choice of phase number (1) is arbitrary.

REGISTER_OPTIMIZATION(kROCmFusionPassGrouping,  // grouping
                      1,                        // phase number
                      ROCmFusionPass);

REGISTER_OPTIMIZATION(kROCmFMAPassGrouping,  // grouping
                      4,                     // phase number
                      ROCmFMAPass);

//----------------------------------------------------------------------

// absract base class for an individual fusion operation
class ROCmFusionOpBase {
 public:
  ROCmFusionOpBase(Graph* g) : graph_(g) {}

  virtual ~ROCmFusionOpBase() {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  virtual bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes);

  using NodeIndexPair = std::pair<Node*, int>;

  using NodeIndexPairVec = std::vector<NodeIndexPair>;

 protected:
  struct FusionOpData {
    string op_type;  // fusion op type (_ROCmFused*)

    string op_name;  // fusion op name ( unique name for this op instance )

    string fusion_type;  // simple description, for eg: Add+Relu

    std::vector<const Node*> nodes;  // all the nodes in the fusion

    // map of input data connections
    // key is input index
    // val is vec of node-index pairs, that connect to the input index
    std::map<int, NodeIndexPairVec> data_inputs;

    // dont need node indices for control edges, so a vector suffices
    std::vector<Node*> control_inputs;

    // map of output data connections
    // key is output index
    // val is vec of node-index pairs, that the output index connects to
    std::map<int, NodeIndexPairVec> data_outputs;

    // dont need node indices for control edges, so a vector suffices
    std::vector<Node*> control_outputs;

    // map of atrribute name --> value
    std::map<string, AttrValue> attributes;

    // conveninece function to add a data input
    void add_data_input(int dst_index, Node* src_node, int src_index) {
      auto it = data_inputs.find(dst_index);
      if (it == data_inputs.end()) {
        NodeIndexPairVec inputs;
        inputs.push_back(std::make_pair(src_node, src_index));
        data_inputs[dst_index] = inputs;
      } else {
        it->second.push_back(std::make_pair(src_node, src_index));
      }
    }

    // convenience function to add a data output
    void add_data_output(int src_index, Node* dst_node, int dst_index) {
      auto it = data_outputs.find(src_index);
      if (it == data_outputs.end()) {
        NodeIndexPairVec outputs;
        outputs.push_back(std::make_pair(dst_node, dst_index));
        data_outputs[src_index] = outputs;
      } else {
        it->second.push_back(std::make_pair(dst_node, dst_index));
      }
    }

    void add_controls(const Node* node) {
      for (const Edge* e : node->in_edges())
        if (e->IsControlEdge() && !isConsumed(e->src()))
          control_inputs.push_back(e->src());
      for (const Edge* e : node->out_edges())
        if (e->IsControlEdge() && !isConsumed(e->dst()))
          control_outputs.push_back(e->dst());
    }

    // conveniece function to add an attribute
    template <typename T>
    void add_attribute(string name, T value) {
      AttrValue attr_value;
      SetAttrValue(value, &attr_value);
      attributes[name] = attr_value;
    }

    bool isConsumed(const Node* p) const {
      for (const auto x : nodes)
        if (p == x) return true;
      return false;
    }
  };

  // abstract routine that must be implemented by the derived classes.
  // this routine needs to do the following
  // ++ determine if the node sequence *ending* at the given node is a
  //    candidate for fusion
  //    ++ if it is not,
  //         return false
  //    ++ else,
  //         populate the FusionOpData details and return true
  virtual bool IsFusionEligible(const Node* n, FusionOpData* d) = 0;

 private:
  void CreateFusionOp(const FusionOpData& d,
                      std::set<const Node*>& fused_nodes);

  Graph* graph_;
};

//----------------------------------------------------------------------

// Convolution-Bias-BatchNorm-Activation Fusion
class ROCmFusionOpConvolutionBiasBatchNormActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpConvolutionBiasBatchNormActivation(Graph* g)
      : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};
//----------------------------------------------------------------------

// Convolution-Bias-Activation Fusion
class ROCmFusionOpConvolutionBiasActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpConvolutionBiasActivation(Graph* g) : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};

//----------------------------------------------------------------------

// BatchNorm-Activation (Inference + Training-Forward) Fusion
class ROCmFusionOpBatchNormActivationInference : public ROCmFusionOpBase {
 public:
  ROCmFusionOpBatchNormActivationInference(Graph* g) : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};

//----------------------------------------------------------------------

// BatchNorm-Activation (Training-Backward) Fusion
class ROCmFusionOpBatchNormActivationBackward : public ROCmFusionOpBase {
 public:
  ROCmFusionOpBatchNormActivationBackward(Graph* g) : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};

//----------------------------------------------------------------------

// Add-Relu Fusion
class ROCmFusionOpAddRelu : public ROCmFusionOpBase {
 public:
  ROCmFusionOpAddRelu(Graph* g) : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};

//----------------------------------------------------------------------

// AddN-ReluGrad Fusion
class ROCmFusionOpAddNReluGrad : public ROCmFusionOpBase {
 public:
  ROCmFusionOpAddNReluGrad(Graph* g) : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};

//----------------------------------------------------------------------

class ROCmFusionOpFMA : public ROCmFusionOpBase {
 public:
  ROCmFusionOpFMA(Graph* g) : ROCmFusionOpBase(g) {}

 protected:
  bool IsFusionEligible(const Node* n, FusionOpData* d) override;
};

//----------------------------------------------------------------------

Status ROCmFusionPass::Run(const GraphOptimizationPassOptions& options) {
  // enable the fusion pass if the env var TF_ROCM_FUSION_ENABLE is set
  if (ReadBoolFromEnvVar("TF_ROCM_FUSION_ENABLE")) {
    return ROCmFusionPassBase::Run(options, kROCmFusionPassGrouping);
  } else {
    VLOG(kVlogLevel) << "ROCmFusionPass was not enabled!";
  }
  return OkStatus();
}

Status ROCmFMAPass::Run(const GraphOptimizationPassOptions& options) {
  if (ReadBoolFromEnvVar("TF_ROCM_FMA_DISABLE")) return OkStatus();
  return ROCmFusionPassBase::Run(options, kROCmFMAPassGrouping);
}

Status ROCmFusionPassBase::Run(const GraphOptimizationPassOptions& options,
                               int grouping) {
  // Check if the graph is present, should be either in
  // - options.graph (for all but POST_PARTITIONING grouping)
  // - options.partition_graphs (for POST_PARTITIONING_grouping)
  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return OkStatus();
  }

  if (grouping == OptimizationPassRegistry::POST_PARTITIONING) {
    for (auto& pg : *options.partition_graphs) {
      if (isGpuDevice(pg.first)) {
        VLOG(kVlogLevel) << "Running ROCmFusionPass on GPU partition : "
                         << pg.first;
        // run the pass
        RunPass(pg.second.get());

      } else {
        VLOG(kVlogLevel) << "Skipping ROCmFusionPass on non-GPU partition : "
                         << pg.first;
      }
    }
  } else {
    VLOG(kVlogLevel) << "Running ROCmFusionPass on entire graph ";

    // run the pass
    RunPass(options.graph->get());
  }
  return OkStatus();
}

bool ROCmFusionPassBase::RunPass(Graph* graph) {
  if (ReadBoolFromEnvVar("TF_ROCM_FUSION_DUMP_GRAPH_BEFORE")) {
    DumpGraph(kVlogLevel, "Before running ROCmFusionPass", &*graph);
  }

  std::vector<std::unique_ptr<ROCmFusionOpBase> > fusions;

  // Initialize a vector of all the fusion operations we currently support
  InitializeFusions(fusions, graph);

  for (auto& fusion : fusions) {
    std::vector<Node*> order;
    GetPostOrder(*graph,
                 &order);  // This will give us reverse topological sort.

    std::set<const Node*> fused_nodes;

    for (const Node* n : order) {
      if (fused_nodes.count(n)) {
        // We have fused already this node...skip it
        // Note that we are traversing nodes in reverse topological order, and
        // matches are found by comparing node sequences from back to front, so
        // we will hit nodes that we have already fused into a fusion operation.
        continue;
      }

      fusion->DoFusion(n, fused_nodes);
    }
  }

  if (ReadBoolFromEnvVar("TF_ROCM_FUSION_DUMP_GRAPH_AFTER")) {
    DumpGraph(kVlogLevel, "After running ROCmFusionPass", &*graph);
  }

  return true;
}

void ROCmFusionPass::InitializeFusions(
    std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions, Graph* g) {
  if (!ReadBoolFromEnvVar("TF_ROCM_FUSION_DISABLE_CBNA")) {
    fusions.emplace_back(new ROCmFusionOpConvolutionBiasBatchNormActivation(g));
  }

  if (!ReadBoolFromEnvVar("TF_ROCM_FUSION_DISABLE_CBA")) {
    fusions.emplace_back(new ROCmFusionOpConvolutionBiasActivation(g));
  }

  if (!ReadBoolFromEnvVar("TF_ROCM_FUSION_DISABLE_BNA")) {
    fusions.emplace_back(new ROCmFusionOpBatchNormActivationInference(g));
    fusions.emplace_back(new ROCmFusionOpBatchNormActivationBackward(g));
  }

  if (!ReadBoolFromEnvVar("TF_ROCM_FUSION_DISABLE_ADDRELU")) {
    fusions.emplace_back(new ROCmFusionOpAddRelu(g));
  }

  if (!ReadBoolFromEnvVar("TF_ROCM_FUSION_DISABLE_ADDNRELUGRAD")) {
    fusions.emplace_back(new ROCmFusionOpAddNReluGrad(g));
  }
}

void ROCmFMAPass::InitializeFusions(
    std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions, Graph* g) {
  fusions.emplace_back(new ROCmFusionOpFMA(g));
}

//----------------------------------------------------------------------

// -------------------------------------------------------------
// ROCmFusionOpBase implementation
// -------------------------------------------------------------
bool ROCmFusionOpBase::DoFusion(const Node* n,
                                std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionOpData d;
  if (IsFusionEligible(n, &d)) {
    CreateFusionOp(d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

void ROCmFusionOpBase::CreateFusionOp(const FusionOpData& d,
                                      std::set<const Node*>& fused_nodes) {
  // create an instance of the fusion node
  NodeBuilder nb(d.op_name, d.op_type);

  // populate input data edges
  // this needs to be done in the correct order, and hence the strange loop
  int num_inputs = d.data_inputs.size();
  for (int i = 0; i < num_inputs; i++) {
    auto inputs = d.data_inputs.find(i)->second;
    if (inputs.size() == 1) {  // input is a singular tensor
      nb.Input(const_cast<Node*>(inputs[0].first), inputs[0].second);
    } else {  // input is a vector of tensors
      std::vector<NodeBuilder::NodeOut> vec_input;
      for (auto input : inputs) {
        vec_input.push_back(NodeBuilder::NodeOut(input.first, input.second));
      }
      nb.Input(vec_input);
    }
  }

  // populate attributes
  for (auto& kv : d.attributes) {
    nb.Attr(kv.first, kv.second);
  }

  // populate the device
  nb.Device(d.nodes[0]->def().device());

  // create the new fusion node
  Node* fusion_node = nullptr;
  TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));

  // populate the input control edges
  for (Node* n : d.control_inputs) {
    CHECK_NOTNULL(graph_->AddControlEdge(n, fusion_node, true));
  }

  // populate output data edges.
  for (auto& kv : d.data_outputs) {
    int src_index = kv.first;
    auto& outputs = kv.second;
    for (auto dst : outputs) {
      CHECK_NOTNULL(
          graph_->AddEdge(fusion_node, src_index, dst.first, dst.second));
    }
  }

  // populate output control edges
  for (Node* n : d.control_outputs) {
    CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, n, true));
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d.nodes[0]->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated Fusion Node (" << d.fusion_type << " : "
                   << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // remove the now redundant nodes from the graph
  for (const Node* n : d.nodes) {
    fused_nodes.insert(n);
    graph_->RemoveNode(const_cast<Node*>(n));
  }
}

// -------------------------------------------------------------
// ROCmFusionOpConvolutionBiasBatchNormActivation implementation
// -------------------------------------------------------------
bool ROCmFusionOpConvolutionBiasBatchNormActivation::IsFusionEligible(
    const Node* actv, FusionOpData* d) {
  // First check whether we have the right sequence of ops
  bool is_eligible = false;

  const Node* conv = nullptr;
  const Node* bias = nullptr;
  const Node* norm = nullptr;

  if (isOpActivation(actv)) {  // activation node

    TF_CHECK_OK(actv->input_node(0, &norm));
    if (isOpBatchNorm(norm)) {  // preceded by a batchnorm node

      TF_CHECK_OK(norm->input_node(0, &bias));
      if (isOpBias(bias)) {  // preceded by a bias node

        TF_CHECK_OK(bias->input_node(0, &conv));
        if (isOpConvolution(conv)) {  // preceded by a convolution node

          d->op_type = "_ROCmFusedConvolutionBiasBatchNormActivation";
          d->op_name = strings::StrCat(conv->name(), bias->name(), norm->name(),
                                       actv->name());
          d->fusion_type = "Convolution+Bias+BatchNorm+Activation";
          d->nodes.push_back(conv);
          d->nodes.push_back(bias);
          d->nodes.push_back(norm);
          d->nodes.push_back(actv);

          VLOG(kVlogLevel) << "===========";
          DumpNodeList(kVlogLevel,
                       "Found Fusion Candidate " + d->fusion_type + " : ",
                       {conv, bias, norm, actv});

          is_eligible = true;
        }
      }
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({conv, bias, norm, actv})) {
    is_eligible = false;
  }

  if (is_eligible) {
    VLOG(kVlogLevel) << "\tSkipping Fusion : Not yet supported";
    VLOG(kVlogLevel) << "===========";
    is_eligible = false;
  }

  return is_eligible;
}

//----------------------------------------------------------------------

// ----------------------------------------------------
// ROCmFusionOpConvolutionBiasActivation implementation
// ----------------------------------------------------
bool ROCmFusionOpConvolutionBiasActivation::IsFusionEligible(const Node* actv,
                                                             FusionOpData* d) {
  bool is_eligible = false;

  const Node* conv = nullptr;
  const Node* bias = nullptr;

  // First check whether we have the right sequence of ops
  if (isOpActivation(actv)) {  // activation node

    TF_CHECK_OK(actv->input_node(0, &bias));
    if (isOpBias(bias)) {  // preceded by a bias node

      TF_CHECK_OK(bias->input_node(0, &conv));
      if (isOpConvolution(conv)) {  // precedded by a convolution node

        d->op_type = "_ROCmFusedConvolutionBiasActivation";
        d->op_name = strings::StrCat(conv->name(), bias->name(), actv->name());
        d->fusion_type = "Convolution+Bias+Activation (Forward)";
        d->nodes.push_back(conv);
        d->nodes.push_back(bias);
        d->nodes.push_back(actv);

        VLOG(kVlogLevel) << "===========";
        DumpNodeList(kVlogLevel,
                     "Found Fusion Candidate " + d->fusion_type + " : ",
                     {conv, bias, actv});

        is_eligible = true;
      }
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({conv, bias, actv})) {
    is_eligible = false;
  }

  // Next check if the output of conv node feeds a node other then the bias node
  if (is_eligible) {
    for (const Edge* e : conv->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != bias)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Convolution output feeds a node other the the bias node : "
            << e->dst()->id() << ", " << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the output of bias node feeds a node other then the
  // activation node
  if (is_eligible) {
    for (const Edge* e : bias->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != actv)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Bias output feeds a node other the the activation node : "
            << e->dst()->id() << ", " << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the datatype(s) are supported
  if (is_eligible) {
    is_eligible = false;

    DataType T_conv, T_bias, T_actv;
    TF_CHECK_OK(GetNodeAttr(conv->def(), kAttr_T, &T_conv));
    TF_CHECK_OK(GetNodeAttr(bias->def(), kAttr_T, &T_bias));
    TF_CHECK_OK(GetNodeAttr(actv->def(), kAttr_T, &T_actv));

    // only float type is supported for now
    if (T_conv == DT_FLOAT) {
      // all three types must match
      if ((T_conv == T_bias) && (T_conv == T_actv)) {
        d->add_attribute(kAttr_T, T_conv);
        is_eligible = true;
      } else {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "\t DataTypes not matching : "
                         << " " << DataType_Name(T_conv) << " "
                         << DataType_Name(T_bias) << " "
                         << DataType_Name(T_actv);
        VLOG(kVlogLevel) << "===========";
      }
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " DataType not supported : " << DataType_Name(T_conv);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // Next check if the data format(s) are supported
  if (is_eligible) {
    is_eligible = false;

    string df_conv_str, df_bias_str;
    TF_CHECK_OK(GetNodeAttr(conv->def(), kAttr_data_format, &df_conv_str));
    TF_CHECK_OK(GetNodeAttr(bias->def(), kAttr_data_format, &df_bias_str));

    TensorFormat df_conv, df_bias;
    CHECK_EQ(FormatFromString(df_conv_str, &df_conv), true);
    CHECK_EQ(FormatFromString(df_bias_str, &df_bias), true);

    if ((df_conv == FORMAT_NHWC) || (df_conv == FORMAT_NCHW)) {
      if (df_conv == df_bias) {
        d->add_attribute(kAttr_data_format, ToString(df_conv));
        is_eligible = true;
      } else {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << " Data Formats not matching : "
                         << " " << ToString(df_conv) << " "
                         << ToString(df_bias);
        VLOG(kVlogLevel) << "===========";
      }
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " Data Format not supported for Fusion : "
                       << ToString(df_conv);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // Next check if the specified stride is supported
  if (is_eligible) {
    std::vector<int32> strides;
    TF_CHECK_OK(GetNodeAttr(conv->def(), kAttr_strides, &strides));

    d->add_attribute(kAttr_strides, strides);

    for (auto stride : strides) {
      // MIOpen only supports stride values of 1 or 2
      if ((stride != 1) && (stride != 2)) {
        is_eligible = false;
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << " Stride value not supported : " << stride;
        VLOG(kVlogLevel) << "===========";
        break;
      }
    }
  }

  // Next check if the specified padding is supported
  if (is_eligible) {
    string padding;
    TF_CHECK_OK(GetNodeAttr(conv->def(), kAttr_padding, &padding));

    d->add_attribute(kAttr_padding, padding);
  }

  // Next check if the specified dilation is supported
  if (is_eligible) {
    std::vector<int32> dilations;
    TF_CHECK_OK(GetNodeAttr(conv->def(), kAttr_dilations, &dilations));

    d->add_attribute(kAttr_dilations, dilations);

    for (auto dilation : dilations) {
      // MIOpen only supports dilation value of 1
      if (dilation != 1) {
        is_eligible = false;
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << " Dilation value not supported for Fusion : "
                         << dilation;
        VLOG(kVlogLevel) << "===========";
        break;
      }
    }
  }

  // finally check if the specified activation is supported
  if (is_eligible) {
    d->add_attribute(kAttr_activation_mode, actv->type_string());
  }

  if (is_eligible) {
    std::vector<const Edge*> conv_input_edges;
    TF_CHECK_OK(conv->input_edges(&conv_input_edges));

    std::vector<const Edge*> bias_input_edges;
    TF_CHECK_OK(bias->input_edges(&bias_input_edges));

    // populate input data edges
    d->add_data_input(0, conv_input_edges[0]->src(),
                      conv_input_edges[0]->src_output());
    d->add_data_input(1, conv_input_edges[1]->src(),
                      conv_input_edges[1]->src_output());
    d->add_data_input(2, bias_input_edges[1]->src(),
                      bias_input_edges[1]->src_output());

    // populate the input control edges
    for (const Edge* e : conv->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }
    for (const Edge* e : bias->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }
    for (const Edge* e : actv->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }

    // populate output data and control edges
    for (const Edge* e : conv->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      }
    }
    for (const Edge* e : bias->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      }
    }
    for (const Edge* e : actv->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      } else {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
  }

  return is_eligible;
}

//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpBatchNormActivationInference implementation
// ----------------------------------------------

bool ROCmFusionOpBatchNormActivationInference::IsFusionEligible(
    const Node* actv, FusionOpData* d) {
  // First check whether we have the right sequence of ops
  bool is_eligible = false;

  const Node* norm = nullptr;
  bool is_training = false;

  if (isOpActivation(actv)) {  // activation node

    TF_CHECK_OK(actv->input_node(0, &norm));
    if (isOpBatchNorm(norm)) {  // preceded by a batchnorm node

      // check the is_training attribute to determine the type of op to create
      // (i.e. training version or inference version)
      TF_CHECK_OK(GetNodeAttr(norm->def(), kAttr_is_training, &is_training));
      if (is_training) {
        d->op_type = "_ROCmFusedBatchNormActivationForward";
        d->op_name = strings::StrCat(norm->name(), actv->name());
        d->fusion_type = "BatchNorm+Activation (training-fwd)";

      } else {
        d->op_type = "_ROCmFusedBatchNormActivationInference";
        d->op_name = strings::StrCat(norm->name(), actv->name());
        d->fusion_type = "BatchNorm+Activation (inference)";
      }

      d->nodes.push_back(norm);
      d->nodes.push_back(actv);

      VLOG(kVlogLevel) << "===========";
      DumpNodeList(kVlogLevel,
                   "Found Fusion Candidate " + d->fusion_type + " : ",
                   {norm, actv});

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({norm, actv})) {
    is_eligible = false;
  }

  // Next check if the first output of batch-norm node feeds a node other then
  // the activation node
  if (is_eligible) {
    for (const Edge* e : norm->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != actv)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "BatchNorm output feeds a node other the the activation node : "
            << e->dst()->id() << ", " << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the datatype(s) are supported
  if (is_eligible) {
    is_eligible = false;

    DataType U_norm = DT_FLOAT;
    if (norm->type_string() == "FusedBatchNorm") {
      TF_CHECK_OK(GetNodeAttr(norm->def(), kAttr_T, &U_norm));
    } else if (norm->type_string() == "FusedBatchNormV2") {
      TF_CHECK_OK(GetNodeAttr(norm->def(), kAttr_U, &U_norm));
    }

    if (U_norm == DT_FLOAT) {
      d->add_attribute(kAttr_U, U_norm);
      is_eligible = true;
    } else {
      VLOG(kVlogLevel)
          << "\tSkipping Fusion : "
          << "BatchNorm scale/offset/mean/variance datatype not supported : "
          << DataType_Name(U_norm);
      VLOG(kVlogLevel) << "===========";
    }
  }

  if (is_eligible) {
    is_eligible = false;

    DataType T_norm, T_actv;
    TF_CHECK_OK(GetNodeAttr(norm->def(), kAttr_T, &T_norm));
    TF_CHECK_OK(GetNodeAttr(actv->def(), kAttr_T, &T_actv));

    // only float and half types are supported for now
    if ((T_norm == DT_FLOAT) || (T_norm == DT_HALF)) {

      // datatype for batch-norm and activation must match
      if (T_norm == T_actv) {
        d->add_attribute(kAttr_T, T_norm);
        is_eligible = true;
      } else {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "\t DataTypes not matching : "
                         << " " << DataType_Name(T_norm) << " "
                         << DataType_Name(T_actv);
        VLOG(kVlogLevel) << "===========";
      }
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " DataType not supported : " << DataType_Name(T_norm);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // Next check if the data format(s) are supported
  if (is_eligible) {
    is_eligible = false;

    string df_norm_str;
    TF_CHECK_OK(GetNodeAttr(norm->def(), kAttr_data_format, &df_norm_str));

    TensorFormat df_norm;
    CHECK_EQ(FormatFromString(df_norm_str, &df_norm), true);

    if ((df_norm == FORMAT_NHWC) || (df_norm == FORMAT_NCHW)) {
      d->add_attribute(kAttr_data_format, ToString(df_norm));
      is_eligible = true;
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " Data Format not supported for Fusion : "
                       << ToString(df_norm);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // Next check the epsilon value is supported
  if (is_eligible) {
    float epsilon = 0.0;
    TF_CHECK_OK(GetNodeAttr(norm->def(), kAttr_epsilon, &epsilon));

    d->add_attribute(kAttr_epsilon, epsilon);
  }

  // finally check if the specified activation is supported
  if (is_eligible) {
    d->add_attribute(kAttr_activation_mode, actv->type_string());
  }

  if (is_eligible) {
    std::vector<const Edge*> norm_input_edges;
    TF_CHECK_OK(norm->input_edges(&norm_input_edges));

    // populate input data edges

    // batchnorm x
    d->add_data_input(0, norm_input_edges[0]->src(),
                      norm_input_edges[0]->src_output());
    // batchnorm scale
    d->add_data_input(1, norm_input_edges[1]->src(),
                      norm_input_edges[1]->src_output());
    // batchnorm offset
    d->add_data_input(2, norm_input_edges[2]->src(),
                      norm_input_edges[2]->src_output());

    if (!is_training) {
      // batchnorm mean
      d->add_data_input(3, norm_input_edges[3]->src(),
                        norm_input_edges[3]->src_output());
      // batchnorm variance
      d->add_data_input(4, norm_input_edges[4]->src(),
                        norm_input_edges[4]->src_output());
    }

    // populate the input control edges
    for (const Edge* e : norm->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }
    for (const Edge* e : actv->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }

    // populate output data and control edges
    for (const Edge* e : norm->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      } else if (is_training) {
        if ((1 <= e->src_output()) && (e->src_output() <= 5)) {
          // 1 - batch-norm mean
          // 2 - batch-norm variance
          // 3 - saved mean
          // 4 - saved variance
          // 5 - unused (in ROCm) reserved space 3
          d->add_data_output(e->src_output(), e->dst(), e->dst_input());
        } else {
          // only other output index should be 0 (which feeds the actv node)
          CHECK_EQ(e->src_output(), 0);
        }
      }
    }
    for (const Edge* e : actv->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      } else {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
  }

  return is_eligible;
}
//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpBatchNormActivationBackward implementationq
// ----------------------------------------------

bool ROCmFusionOpBatchNormActivationBackward::IsFusionEligible(
    const Node* norm_grad, FusionOpData* d) {
  // First check whether we have the right sequence of ops
  bool is_eligible = false;

  const Node* actv_grad = nullptr;
  const Node* norm_offset = nullptr;

  if (isOpBatchNormGrad(norm_grad)) {  // batchnorm gradient node

    TF_CHECK_OK(norm_grad->input_node(0, &actv_grad));
    if (isOpActivationGrad(
            actv_grad)) {  // preceded by a activation graident node

      // we need to cache the offset input to the batchnorm node
      // this is because the fused-op needs this input, and it is not present
      // as an input on the batchnorm grad node (apparently TF does not need it)
      const Edge* e = nullptr;

      // 4th input to batchnorm grad is the saved mean from batchnorn fwd
      TF_CHECK_OK(norm_grad->input_edge(3, &e));
      const Node* norm = e->src();

      // 3rd input to batchnorm fwd is the offset node we want
      TF_CHECK_OK(norm->input_node(2, &norm_offset));

      d->op_type = "_ROCmFusedBatchNormActivationBackward";
      d->op_name = strings::StrCat(norm_grad->name(), actv_grad->name());
      d->fusion_type = "BatchNorm+Activation (training-bwd)";
      d->nodes.push_back(norm_grad);
      d->nodes.push_back(actv_grad);

      VLOG(kVlogLevel) << "===========";
      DumpNodeList(kVlogLevel,
                   "Found Fusion Candidate " + d->fusion_type + " : ",
                   {norm_grad, actv_grad});

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({norm_grad, actv_grad})) {
    is_eligible = false;
  }

  // Next check if the first output of activation gradient node feeds a node
  // other then the batch norm gradient node
  if (is_eligible) {
    for (const Edge* e : actv_grad->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != norm_grad)) {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "ActivationGrad output feeds a node other the the "
                            "BatchNormGrad node : "
                         << e->dst()->id() << ", " << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // BatchNormGrad has two "reserved" outputs, we do not know what is output
  // from them, so check to make sure they are not connected. If they are
  // we cannot fuse the node
  if (is_eligible) {
    for (const Edge* e : norm_grad->out_edges()) {
      if (!e->IsControlEdge() && (e->src_output() > 2)) {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "BatchNormGrad reserved output node (idx = "
                         << e->src_output()
                         << " ) has a connection : " << e->dst()->id() << ", "
                         << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the datatype(s) are supported
  if (is_eligible) {
    is_eligible = false;

    DataType U_norm_grad = DT_FLOAT;
    if (norm_grad->type_string() == "FusedBatchNormGrad") {
      TF_CHECK_OK(GetNodeAttr(norm_grad->def(), kAttr_T, &U_norm_grad));
    } else if (norm_grad->type_string() == "FusedBatchNormGradV2") {
      TF_CHECK_OK(GetNodeAttr(norm_grad->def(), kAttr_U, &U_norm_grad));
    }

    if (U_norm_grad == DT_FLOAT) {
      d->add_attribute(kAttr_U, U_norm_grad);
      is_eligible = true;
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << "BatchNormGrad scale/offset/mean/variance datatype "
                          "not supported : "
                       << DataType_Name(U_norm_grad);
      VLOG(kVlogLevel) << "===========";
    }
  }

  if (is_eligible) {
    is_eligible = false;

    DataType T_norm_grad, T_actv_grad;
    TF_CHECK_OK(GetNodeAttr(norm_grad->def(), kAttr_T, &T_norm_grad));
    TF_CHECK_OK(GetNodeAttr(actv_grad->def(), kAttr_T, &T_actv_grad));

    // only float and half types are supported for now
    if ((T_norm_grad == DT_FLOAT) || (T_norm_grad == DT_HALF)) {

      // datatype for batch-norm and activation must match
      if (T_norm_grad == T_actv_grad) {
        d->add_attribute(kAttr_T, T_norm_grad);
        is_eligible = true;
      } else {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "\t DataTypes not matching : "
                         << " " << DataType_Name(T_norm_grad) << " "
                         << DataType_Name(T_actv_grad);
        VLOG(kVlogLevel) << "===========";
      }
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " DataType not supported : "
                       << DataType_Name(T_norm_grad);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // Next check if the data format(s) are supported
  if (is_eligible) {
    is_eligible = false;

    string df_norm_grad_str;
    TF_CHECK_OK(
        GetNodeAttr(norm_grad->def(), kAttr_data_format, &df_norm_grad_str));

    TensorFormat df_norm_grad;
    CHECK_EQ(FormatFromString(df_norm_grad_str, &df_norm_grad), true);

    if ((df_norm_grad == FORMAT_NHWC) || (df_norm_grad == FORMAT_NCHW)) {
      d->add_attribute(kAttr_data_format, ToString(df_norm_grad));
      is_eligible = true;
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " Data Format not supported for Fusion : "
                       << ToString(df_norm_grad);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // Next check the epsilon value is supported
  if (is_eligible) {
    float epsilon = 0.0;
    TF_CHECK_OK(GetNodeAttr(norm_grad->def(), kAttr_epsilon, &epsilon));

    d->add_attribute(kAttr_epsilon, epsilon);
  }

  // finally check if the specified activation is supported
  if (is_eligible) {
    d->add_attribute(kAttr_activation_mode, getActivationOpType(actv_grad));
  }

  if (is_eligible) {
    std::vector<const Edge*> actv_grad_input_edges;
    TF_CHECK_OK(actv_grad->input_edges(&actv_grad_input_edges));

    std::vector<const Edge*> norm_grad_input_edges;
    TF_CHECK_OK(norm_grad->input_edges(&norm_grad_input_edges));

    // populate input data edges

    // activation grad gradients
    d->add_data_input(0, actv_grad_input_edges[0]->src(),
                      actv_grad_input_edges[0]->src_output());
    // activation grad features
    d->add_data_input(1, actv_grad_input_edges[1]->src(),
                      actv_grad_input_edges[1]->src_output());

    // batchnorm grad x
    d->add_data_input(2, norm_grad_input_edges[1]->src(),
                      norm_grad_input_edges[1]->src_output());
    // batchnorm grad scale
    d->add_data_input(3, norm_grad_input_edges[2]->src(),
                      norm_grad_input_edges[2]->src_output());
    // batchnorm offset
    d->add_data_input(4, const_cast<Node*>(norm_offset), 0);
    // batchnorm grad saved_mean
    d->add_data_input(5, norm_grad_input_edges[3]->src(),
                      norm_grad_input_edges[3]->src_output());
    // batchnorm grad saved_var
    d->add_data_input(6, norm_grad_input_edges[4]->src(),
                      norm_grad_input_edges[4]->src_output());

    // populate the input control edges
    for (const Edge* e : norm_grad->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }
    for (const Edge* e : actv_grad->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }

    // populate output data and control edges
    for (const Edge* e : actv_grad->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      }
    }
    for (const Edge* e : norm_grad->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      } else {
        if ((0 <= e->src_output()) && (e->src_output() <= 2)) {
          // 0 - x_bn_backprop
          // 1 - scale_backprop
          // 2 - offset_backprop
          d->add_data_output(e->src_output(), e->dst(), e->dst_input());
        } else {
          LOG(FATAL)
              << "Unexpected output connection on the BatchNormGrad node :"
              << norm_grad << ". output(idx, dest_node) :" << e->src_output()
              << ", " << e->dst();
        }
      }
    }
  }

  return is_eligible;
}
//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpAddRelu implementation
// ----------------------------------------------

bool ROCmFusionOpAddRelu::IsFusionEligible(const Node* relu, FusionOpData* d) {
  bool is_eligible = false;

  const Node* add = nullptr;

  // First check whether we have the right sequence of ops
  if (isOpRelu(relu)) {  // "Relu" node

    TF_CHECK_OK(relu->input_node(0, &add));

    if (isOpAdd(add)) {  // preceded by a "Add" op

      d->op_type = "_ROCmFusedAddRelu";
      d->op_name = strings::StrCat(add->name(), relu->name());
      d->fusion_type = "Add+Relu";
      d->nodes.push_back(add);
      d->nodes.push_back(relu);

      VLOG(kVlogLevel) << "===========";
      DumpNodeList(kVlogLevel,
                   "Found Fusion Candidate " + d->fusion_type + " : ",
                   {add, relu});

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({add, relu})) {
    is_eligible = false;
  }

  // Next check if the first output of add node feeds a node other then
  // the activation node
  if (is_eligible) {
    for (const Edge* e : add->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != relu)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Output from Add also feeds a node other then Relu : "
            << e->dst()->id() << ", " << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the datatype(s) are supported
  if (is_eligible) {
    is_eligible = false;

    DataType T_add, T_relu;
    TF_CHECK_OK(GetNodeAttr(add->def(), kAttr_T, &T_add));
    TF_CHECK_OK(GetNodeAttr(relu->def(), kAttr_T, &T_relu));

    // only float and half types are supported for now
    if ((T_add == DT_FLOAT) || (T_add == DT_HALF)) {
      // datatype for add and relu must match
      if (T_add == T_relu) {
        d->add_attribute(kAttr_T, T_add);
        is_eligible = true;
      } else {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "\t DataTypes not matching : "
                         << " " << DataType_Name(T_add) << " "
                         << DataType_Name(T_relu);
        VLOG(kVlogLevel) << "===========";
      }
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " DataType not supported : " << DataType_Name(T_add);
      VLOG(kVlogLevel) << "===========";
    }
  }

  if (is_eligible) {
    std::vector<const Edge*> add_input_edges;
    TF_CHECK_OK(add->input_edges(&add_input_edges));

    // populate input data edges
    d->add_data_input(0, add_input_edges[0]->src(),
                      add_input_edges[0]->src_output());
    d->add_data_input(1, add_input_edges[1]->src(),
                      add_input_edges[1]->src_output());

    // populate the input control edges
    for (const Edge* e : add->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }
    for (const Edge* e : relu->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }

    // populate output data and control edges
    for (const Edge* e : add->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      }
    }
    for (const Edge* e : relu->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      } else {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
  }

  return is_eligible;
}

bool ROCmFusionOpFMA::IsFusionEligible(const Node* node, FusionOpData* d) {
  bool add = isOpAddX(node);
  bool sub = isOpSub(node);
  if (!add && !sub) return false;

  if (node->in_edges().size() != 2) return false;
  // todo: can we reject if node output is 7+ dim?

  DataType dtype;
  TF_CHECK_OK(GetNodeAttr(node->def(), kAttr_T, &dtype));
  if (!(dtype == DT_HALF || dtype == DT_FLOAT || dtype == DT_DOUBLE))
    return false;

  Node *b, *c;
  TF_CHECK_OK(node->input_node(0, &b));
  TF_CHECK_OK(node->input_node(1, &c));

  if (!areAssignedToSameGpu({node, b, c})) return false;
  VLOG(kVlogLevel) << node;
  VLOG(kVlogLevel) << "Trying to fuse " << node->type_string() << " "
                   << node->in_edges().size() << b->type_string() << " "
                   << b->in_edges().size() << " " << b->out_edges().size()
                   << " " << c->type_string() << " " << c->in_edges().size()
                   << " " << c->out_edges().size();
  bool can_absorb_b = (b->type_string() == "Mul" && b->in_edges().size() == 2 &&
                       b->out_edges().size() == 1);
  bool can_absorb_c = (c->type_string() == "Mul" && c->in_edges().size() == 2 &&
                       c->out_edges().size() == 1);

  if (can_absorb_b && can_absorb_c) {
    d->op_type = add ? "_FusedMulAdd2" : "_FusedMulSub2";
    d->op_name = strings::StrCat(b->name(), c->name());
    d->fusion_type = d->op_type;
    d->nodes.push_back(node);
    d->nodes.push_back(b);
    d->nodes.push_back(c);
    d->add_attribute(kAttr_T, dtype);

    std::vector<const Edge*> b_input_edges;
    TF_CHECK_OK(b->input_edges(&b_input_edges));

    std::vector<const Edge*> c_input_edges;
    TF_CHECK_OK(c->input_edges(&c_input_edges));

    d->add_data_input(0, b_input_edges[0]->src(),
                      b_input_edges[0]->src_output());
    d->add_data_input(1, b_input_edges[1]->src(),
                      b_input_edges[1]->src_output());
    d->add_data_input(2, c_input_edges[0]->src(),
                      c_input_edges[0]->src_output());
    d->add_data_input(3, c_input_edges[1]->src(),
                      c_input_edges[1]->src_output());

    d->add_controls(b);
    d->add_controls(c);
    d->add_controls(node);

    for (const Edge* e : node->out_edges()) {
      if (!e->IsControlEdge()) {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
    return true;
  } else if (can_absorb_b) {
    d->op_type = add ? "_FusedMulAdd" : "_FusedMulSub";
    d->op_name = strings::StrCat(b->name(), c->name());
    d->fusion_type = d->op_type;
    d->nodes.push_back(node);
    d->nodes.push_back(b);
    d->add_attribute(kAttr_T, dtype);

    std::vector<const Edge*> b_input_edges;
    TF_CHECK_OK(b->input_edges(&b_input_edges));

    std::vector<const Edge*> node_input_edges;
    TF_CHECK_OK(node->input_edges(&node_input_edges));

    d->add_data_input(0, b_input_edges[0]->src(),
                      b_input_edges[0]->src_output());
    d->add_data_input(1, b_input_edges[1]->src(),
                      b_input_edges[1]->src_output());
    d->add_data_input(2, node_input_edges[1]->src(),
                      node_input_edges[1]->src_output());

    d->add_controls(b);
    d->add_controls(node);

    for (const Edge* e : node->out_edges()) {
      if (!e->IsControlEdge()) {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
    return true;
  } else if (can_absorb_c) {
    d->op_type = add ? "_FusedMulAdd" : "_FusedMulSubRev";
    d->op_name = strings::StrCat(b->name(), c->name());
    d->fusion_type = d->op_type;
    d->nodes.push_back(node);
    d->nodes.push_back(c);
    d->add_attribute(kAttr_T, dtype);

    std::vector<const Edge*> c_input_edges;
    TF_CHECK_OK(c->input_edges(&c_input_edges));

    std::vector<const Edge*> node_input_edges;
    TF_CHECK_OK(node->input_edges(&node_input_edges));

    d->add_data_input(0, c_input_edges[0]->src(),
                      c_input_edges[0]->src_output());
    d->add_data_input(1, c_input_edges[1]->src(),
                      c_input_edges[1]->src_output());
    d->add_data_input(2, node_input_edges[0]->src(),
                      node_input_edges[0]->src_output());

    d->add_controls(c);
    d->add_controls(node);

    for (const Edge* e : node->out_edges()) {
      if (!e->IsControlEdge()) {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
    return true;
  }
  return false;
}

//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpAddNReluGrad implementation
// ----------------------------------------------

bool ROCmFusionOpAddNReluGrad::IsFusionEligible(const Node* reluGrad,
                                                FusionOpData* d) {
  bool is_eligible = false;

  const Node* addN = nullptr;
  const Node* relu = nullptr;

  // First check whether we have the right sequence of ops
  if (isOpReluGrad(reluGrad)) {  // "ReluGrad" node

    TF_CHECK_OK(reluGrad->input_node(0, &addN));
    TF_CHECK_OK(reluGrad->input_node(1, &relu));

    if (isOpAddN(addN) && isOpReluOrFusedAddRelu(relu)) {
      // preceded by "AddN" and "Relu"/"_ROCmFusedAddRelu" Ops

      d->op_type = "_ROCmFusedAddNReluGrad";
      d->op_name = strings::StrCat(addN->name(), reluGrad->name());
      d->fusion_type = "AddN+ReluGrad";
      d->nodes.push_back(addN);
      d->nodes.push_back(reluGrad);

      DumpNodeList(kVlogLevel,
                   "Found Fusion Candidate " + d->fusion_type + " : ",
                   {addN, reluGrad});

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({addN, relu, reluGrad})) {
    is_eligible = false;
  }

  // Next check if the first output of addN node feeds a node other then
  // the activation node
  if (is_eligible) {
    for (const Edge* e : addN->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != reluGrad)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Output from AddN also feeds a node other then ReluGrad : "
            << e->dst()->id() << ", " << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the datatype(s) are supported
  if (is_eligible) {
    is_eligible = false;

    DataType T_addN, T_relu, T_reluGrad;
    TF_CHECK_OK(GetNodeAttr(addN->def(), kAttr_T, &T_addN));
    TF_CHECK_OK(GetNodeAttr(relu->def(), kAttr_T, &T_relu));
    TF_CHECK_OK(GetNodeAttr(reluGrad->def(), kAttr_T, &T_reluGrad));

    // only float and half types are supported for now
    if ((T_addN == DT_FLOAT) || (T_addN == DT_HALF)) {
      // datatype for binary-op and activation must match
      if ((T_addN == T_relu) && (T_addN == T_reluGrad)) {
        d->add_attribute(kAttr_T, T_addN);
        is_eligible = true;
      } else {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "\t DataTypes not matching : "
                         << " " << DataType_Name(T_addN) << " "
                         << DataType_Name(T_relu) << " "
                         << DataType_Name(T_reluGrad);
        VLOG(kVlogLevel) << "===========";
      }
    } else {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " DataType not supported : " << DataType_Name(T_addN);
      VLOG(kVlogLevel) << "===========";
    }
  }

  // scan in the number of inputs to the AddN node
  if (is_eligible) {
    int num_addN_inputs;
    TF_CHECK_OK(GetNodeAttr(addN->def(), kAttr_N, &num_addN_inputs));
    if (num_addN_inputs != 2) {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " AddN node has unsupported N value : "
                       << num_addN_inputs;
      VLOG(kVlogLevel) << "===========";
      is_eligible = false;
    } else {
      d->add_attribute(kAttr_N, num_addN_inputs);
    }
  }

  if (is_eligible) {
    std::vector<const Edge*> addN_input_edges;
    TF_CHECK_OK(addN->input_edges(&addN_input_edges));

    std::vector<const Edge*> reluGrad_input_edges;
    TF_CHECK_OK(reluGrad->input_edges(&reluGrad_input_edges));

    // populate input data edges
    // add addN inputs
    for (const Edge* e : addN_input_edges) {
      d->add_data_input(0, e->src(), e->src_output());
    }

    // add the relu output
    d->add_data_input(1, reluGrad_input_edges[1]->src(),
                      reluGrad_input_edges[1]->src_output());

    // populate the input control edges
    for (const Edge* e : addN->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }
    for (const Edge* e : reluGrad->in_edges()) {
      if (e->IsControlEdge()) {
        d->control_inputs.push_back(e->src());
      }
    }

    // populate output data and control edges
    for (const Edge* e : addN->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      }
    }
    for (const Edge* e : reluGrad->out_edges()) {
      if (e->IsControlEdge()) {
        d->control_outputs.push_back(e->dst());
      } else {
        CHECK_EQ(e->src_output(), 0);
        d->add_data_output(0, e->dst(), e->dst_input());
      }
    }
  }

  return is_eligible;
}
//----------------------------------------------------------------------

}  // namespace gpu_fusion_pass
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
