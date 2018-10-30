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

#include "tensorflow/core/graph/gpu_fusion_pass.h"

namespace tensorflow {
namespace gpu_fusion_pass {

//----------------------------------------------------------------------

// declaration of all the constants that are used in the gpu_fusion_pass
// namespace
const int kVlogLevel = -1;

// an adapter on the util version of the ReadBoolFromEnvVar routine
// this version better suits the usage of the routine in this file.
// main modifications are
// 1. default value is hard-coded to false
// 2. return value is the bool
//    - true if env-var value is set to 1/true
//    - false if env-var value is either not set or set to any other value
bool ReadBoolFromEnvVar(const char* env_var_name) {
  bool value = false;
  tensorflow::ReadBoolFromEnvVar(env_var_name, false, &value);
  return value;
}

// graph pass grouping for this fusion pass
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
          (n->type_string() == "FusedBatchNormV2"));
}

// is this node an instance of a batchnorm gradient op for which we support
// fusion for?
inline bool isOpBatchNormGrad(const Node* n) {
  return ((n->type_string() == "FusedBatchNormGrad") ||
          (n->type_string() == "FusedBatchNormGradV2"));
}

// is this node an instance of the "Add" op?
inline bool isOpAdd(const Node* n) { return (n->type_string() == "Add"); }

// is this node an instance of the "AddN" op?
inline bool isOpAddN(const Node* n) { return (n->type_string() == "AddN"); }

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

class ROCmFusionPass : public GraphOptimizationPass {
 public:
  // optimization pass entry point,
  // application code will call this routine to run the pass
  Status Run(const GraphOptimizationPassOptions& options) override;

  // helper function that does all the work for this pass
  bool RunPass(Graph* g);

 private:
  void InitializeFusions(
      std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions, Graph* g);
};

// Register the ROCmFusionPass with the registry.
// The choice of phase number (1) is arbitrary.

REGISTER_OPTIMIZATION(kROCmFusionPassGrouping,  // grouping
                      1,                        // phase number
                      ROCmFusionPass);

//----------------------------------------------------------------------

// absract base class for an individual fusion operation
class ROCmFusionOpBase {
 public:
  ROCmFusionOpBase(Graph* g) : graph_(g) {}

  virtual ~ROCmFusionOpBase() {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  virtual bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) = 0;

  using NodeIndexPair = std::pair<const Node*, int>;

 protected:
  Graph* graph_;
};

//----------------------------------------------------------------------

// Convolution-Bias-BatchNorm-Activation Fusion
class ROCmFusionOpConvolutionBiasBatchNormActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpConvolutionBiasBatchNormActivation(Graph* g)
      : ROCmFusionOpBase(g) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) override;

 protected:
  struct FusionData {
    const Node* conv;
    const Node* bias;
    const Node* norm;
    const Node* actv;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d, std::set<const Node*>& fused_nodes);
};

//----------------------------------------------------------------------

// Convolution-Bias-Activation Fusion
class ROCmFusionOpConvolutionBiasActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpConvolutionBiasActivation(Graph* g) : ROCmFusionOpBase(g) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) override;

 protected:
  struct FusionData {
    const Node* conv;
    const Node* bias;
    const Node* actv;

    DataType data_type;
    std::vector<int32> strides;
    string padding;
    string data_format;
    std::vector<int32> dilations;
    string activation_mode;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d, std::set<const Node*>& fused_nodes);
};

//----------------------------------------------------------------------

// BatchNorm-Activation (Inference + Training-Forward) Fusion
class ROCmFusionOpBatchNormActivationInference : public ROCmFusionOpBase {
 public:
  ROCmFusionOpBatchNormActivationInference(Graph* g) : ROCmFusionOpBase(g) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) override;

 protected:
  struct FusionData {
    const Node* norm;
    const Node* actv;

    bool is_training;
    DataType data_type;
    float epsilon;
    string data_format;
    string activation_mode;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d, std::set<const Node*>& fused_nodes);
};

//----------------------------------------------------------------------

// BatchNorm-Activation (Training-Backward) Fusion
class ROCmFusionOpBatchNormActivationBackward : public ROCmFusionOpBase {
 public:
  ROCmFusionOpBatchNormActivationBackward(Graph* g) : ROCmFusionOpBase(g) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) override;

 protected:
  struct FusionData {
    const Node* norm_grad;
    const Node* actv_grad;

    const Node* norm_offset;

    DataType data_type;
    float epsilon;
    string data_format;
    string activation_mode;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d, std::set<const Node*>& fused_nodes);
};

//----------------------------------------------------------------------

// Add-Relu Fusion
class ROCmFusionOpAddRelu : public ROCmFusionOpBase {
 public:
  ROCmFusionOpAddRelu(Graph* g) : ROCmFusionOpBase(g) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) override;

 protected:
  struct FusionData {
    const Node* add;
    const Node* relu;

    DataType data_type;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d, std::set<const Node*>& fused_nodes);
};

//----------------------------------------------------------------------

// AddN-Relu-ReluGrad Fusion
class ROCmFusionOpAddNReluGrad : public ROCmFusionOpBase {
 public:
  ROCmFusionOpAddNReluGrad(Graph* g) : ROCmFusionOpBase(g) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n, std::set<const Node*>& fused_nodes) override;

 protected:
  struct FusionData {
    const Node* addN;
    const Node* relu;
    const Node* reluGrad;

    DataType data_type;
    int num_addN_inputs;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d, std::set<const Node*>& fused_nodes);
};

//----------------------------------------------------------------------

Status ROCmFusionPass::Run(const GraphOptimizationPassOptions& options) {
  // enable the fusion pass if the env var TF_ROCM_FUSION_ENABLE is set
  if (ReadBoolFromEnvVar("TF_ROCM_FUSION_ENABLE")) {
    // Check if the graph is present, should be either in
    // - options.graph (for all but POST_PARTITIONING grouping)
    // - options.partition_graphs (for POST_PARTITIONING_grouping)
    if (options.graph == nullptr && options.partition_graphs == nullptr) {
      return Status::OK();
    }

    if (kROCmFusionPassGrouping ==
        OptimizationPassRegistry::POST_PARTITIONING) {
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

  } else {
    VLOG(kVlogLevel) << "ROCmFusionPass was not enabled!";
  }

  return Status::OK();
}

bool ROCmFusionPass::RunPass(Graph* graph) {
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

//----------------------------------------------------------------------

// -------------------------------------------------------------
// ROCmFusionOpConvolutionBiasBatchNormActivation implementation
// -------------------------------------------------------------
bool ROCmFusionOpConvolutionBiasBatchNormActivation::DoFusion(
    const Node* n4, std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n4, &d)) {
    CreateFusionOp(&d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpConvolutionBiasBatchNormActivation::IsFusionEligible(
    const Node* n4, FusionData* d) {
  // First check whether we have the right sequence of ops
  bool is_eligible = false;
  if (isOpActivation(n4)) {  // activation node
    Node* n3 = nullptr;
    TF_CHECK_OK(n4->input_node(0, &n3));
    if (isOpBatchNorm(n3)) {  // preceded by a batchnorm node
      Node* n2 = nullptr;
      TF_CHECK_OK(n3->input_node(0, &n2));
      if (isOpBias(n2)) {  // preceded by a bias node
        Node* n1 = nullptr;
        TF_CHECK_OK(n2->input_node(0, &n1));
        if (isOpConvolution(n1)) {  // preceded by a convolution node
          d->conv = n1;
          d->bias = n2;
          d->norm = n3;
          d->actv = n4;
          VLOG(kVlogLevel) << "===========";
          DumpNodeList(
              kVlogLevel,
              "Found Fusion Candidate Convolution+Bias+BatchNorm+Activation : ",
              {d->conv, d->bias, d->norm, d->actv});
          is_eligible = true;
        }
      }
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible &&
      !areAssignedToSameGpu({d->conv, d->bias, d->norm, d->actv})) {
    is_eligible = false;
  }

  return is_eligible;
}

void ROCmFusionOpConvolutionBiasBatchNormActivation::CreateFusionOp(
    const FusionData* d, std::set<const Node*>& fused_nodes) {
  // todo
}
//----------------------------------------------------------------------

// ----------------------------------------------------
// ROCmFusionOpConvolutionBiasActivation implementation
// ----------------------------------------------------
bool ROCmFusionOpConvolutionBiasActivation::DoFusion(
    const Node* n3, std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n3, &d)) {
    CreateFusionOp(&d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpConvolutionBiasActivation::IsFusionEligible(const Node* n3,
                                                             FusionData* d) {
  bool is_eligible = false;

  // First check whether we have the right sequence of ops
  if (isOpActivation(n3)) {  // activation node
    Node* n2 = nullptr;
    TF_CHECK_OK(n3->input_node(0, &n2));
    if (isOpBias(n2)) {  // preceded by a bias node
      Node* n1 = nullptr;
      TF_CHECK_OK(n2->input_node(0, &n1));
      if (isOpConvolution(n1)) {  // precedded by a convolution node
        d->conv = n1;
        d->bias = n2;
        d->actv = n3;
        VLOG(kVlogLevel) << "===========";
        DumpNodeList(kVlogLevel,
                     "Found Fusion Candidate Convolution+Bias+Activation : ",
                     {d->conv, d->bias, d->actv});
        is_eligible = true;
      }
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({d->conv, d->bias, d->actv})) {
    is_eligible = false;
  }

  // Next check if the output of conv node feeds a node other then the bias node
  if (is_eligible) {
    for (const Edge* e : d->conv->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != d->bias)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Convolution output feeds a node other the the bias node : "
            << e->dst()->name();
        VLOG(kVlogLevel) << "===========";
        is_eligible = false;
        break;
      }
    }
  }

  // Next check if the output of bias node feeds a node other then the
  // activation node
  if (is_eligible) {
    for (const Edge* e : d->bias->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != d->actv)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Bias output feeds a node other the the activation node : "
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

    DataType T_conv, T_bias, T_actv;
    TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_T, &T_conv));
    TF_CHECK_OK(GetNodeAttr(d->bias->def(), kAttr_T, &T_bias));
    TF_CHECK_OK(GetNodeAttr(d->actv->def(), kAttr_T, &T_actv));

    // only float type is supported for now
    if (T_conv == DT_FLOAT) {
      // all three types must match
      if ((T_conv == T_bias) && (T_conv == T_actv)) {
        d->data_type = T_conv;
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
    TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_data_format, &df_conv_str));
    TF_CHECK_OK(GetNodeAttr(d->bias->def(), kAttr_data_format, &df_bias_str));

    TensorFormat df_conv, df_bias;
    CHECK_EQ(FormatFromString(df_conv_str, &df_conv), true);
    CHECK_EQ(FormatFromString(df_bias_str, &df_bias), true);

    if ((df_conv == FORMAT_NHWC) || (df_conv == FORMAT_NCHW)) {
      if (df_conv == df_bias) {
        d->data_format = ToString(df_conv);
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

    TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_strides, &d->strides));
    for (auto stride : d->strides) {
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
    TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_padding, &d->padding));
  }

  // Next check if the specified dilation is supported
  if (is_eligible) {

    TF_CHECK_OK(GetNodeAttr(d->conv->def(), kAttr_dilations, &d->dilations));
    for (auto dilation : d->dilations) {
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
    d->activation_mode = d->actv->type_string();
  }

  return is_eligible;
}

void ROCmFusionOpConvolutionBiasActivation::CreateFusionOp(
    const FusionData* d, std::set<const Node*>& fused_nodes) {
  std::vector<const Edge*> conv_input_edges;
  TF_CHECK_OK(d->conv->input_edges(&conv_input_edges));

  std::vector<const Edge*> bias_input_edges;
  TF_CHECK_OK(d->bias->input_edges(&bias_input_edges));

  // create an instance of the fusion node
  string op_name =
      strings::StrCat(d->conv->name(), d->bias->name(), d->actv->name());

  NodeBuilder nb(op_name, "_ROCmFusedConvolutionBiasActivation");

  // populate input data edges
  nb.Input(conv_input_edges[0]->src(), conv_input_edges[0]->src_output());
  nb.Input(conv_input_edges[1]->src(), conv_input_edges[1]->src_output());
  nb.Input(bias_input_edges[1]->src(), bias_input_edges[1]->src_output());

  // populate attributes
  nb.Attr(kAttr_T, d->data_type);
  nb.Attr(kAttr_strides, d->strides);
  nb.Attr(kAttr_padding, d->padding);
  nb.Attr(kAttr_data_format, d->data_format);
  nb.Attr(kAttr_dilations, d->dilations);
  nb.Attr(kAttr_activation_mode, d->activation_mode);

  // populate the device
  nb.Device(d->conv->def().device());

  // create the new fusion node
  Node* fusion_node = nullptr;
  TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));

  // populate the input control edges
  for (const Edge* e : d->conv->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->bias->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->actv->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }

  // populate output data and control edges
  for (const Edge* e : d->conv->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    }
  }
  for (const Edge* e : d->bias->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    }
  }
  for (const Edge* e : d->actv->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    } else {
      CHECK_NOTNULL(graph_->AddEdge(fusion_node, 0, e->dst(), e->dst_input()));
    }
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d->conv->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated Convolution+Bias+Activation Fusion Node: "
                   << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // add the now redundant nodes to the set of fused nodes
  fused_nodes.insert(d->conv);
  fused_nodes.insert(d->bias);
  fused_nodes.insert(d->actv);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->conv));
  graph_->RemoveNode(const_cast<Node*>(d->bias));
  graph_->RemoveNode(const_cast<Node*>(d->actv));
}

//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpBatchNormActivationInference implementation
// ----------------------------------------------

bool ROCmFusionOpBatchNormActivationInference::DoFusion(
    const Node* n2, std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n2, &d)) {
    CreateFusionOp(&d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpBatchNormActivationInference::IsFusionEligible(const Node* n2,
                                                                FusionData* d) {
  // First check whether we have the right sequence of ops
  bool is_eligible = false;

  if (isOpActivation(n2)) {  // activation node
    Node* n1 = nullptr;
    TF_CHECK_OK(n2->input_node(0, &n1));
    if (isOpBatchNorm(n1)) {  // preceded by a batchnorm node
      d->norm = n1;
      d->actv = n2;

      // check the is_training attribute to determine the type of op to create
      // (i.e. training version or inference version)
      TF_CHECK_OK(
          GetNodeAttr(d->norm->def(), kAttr_is_training, &d->is_training));

      VLOG(kVlogLevel) << "===========";
      if (d->is_training) {
        DumpNodeList(
            kVlogLevel,
            "Found Fusion Candidate BatchNorm+Activation (training-fwd): ",
            {d->norm, d->actv});
      } else {
        DumpNodeList(
            kVlogLevel,
            "Found Fusion Candidate BatchNorm+Activation (inference): ",
            {d->norm, d->actv});
      }

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({d->norm, d->actv})) {
    is_eligible = false;
  }

  // Next check if the first output of batch-norm node feeds a node other then
  // the activation node
  if (is_eligible) {
    for (const Edge* e : d->norm->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != d->actv)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "BatchNorm output feeds a node other the the activation node : "
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

    DataType T_norm, T_actv;
    TF_CHECK_OK(GetNodeAttr(d->norm->def(), kAttr_T, &T_norm));
    TF_CHECK_OK(GetNodeAttr(d->actv->def(), kAttr_T, &T_actv));

    // only float and half types are supported for now
    if ((T_norm == DT_FLOAT) || (T_norm == DT_HALF)) {
      DataType U_norm = T_norm;
      if (d->norm->type_string() == "FusedBatchNormV2") {
        TF_CHECK_OK(GetNodeAttr(d->norm->def(), kAttr_U, &U_norm));
      }

      // datatype for batch-norm and activation must match
      if ((T_norm == U_norm) && (T_norm == T_actv)) {
        d->data_type = T_norm;
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
    TF_CHECK_OK(GetNodeAttr(d->norm->def(), kAttr_data_format, &df_norm_str));

    TensorFormat df_norm;
    CHECK_EQ(FormatFromString(df_norm_str, &df_norm), true);

    if ((df_norm == FORMAT_NHWC) || (df_norm == FORMAT_NCHW)) {
      d->data_format = ToString(df_norm);
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
    TF_CHECK_OK(GetNodeAttr(d->norm->def(), kAttr_epsilon, &d->epsilon));
  }

  // finally check if the specified activation is supported
  if (is_eligible) {
    d->activation_mode = d->actv->type_string();
  }

  return is_eligible;
}

void ROCmFusionOpBatchNormActivationInference::CreateFusionOp(
    const FusionData* d, std::set<const Node*>& fused_nodes) {
  std::vector<const Edge*> norm_input_edges;
  TF_CHECK_OK(d->norm->input_edges(&norm_input_edges));

  // create an instance of the fusion node
  string op_name = strings::StrCat(d->norm->name(), d->actv->name());

  string fusion_type = d->is_training
                           ? "_ROCmFusedBatchNormActivationForward"
                           : "_ROCmFusedBatchNormActivationInference";

  NodeBuilder nb(op_name, fusion_type);

  // populate input data edges

  // batchnorm x
  nb.Input(norm_input_edges[0]->src(), norm_input_edges[0]->src_output());
  // batchnorm scale
  nb.Input(norm_input_edges[1]->src(), norm_input_edges[1]->src_output());
  // batchnorm offset
  nb.Input(norm_input_edges[2]->src(), norm_input_edges[2]->src_output());

  if (!d->is_training) {
    // batchnorm mean
    nb.Input(norm_input_edges[3]->src(), norm_input_edges[3]->src_output());
    // batchnorm variance
    nb.Input(norm_input_edges[4]->src(), norm_input_edges[4]->src_output());
  }

  // populate attributes
  nb.Attr(kAttr_T, d->data_type);
  nb.Attr(kAttr_epsilon, d->epsilon);
  nb.Attr(kAttr_data_format, d->data_format);
  nb.Attr(kAttr_activation_mode, d->activation_mode);

  // populate the device
  nb.Device(d->norm->def().device());

  // create the new fusion node
  Node* fusion_node = nullptr;
  TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));

  // populate the input control edges
  for (const Edge* e : d->norm->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->actv->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }

  // populate output data and control edges
  for (const Edge* e : d->norm->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    } else if (d->is_training) {
      if (e->src_output() == 1) {  // batch-norm mean
        CHECK_NOTNULL(
            graph_->AddEdge(fusion_node, 1, e->dst(), e->dst_input()));
      } else if (e->src_output() == 2) {  // batch-norm variance
        CHECK_NOTNULL(
            graph_->AddEdge(fusion_node, 2, e->dst(), e->dst_input()));
      } else if (e->src_output() == 3) {  // saved mean
        CHECK_NOTNULL(
            graph_->AddEdge(fusion_node, 3, e->dst(), e->dst_input()));
      } else if (e->src_output() == 4) {  // saved variance
        CHECK_NOTNULL(
            graph_->AddEdge(fusion_node, 4, e->dst(), e->dst_input()));
      }
    }
  }
  for (const Edge* e : d->actv->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    } else {
      CHECK_NOTNULL(graph_->AddEdge(fusion_node, 0, e->dst(), e->dst_input()));
    }
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d->norm->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated BatchNorm+Activation "
                   << (d->is_training ? "(training-fwd)" : "(inference)")
                   << " Fusion Node: " << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // add the now redundant nodes to the set of fused nodes
  fused_nodes.insert(d->norm);
  fused_nodes.insert(d->actv);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->norm));
  graph_->RemoveNode(const_cast<Node*>(d->actv));
}
//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpBatchNormActivationBackward implementationq
// ----------------------------------------------

bool ROCmFusionOpBatchNormActivationBackward::DoFusion(
    const Node* n2, std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n2, &d)) {
    CreateFusionOp(&d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpBatchNormActivationBackward::IsFusionEligible(const Node* n2,
                                                               FusionData* d) {
  // First check whether we have the right sequence of ops
  bool is_eligible = false;

  if (isOpBatchNormGrad(n2)) {  // batchnorm gradient node
    Node* n1 = nullptr;
    TF_CHECK_OK(n2->input_node(0, &n1));
    if (isOpActivationGrad(n1)) {  // preceded by a activation graident node
      d->norm_grad = n2;
      d->actv_grad = n1;

      // we need to cache the offset input to the batchnorm node
      // this is because the fused-op needs this input, and it is not present
      // as an input on the batchnorm grad node (apparently TF does not need it)
      const Edge* e = nullptr;

      // 4th input to batchnorm grad is the saved mean from batchnorn fwd
      TF_CHECK_OK(d->norm_grad->input_edge(3, &e));
      const Node* norm = e->src();

      // 3rd input to batchnorm fwd is the offset node we want
      TF_CHECK_OK(norm->input_node(2, &d->norm_offset));

      VLOG(kVlogLevel) << "===========";
      DumpNodeList(
          kVlogLevel,
          "Found Fusion Candidate BatchNorm+Activation (training-bwd): ",
          {d->norm_grad, d->actv_grad});
      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({d->norm_grad, d->actv_grad})) {
    is_eligible = false;
  }

  // Next check if the first output of activation gradient node feeds a node
  // other then the batch norm gradient node
  if (is_eligible) {
    for (const Edge* e : d->actv_grad->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != d->norm_grad)) {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "ActivationGrad output feeds a node other the the "
                            "BatchNormGrad node : "
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

    DataType T_norm_grad, T_actv_grad;
    TF_CHECK_OK(GetNodeAttr(d->norm_grad->def(), kAttr_T, &T_norm_grad));
    TF_CHECK_OK(GetNodeAttr(d->actv_grad->def(), kAttr_T, &T_actv_grad));

    // only float and half types are supported for now
    if ((T_norm_grad == DT_FLOAT) || (T_norm_grad == DT_HALF)) {
      DataType U_norm_grad = T_norm_grad;
      if (d->norm_grad->type_string() == "FusedBatchNormGradV2") {
        TF_CHECK_OK(GetNodeAttr(d->norm_grad->def(), kAttr_U, &U_norm_grad));
      }

      // datatype for batch-norm and activation must match
      if ((T_norm_grad == U_norm_grad) && (T_norm_grad == T_actv_grad)) {
        d->data_type = T_norm_grad;
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
        GetNodeAttr(d->norm_grad->def(), kAttr_data_format, &df_norm_grad_str));

    TensorFormat df_norm_grad;
    CHECK_EQ(FormatFromString(df_norm_grad_str, &df_norm_grad), true);

    if ((df_norm_grad == FORMAT_NHWC) || (df_norm_grad == FORMAT_NCHW)) {
      d->data_format = ToString(df_norm_grad);
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
    TF_CHECK_OK(GetNodeAttr(d->norm_grad->def(), kAttr_epsilon, &d->epsilon));
  }

  // finally check if the specified activation is supported
  if (is_eligible) {
    d->activation_mode = getActivationOpType(d->actv_grad);
  }

  return is_eligible;
}

void ROCmFusionOpBatchNormActivationBackward::CreateFusionOp(
    const FusionData* d, std::set<const Node*>& fused_nodes) {
  std::vector<const Edge*> actv_grad_input_edges;
  TF_CHECK_OK(d->actv_grad->input_edges(&actv_grad_input_edges));

  std::vector<const Edge*> norm_grad_input_edges;
  TF_CHECK_OK(d->norm_grad->input_edges(&norm_grad_input_edges));

  // create an instance of the fusion node
  string op_name = strings::StrCat(d->norm_grad->name(), d->actv_grad->name());

  NodeBuilder nb(op_name, "_ROCmFusedBatchNormActivationBackward");

  // populate input data edges

  // activation grad gradients
  nb.Input(actv_grad_input_edges[0]->src(),
           actv_grad_input_edges[0]->src_output());
  // activation grad features
  nb.Input(actv_grad_input_edges[1]->src(),
           actv_grad_input_edges[1]->src_output());

  // batchnorm grad x
  nb.Input(norm_grad_input_edges[1]->src(),
           norm_grad_input_edges[1]->src_output());
  // batchnorm grad scale
  nb.Input(norm_grad_input_edges[2]->src(),
           norm_grad_input_edges[2]->src_output());
  // batchnorm offset
  nb.Input(const_cast<Node*>(d->norm_offset), 0);
  // batchnorm grad saved_mean
  nb.Input(norm_grad_input_edges[3]->src(),
           norm_grad_input_edges[3]->src_output());
  // batchnorm grad saved_var
  nb.Input(norm_grad_input_edges[4]->src(),
           norm_grad_input_edges[4]->src_output());

  // populate attributes
  nb.Attr(kAttr_T, d->data_type);
  nb.Attr(kAttr_epsilon, d->epsilon);
  nb.Attr(kAttr_data_format, d->data_format);
  nb.Attr(kAttr_activation_mode, d->activation_mode);

  // populate the device
  nb.Device(d->norm_grad->def().device());

  // create the new fusion node
  Node* fusion_node = nullptr;
  TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));

  // populate the input control edges
  for (const Edge* e : d->norm_grad->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->actv_grad->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }

  // populate output data and control edges
  for (const Edge* e : d->actv_grad->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    }
  }
  for (const Edge* e : d->norm_grad->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    } else {
      if ((0 <= e->src_output()) && (e->src_output() <= 2)) {
        CHECK_NOTNULL(graph_->AddEdge(fusion_node, e->src_output(), e->dst(),
                                      e->dst_input()));
      } else {
        LOG(FATAL) << "Unexpected output connection on the BatchNormGrad node :"
                   << d->norm_grad
                   << ". output(idx, dest_node) :" << e->src_output() << ", "
                   << e->dst();
      }
    }
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d->norm_grad->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated BatchNorm+Activation (training-bwd)"
                   << " Fusion Node: " << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // add the now redundant nodes to the set of fused nodes
  fused_nodes.insert(d->norm_grad);
  fused_nodes.insert(d->actv_grad);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->norm_grad));
  graph_->RemoveNode(const_cast<Node*>(d->actv_grad));
}
//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpAddRelu implementation
// ----------------------------------------------

bool ROCmFusionOpAddRelu::DoFusion(const Node* n2,
                                   std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n2, &d)) {
    CreateFusionOp(&d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpAddRelu::IsFusionEligible(const Node* n2, FusionData* d) {
  bool is_eligible = false;

  // First check whether we have the right sequence of ops
  if (isOpRelu(n2)) {  // "Relu" node
    Node* n1 = nullptr;
    TF_CHECK_OK(n2->input_node(0, &n1));
    if (isOpAdd(n1)) {  // preceded by a "Add" op
      d->add = n1;
      d->relu = n2;

      DumpNodeList(kVlogLevel,
                   "Found Fusion Candidate Add+Relu: ", {d->add, d->relu});

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({d->add, d->relu})) {
    is_eligible = false;
  }

  // Next check if the first output of add node feeds a node other then
  // the activation node
  if (is_eligible) {
    for (const Edge* e : d->add->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != d->relu)) {
        VLOG(kVlogLevel) << "\tSkipping Fusion : "
                         << "Output from Add also feeds node other then Relu : "
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

    DataType T_add, T_relu;
    TF_CHECK_OK(GetNodeAttr(d->add->def(), kAttr_T, &T_add));
    TF_CHECK_OK(GetNodeAttr(d->relu->def(), kAttr_T, &T_relu));

    // only float and half types are supported for now
    if ((T_add == DT_FLOAT) || (T_add == DT_HALF)) {
      // datatype for add and relu must match
      if (T_add == T_relu) {
        d->data_type = T_add;
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

  return is_eligible;
}

void ROCmFusionOpAddRelu::CreateFusionOp(const FusionData* d,
                                         std::set<const Node*>& fused_nodes) {
  std::vector<const Edge*> add_input_edges;
  TF_CHECK_OK(d->add->input_edges(&add_input_edges));

  // create an instance of the fusion node
  string op_name = strings::StrCat(d->add->name(), d->relu->name());

  NodeBuilder nb(op_name, "_ROCmFusedAddRelu");

  // populate input data edges

  // add x
  nb.Input(add_input_edges[0]->src(), add_input_edges[0]->src_output());
  // add y
  nb.Input(add_input_edges[1]->src(), add_input_edges[1]->src_output());

  // populate attributes
  nb.Attr(kAttr_T, d->data_type);

  // populate the device
  nb.Device(d->add->def().device());

  // create the new fusion node
  Node* fusion_node = nullptr;
  TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));

  // populate the input control edges
  for (const Edge* e : d->add->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->relu->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }

  // populate output data and control edges
  for (const Edge* e : d->add->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    }
  }
  for (const Edge* e : d->relu->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    } else {
      CHECK_NOTNULL(graph_->AddEdge(fusion_node, 0, e->dst(), e->dst_input()));
    }
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d->add->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated Add+Relu Fusion Node: " << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // add the now redundant nodes to the set of fused nodes
  fused_nodes.insert(d->add);
  fused_nodes.insert(d->relu);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->add));
  graph_->RemoveNode(const_cast<Node*>(d->relu));
}

//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpAddNReluGrad implementation
// ----------------------------------------------

bool ROCmFusionOpAddNReluGrad::DoFusion(const Node* n2,
                                        std::set<const Node*>& fused_nodes) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n2, &d)) {
    CreateFusionOp(&d, fused_nodes);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpAddNReluGrad::IsFusionEligible(const Node* n2, FusionData* d) {
  bool is_eligible = false;

  // First check whether we have the right sequence of ops
  if (isOpReluGrad(n2)) {  // "ReluGrad" node

    Node* n1_1 = nullptr;
    TF_CHECK_OK(n2->input_node(0, &n1_1));

    Node* n1_2 = nullptr;
    TF_CHECK_OK(n2->input_node(1, &n1_2));

    // VLOG(kVlogLevel) << "===========";
    // DumpNodeList(kVlogLevel, "ReluGrad Inputs: ",
    // 		 {n1_1, n1_2});
    // VLOG(kVlogLevel) << "===========";

    if (isOpAddN(n1_1) && isOpReluOrFusedAddRelu(n1_2)) {
      // preceded by "AddN" and "Relu"/"_ROCmFusedAddRelu" Ops

      d->addN = n1_1;
      d->relu = n1_2;
      d->reluGrad = n2;

      DumpNodeList(kVlogLevel, "Found Fusion Candidate AddN+ReluGrad : ",
                   {d->addN, d->reluGrad});

      is_eligible = true;
    }
  }

  // ensure all the nodes are placed on the same GPU
  if (is_eligible && !areAssignedToSameGpu({d->addN, d->relu, d->reluGrad})) {
    is_eligible = false;
  }

  // Next check if the first output of addN node feeds a node other then
  // the activation node
  if (is_eligible) {
    for (const Edge* e : d->addN->out_edges()) {
      if ((e->src_output() == 0) && (e->dst() != d->reluGrad)) {
        VLOG(kVlogLevel)
            << "\tSkipping Fusion : "
            << "Output from AddN also feeds node other then ReluGrad : "
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

    DataType T_addN, T_relu, T_reluGrad;
    TF_CHECK_OK(GetNodeAttr(d->addN->def(), kAttr_T, &T_addN));
    TF_CHECK_OK(GetNodeAttr(d->relu->def(), kAttr_T, &T_relu));
    TF_CHECK_OK(GetNodeAttr(d->reluGrad->def(), kAttr_T, &T_reluGrad));

    // only float and half types are supported for now
    if ((T_addN == DT_FLOAT) || (T_addN == DT_HALF)) {
      // datatype for binary-op and activation must match
      if ((T_addN == T_relu) && (T_addN == T_reluGrad)) {
        d->data_type = T_addN;
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
    TF_CHECK_OK(GetNodeAttr(d->addN->def(), kAttr_N, &d->num_addN_inputs));
    if (d->num_addN_inputs != 2) {
      VLOG(kVlogLevel) << "\tSkipping Fusion : "
                       << " AddN node has unsupported N value : "
                       << d->num_addN_inputs;
      VLOG(kVlogLevel) << "===========";
    }
  }

  return is_eligible;
}

void ROCmFusionOpAddNReluGrad::CreateFusionOp(
    const FusionData* d, std::set<const Node*>& fused_nodes) {
  std::vector<const Edge*> addN_input_edges;
  TF_CHECK_OK(d->addN->input_edges(&addN_input_edges));

  std::vector<const Edge*> reluGrad_input_edges;
  TF_CHECK_OK(d->reluGrad->input_edges(&reluGrad_input_edges));

  // create an instance of the fusion node
  string op_name = strings::StrCat(d->addN->name(), d->reluGrad->name());

  NodeBuilder nb(op_name, "_ROCmFusedAddNReluGrad");

  // populate input data edges

  // add addN inputs
  std::vector<NodeBuilder::NodeOut> addN_inputs;
  for (const Edge* e : addN_input_edges) {
    addN_inputs.push_back(NodeBuilder::NodeOut(e->src(), e->src_output()));
  }
  nb.Input(addN_inputs);

  // add the relu output
  CHECK_EQ(reluGrad_input_edges[1]->src(), d->relu);
  nb.Input(reluGrad_input_edges[1]->src(),
           reluGrad_input_edges[1]->src_output());

  // populate attributes
  nb.Attr(kAttr_T, d->data_type);
  nb.Attr(kAttr_N, d->num_addN_inputs);

  // populate the device
  nb.Device(d->addN->def().device());

  // create the new fusion node
  Node* fusion_node = nullptr;
  TF_CHECK_OK(nb.Finalize(graph_, &fusion_node));

  // populate the input control edges
  for (const Edge* e : d->addN->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->reluGrad->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }

  // populate output data and control edges
  for (const Edge* e : d->addN->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    }
  }
  for (const Edge* e : d->reluGrad->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(fusion_node, e->dst(), true));
    } else {
      CHECK_NOTNULL(graph_->AddEdge(fusion_node, 0, e->dst(), e->dst_input()));
    }
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d->addN->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated AddN+ReluGrad Fusion Node: " << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // add the now redundant nodes to the set of fused nodes
  fused_nodes.insert(d->addN);
  fused_nodes.insert(d->reluGrad);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->addN));
  graph_->RemoveNode(const_cast<Node*>(d->reluGrad));
}

//----------------------------------------------------------------------

}  // namespace gpu_fusion_pass
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
