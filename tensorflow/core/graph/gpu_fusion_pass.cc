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
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/graph/gpu_fusion_pass.h"

namespace tensorflow {
namespace gpu_fusion_pass {

//----------------------------------------------------------------------

// declaration of all the constants that are used in the gpu_fusion_pass
// namespace
const int kVlogLevel = -1;

// graph pass grouping for this fusion pass
const OptimizationPassRegistry::Grouping kROCmFusionPassGrouping =
    (getenv("TF_ROCM_FUSION_PASS_POST_PARTITIONING") == nullptr)
        ? OptimizationPassRegistry::POST_PLACEMENT
        : OptimizationPassRegistry::POST_PARTITIONING;

// attribute name strings
const char* kAttr_T = "T";
const char* kAttr_U = "U";
const char* kAttr_activation_mode = "activation_mode";
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

// is this node an instance of a batchnorm op for which we support fusion for?
inline bool isOpBatchNorm(const Node* n) {
  return ((n->type_string() == "FusedBatchNorm") ||
          (n->type_string() == "FusedBatchNormV2"));
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
      std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions, Graph* g,
      std::set<const Node*>& fused_nodes);

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
  ROCmFusionOpBase(Graph* g, std::set<const Node*>& fn)
      : graph_(g), fused_nodes_(fn) {}

  virtual ~ROCmFusionOpBase() {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  virtual bool DoFusion(const Node* n) = 0;

 protected:
  Graph* graph_;
  std::set<const Node*>& fused_nodes_;
};

//----------------------------------------------------------------------

// Convolution-Bias-BatchNorm-Activation Fusion
class ROCmFusionOpConvolutionBiasBatchNormActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpConvolutionBiasBatchNormActivation(Graph* g,
                                                 std::set<const Node*>& fn)
      : ROCmFusionOpBase(g, fn) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n) override;

 protected:
  struct FusionData {
    const Node* conv;
    const Node* bias;
    const Node* norm;
    const Node* actv;
  };

  bool IsFusionEligible(const Node* n, FusionData* d);

  void CreateFusionOp(const FusionData* d);
};

//----------------------------------------------------------------------

// Convolution-Bias-Activation Fusion
class ROCmFusionOpConvolutionBiasActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpConvolutionBiasActivation(Graph* g, std::set<const Node*>& fn)
      : ROCmFusionOpBase(g, fn) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n) override;

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

  void CreateFusionOp(const FusionData* d);
};

//----------------------------------------------------------------------

// BatchNorm-Activation Fusion
class ROCmFusionOpBatchNormActivation : public ROCmFusionOpBase {
 public:
  ROCmFusionOpBatchNormActivation(Graph* g, std::set<const Node*>& fn)
      : ROCmFusionOpBase(g, fn) {}

  // routine to (maybe) do fusion on the given node (+ the nodes preceding
  // it). will return true if fusion was done, false otherwise
  bool DoFusion(const Node* n) override;

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

  void CreateFusionOp(const FusionData* d);
};

//----------------------------------------------------------------------

Status ROCmFusionPass::Run(const GraphOptimizationPassOptions& options) {
  // enable the fusion pass if the env var TF_ROCM_FUSION_ENABLE is set
  const char* enable_fusion = getenv("TF_ROCM_FUSION_ENABLE");
  if (enable_fusion != nullptr) {
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
  if (getenv("TF_ROCM_FUSION_DUMP_GRAPH_BEFORE")) {
    DumpGraph(kVlogLevel, "Before running ROCmFusionPass", &*graph);
  }

  std::vector<std::unique_ptr<ROCmFusionOpBase> > fusions;
  std::set<const Node*> fused_nodes;

  // Initialize a vector of all the fusion operations we currently support
  InitializeFusions(fusions, graph, fused_nodes);

  std::vector<Node*> order;
  GetPostOrder(*graph, &order);  // This will give us reverse topological sort.

  for (const Node* n : order) {

    if (fused_nodes.count(n)) {
      // We have fused already this node...skip it
      // Note that we are traversing nodes in reverse topological order, and
      // matches are found by comparing node sequences from back to front, so
      // we will hit nodes that we have already fused into a fusion operation.
      continue;
    }

    for (auto& fusion : fusions) {
      if (fusion->DoFusion(n)) {
        // bail out after the first successful fusion ...
        // cannot do more than one fusions on a op
        break;
      }
    }
  }

  if (getenv("TF_ROCM_FUSION_DUMP_GRAPH_AFTER")) {
    DumpGraph(kVlogLevel, "After running ROCmFusionPass", &*graph);
  }

  return true;
}

void ROCmFusionPass::InitializeFusions(
    std::vector<std::unique_ptr<ROCmFusionOpBase> >& fusions, Graph* g,
    std::set<const Node*>& fused_nodes) {
  fusions.emplace_back(
      new ROCmFusionOpConvolutionBiasBatchNormActivation(g, fused_nodes));
  fusions.emplace_back(
      new ROCmFusionOpConvolutionBiasActivation(g, fused_nodes));
  fusions.emplace_back(new ROCmFusionOpBatchNormActivation(g, fused_nodes));
}

//----------------------------------------------------------------------

// -------------------------------------------------------------
// ROCmFusionOpConvolutionBiasBatchNormActivation implementation
// -------------------------------------------------------------
bool ROCmFusionOpConvolutionBiasBatchNormActivation::DoFusion(const Node* n4) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n4, &d)) {
    CreateFusionOp(&d);
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
    const FusionData* d) {
  // todo
}
//----------------------------------------------------------------------

// ----------------------------------------------------
// ROCmFusionOpConvolutionBiasActivation implementation
// ----------------------------------------------------
bool ROCmFusionOpConvolutionBiasActivation::DoFusion(const Node* n3) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n3, &d)) {
    CreateFusionOp(&d);
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
    const FusionData* d) {
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
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->bias->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    }
  }
  for (const Edge* e : d->actv->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
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
  fused_nodes_.insert(d->conv);
  fused_nodes_.insert(d->bias);
  fused_nodes_.insert(d->actv);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->conv));
  graph_->RemoveNode(const_cast<Node*>(d->bias));
  graph_->RemoveNode(const_cast<Node*>(d->actv));
}

//----------------------------------------------------------------------

// ----------------------------------------------
// ROCmFusionOpBatchNormActivation implementation
// ----------------------------------------------

bool ROCmFusionOpBatchNormActivation::DoFusion(const Node* n2) {
  bool did_fusion = false;
  FusionData d;
  if (IsFusionEligible(n2, &d)) {
    CreateFusionOp(&d);
    did_fusion = true;
  }
  return did_fusion;
}

bool ROCmFusionOpBatchNormActivation::IsFusionEligible(const Node* n2,
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
        DumpNodeList(kVlogLevel,
                     "Found Fusion Candidate BatchNorm+Activation (training): ",
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

  // we currently do not support fusion for training

  if (is_eligible && d->is_training) {
    VLOG(kVlogLevel)
        << "\tSkipping Fusion : "
        << "is_training is set to true, fusion only supported for inference";
    VLOG(kVlogLevel) << "===========";
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

    // only float type is supported for now
    if (T_norm == DT_FLOAT) {
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

void ROCmFusionOpBatchNormActivation::CreateFusionOp(const FusionData* d) {
  std::vector<const Edge*> norm_input_edges;
  TF_CHECK_OK(d->norm->input_edges(&norm_input_edges));

  // create an instance of the fusion node
  string op_name = strings::StrCat(d->norm->name(), d->actv->name());

  string fusion_type = d->is_training
                           ? "_ROCmFusedBatchNormActivationTraining"
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
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    } else if (d->is_training) {
      if (e->src_output() == 1) {  // batch-norm mean
        CHECK_NOTNULL(
            graph_->AddEdge(fusion_node, 1, e->dst(), e->dst_input()));
      } else if (e->src_output() == 1) {  // batch-norm variance
        CHECK_NOTNULL(
            graph_->AddEdge(fusion_node, 2, e->dst(), e->dst_input()));
      }
    }
  }
  for (const Edge* e : d->actv->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL(graph_->AddControlEdge(e->src(), fusion_node, true));
    } else {
      CHECK_NOTNULL(graph_->AddEdge(fusion_node, 0, e->dst(), e->dst_input()));
    }
  }

  // populate the device placement
  fusion_node->set_assigned_device_name(d->norm->assigned_device_name());

  VLOG(kVlogLevel) << "\tCreated BatchNorm+Activation "
                   << (d->is_training ? "(training)" : "(inference)")
                   << " Fusion Node: " << fusion_node;
  VLOG(kVlogLevel) << "===========";

  // add the now redundant nodes to the set of fused nodes
  fused_nodes_.insert(d->norm);
  fused_nodes_.insert(d->actv);

  // and remove them from the graph
  graph_->RemoveNode(const_cast<Node*>(d->norm));
  graph_->RemoveNode(const_cast<Node*>(d->actv));
}
//----------------------------------------------------------------------

}  // namespace gpu_fusion_pass
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
