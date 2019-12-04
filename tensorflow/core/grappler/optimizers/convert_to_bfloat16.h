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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONVERT_TO_BFLOAT16_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONVERT_TO_BFLOAT16_H_

#if defined(INTEL_MKL) && defined(ENABLE_INTEL_MKL_BFLOAT16)
#include "tensorflow/core/graph/mkl_graph_util.h"
#endif

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include <fstream>

namespace tensorflow {
namespace grappler {

#if defined(INTEL_MKL) && defined(ENABLE_INTEL_MKL_BFLOAT16)

// Convert data types to bfloat16 where appropriate to improve performance on

class BFloat16Converter : public GraphOptimizer {
 public:
  explicit BFloat16Converter(
      RewriterConfig::Toggle opt_level = RewriterConfig::OFF) {}
  ~BFloat16Converter() override {}

  string name() const override { return "bfloat16_converter"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override {}

 private:
  bool CanOpRunOnCPUDevice(const Node* n) const {
    bool result = true;
    string reason;

    // Substring that should be checked for in device name for CPU device.
    const char* const kCPUDeviceSubStr = "CPU";

    // If Op has been specifically assigned to a non-CPU device, then No.
    if (!n->assigned_device_name().empty() &&
        !absl::StrContains(n->assigned_device_name(), kCPUDeviceSubStr)) {
      result = false;
      reason = "Op has been assigned a runtime device that is not CPU.";
    }

    // If user has specifically assigned this op to a non-CPU device, then No.
    if (!n->def().device().empty() &&
        !absl::StrContains(n->def().device(), kCPUDeviceSubStr)) {
      result = false;
      reason = "User has assigned a device that is not CPU.";
    }

    if (!result) {
      VLOG(1) << name() << ": Skipping rewriting of the node "
              << n->type_string() << ", reason: " << reason;
    }

    // Otherwise Yes.
    return result;
  }

  inline void ChangeDataType(Node* n, DataType t) const {
    n->ClearAttr("T");
    n->AddAttr("T", t);
  }

  // We want to keep constants, variables, and optimizers in FP32 mode.
  inline bool ShouldSkipOp(const Node* n) const {
    return n->type_string() == "Const" || n->type_string() == "Variable" ||
           n->type_string() == "Softmax" ||
           n->type_string() == "SoftmaxCrossEntropyWithLogits" ||
           n->type_string() == "L2Loss" || n->type_string() == "AddN";
  }

  // --------------------------------------------------------------------------
  // Rules which allow conditional conversion of ops to BFLOAT16 type.
  // TODO(nhasabni): We can reuse rewrite rules from MklLayoutPass here.
  inline bool AlwaysRewriteOp(const Node* n) const { return true; }
  inline bool RewriteIfAllInputsInBFloat16Op(const Node* n) const {
    return n->type_string() == "Add" || n->type_string() == "AddV2" ||
           n->type_string() == "Identity" || n->type_string() == "Maximum" ||
           n->type_string() == "Mul" || n->type_string() == "Sub" ||
           n->type_string() == "SquaredDifference";
  }
  inline bool AreAllInputsInBFloat16(const Node* n) const {
    for (auto e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      if (!CanOpSupportBFloat16(e->src())) return false;
    }
    return true;
  }
  inline bool FusedConv2DRewrite(const Node* n) const {
    // MKL DNN currently doesn't support all fusions that grappler fuses
    // together with Conv2D (ex. batchnorm). We rewrite _FusedConv2D only if
    // it includes those we support.
    DataType T;
    if (!TryGetNodeAttr(n->def(), "T", &T) ||
        !mkl_op_registry::IsMklLayoutDependentOp("_MklFusedConv2D", T)) {
      return false;
    }
    std::vector<string> fused_ops;
    TF_CHECK_OK(GetNodeAttr(n->def(), "fused_ops", &fused_ops));
    return (fused_ops == std::vector<string>{"BiasAdd"} ||
            fused_ops == std::vector<string>{"Relu"} ||
            fused_ops == std::vector<string>{"Relu6"} ||
            fused_ops == std::vector<string>{"Elu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Elu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"});
  }
  inline bool FusedMatMulRewrite(const Node* n) const {
    bool trans_a;
    std::vector<string> fused_ops;

    // Do not rewrite with transpose attribute because reorder has performance
    // impact.
    TF_CHECK_OK(GetNodeAttr(n->def(), "transpose_a", &trans_a));
    // Do not rewrite with more than 1 post op because MKL-DNN doesn't support.
    TF_CHECK_OK(GetNodeAttr(n->def(), "fused_ops", &fused_ops));

    return (!trans_a) && fused_ops == std::vector<string>{"BiasAdd"};
  }
  // --------------------------------------------------------------------------

  inline bool CanOpSupportBFloat16(const Node* n) const {
    string mkl_op_name = mkl_op_registry::GetMklOpName(n->type_string());
    // For some ops, we do not follow standard rule of just
    // prefixing _Mkl to the type_string to obtain corresponding Mkl
    // type_string. For them, we need special case.
    if (n->type_string() == "_FusedConv2D" && FusedConv2DRewrite(n))
      mkl_op_name = "_MklFusedConv2D";
    else if (n->type_string() == "__MklDummyPadWithConv2D")
      mkl_op_name = "_MklPadWithConv2D";
    else if (n->type_string() == "__MklDummyPadWithFusedConv2D")
      mkl_op_name = "_MklPadWithFusedConv2D";
    else if (n->type_string() == "__MklDummyConv2DWithBias")
      mkl_op_name = "_MklConv2DWithBias";
    else if (n->type_string() == "__MklDummyConv2DBackpropFilterWithBias")
      mkl_op_name = "_MklConv2DBackpropFilterWithBias";
    else if (n->type_string() == "_FusedMatMul" && FusedMatMulRewrite(n))
      mkl_op_name = "_MklFusedMatMul";
    return mkl_op_registry::IsMklOp(mkl_op_name, DT_BFLOAT16);
  }

  // Insert a node that casts tensor from 'src_dtype' to 'dst_dtype'
  // on edge 'e' in graph 'g'. Returns Status::OK() if insertion is
  // fine, otherwise returns error.
  Status InsertCastNode(Graph* g, const Edge* e, DataType src_dtype,
                        DataType dst_dtype);

  // Convert input graph 'g' in-place by replacing FP32 nodes that can
  // operate in BFLOAT16 type. The pass will automatically insert
  // appropriate Cast nodes to convert tensors between FP32 and BFLOAT16 types.
  Status ConvertToBFloat16(Graph* g);
};

#else  // INTEL_MKL && ENABLE_INTEL_MKL_BFLOAT16

class BFloat16Converter : public GraphOptimizer {
 public:
  explicit BFloat16Converter(
      RewriterConfig::Toggle opt_level = RewriterConfig::OFF) {}
  ~BFloat16Converter() override {}
  string name() const override { return "bfloat16_converter"; };
  bool UsesFunctionLibrary() const override { return false; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override {
    VLOG(WARNING) << "BFloat16Converter is currently supported only for "
                  << "Intel MKL backend. Skipping it for other backends.";
    return errors::Aborted("Nothing to do.");
  };
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override {}
};

#endif  // ENABLE_INTEL_MKL_BFLOAT16

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONVERT_TO_BFLOAT16_H_
