/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifdef AMD_ZENDNN

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/core/util/zen_util.h"

namespace tensorflow {

class ZenEagerOpRewrite : public EagerOpRewrite {
 public:
  ZenEagerOpRewrite(string name, string file, string line);
  struct ZenEagerOp {
    string op_name;
    string zen_op_name;
    std::function<bool(EagerOperation *)> RewriteRule;
    std::function<Status(EagerOperation *, std::unique_ptr<EagerOperation> *,
                         string)>
        CreateZenOp;
  };

 private:
  std::unordered_map<std::string, ZenEagerOp> zen_eager_ops_;

  // The entry point to execute the op rewrite.
  Status Run(EagerOperation *orig_op,
             std::unique_ptr<tensorflow::EagerOperation> *out_op);

  // Initializes the new op and sets up its inputs and attributes
  static Status SetupNewOp(EagerOperation *orig_op, const string zen_op_name,
                           std::unique_ptr<EagerOperation> *new_zen_op);

  // Generic rewrite that can be used for any zen op that doesn't need
  // special processing.
  static Status CreateGenericZenOp(EagerOperation *orig_op,
                                   std::unique_ptr<EagerOperation> *zen_op,
                                   string zen_op_name);

  // Rewrite rule for Conv2D, Conv2DBackpropInput and Conv2DBackpropFilter.
  static bool RewriteConv2D(EagerOperation *op);

  // Rewrite rule for FusedBatchNormV3 and FusedBatchNormGradV3
  static bool RewriteFusedBatchNormV3(EagerOperation *op);

  // Calls op-specific rewrite function to create new ZEN op.
  Status RewriteToZenOp(EagerOperation *orig_op,
                        std::unique_ptr<EagerOperation> *zen_op);

  // Check whether we can rewrite the op to ZEN one or not.
  bool ShouldRewriteOp(EagerOperation *op);

  // Default rewrite rule to be used when rewrite should happen without any
  // restriction.
  static bool AlwaysRewrite(EagerOperation *op) {
    DataType nT;
    const NodeDef &ndef = op->MutableAttrs()->BuildNodeDef();
    TF_CHECK_OK(GetNodeAttr(ndef, "T", &nT));
    return (nT == DT_FLOAT);
  }

  // Helper function to insert zen_eager_ops to Map
  void InsertZENEagerOps(ZenEagerOp op);
};

REGISTER_REWRITE(EagerOpRewriteRegistry::POST_PLACEMENT, 30000,
                 ZenEagerOpRewrite);

// Constructor
ZenEagerOpRewrite::ZenEagerOpRewrite(string name, string file, string line)
    : EagerOpRewrite(name, file, line) {
  InsertZENEagerOps(
      {"AvgPool", "ZenAvgPool", AlwaysRewrite, CreateGenericZenOp});
  InsertZENEagerOps(
      {"MaxPool", "ZenMaxPool", AlwaysRewrite, CreateGenericZenOp});
  InsertZENEagerOps({"Conv2D", "ZenConv2D", AlwaysRewrite, CreateGenericZenOp});
  InsertZENEagerOps({"MatMul", "ZenMatMul", AlwaysRewrite, CreateGenericZenOp});
  InsertZENEagerOps(
      {"Softmax", "ZenSoftmax", AlwaysRewrite, CreateGenericZenOp});
  InsertZENEagerOps({"FusedBatchNorm", "ZenFusedBatchNorm", AlwaysRewrite,
                     CreateGenericZenOp});
  InsertZENEagerOps({"FusedBatchNormV2", "ZenFusedBatchNormV2", AlwaysRewrite,
                     CreateGenericZenOp});
  InsertZENEagerOps({"FusedBatchNormV3", "ZenFusedBatchNormV3", AlwaysRewrite,
                     CreateGenericZenOp});
};

void ZenEagerOpRewrite::InsertZENEagerOps(ZenEagerOp op) {
  zen_eager_ops_.insert(std::make_pair(op.op_name, op));
}

Status ZenEagerOpRewrite::Run(
    EagerOperation *orig_op,
    std::unique_ptr<tensorflow::EagerOperation> *out_op) {
  if (ShouldRewriteOp(orig_op)) {
    TF_CHECK_OK(RewriteToZenOp(orig_op, out_op));
  }
  return Status::OK();
}

Status ZenEagerOpRewrite::SetupNewOp(
    EagerOperation *orig_op, string zen_op_name,
    std::unique_ptr<EagerOperation> *new_zen_op) {
  bool is_remote = false;
  new_zen_op->reset(new tensorflow::EagerOperation(&orig_op->EagerContext()));
  TF_RETURN_IF_ERROR(new_zen_op->get()->Reset(zen_op_name.c_str(), nullptr,
                                              is_remote, nullptr));

  int num_inputs = orig_op->Inputs().size();
  // Add all inputs to the new op.
  for (int i = 0; i < num_inputs; ++i) {
    TF_RETURN_IF_ERROR((*new_zen_op)->AddInput(orig_op->Inputs()[i]));
  }

  // Copy all attributes to the new op.
  const NodeDef &orig_ndef = orig_op->MutableAttrs()->BuildNodeDef();

  AttrSlice attr_list(orig_ndef);
  for (const auto &attr : attr_list) {
    (*new_zen_op)->MutableAttrs()->Set(attr.first, attr.second);
  }

  // Setting default value for Below Attributes which
  //  are required for BLOCKED FORMAT. However it is
  //  currently disabled.
  (*new_zen_op)->MutableAttrs()->Set("reorder_before", false);
  (*new_zen_op)->MutableAttrs()->Set("reorder_after", false);

  // Curently Blocked Format is not supported with Eager Mode.
  if (zendnn_getenv_int("ZENDNN_BLOCKED_FORMAT", 0) == 1) {
    setenv("ZENDNN_BLOCKED_FORMAT", "0", 1);
  }

  // Setting default value for Below Attributes which
  //  are required for MemPool feature. However it is
  //  currently disabled.
  (*new_zen_op)->MutableAttrs()->Set("in_links", 1);
  (*new_zen_op)->MutableAttrs()->Set("out_links", 1);
  (*new_zen_op)->MutableAttrs()->Set("reset", true);

  // Curently MemPool is not supported with Eager Mode.
  if (zendnn_getenv_int("ZENDNN_ENABLE_MEMPOOL", 0) == 1) {
    setenv("ZENDNN_ENABLE_MEMPOOL", "0", 1);
  }

  string device_name = orig_op->DeviceName();
  return (*new_zen_op)->SetDeviceName(device_name.c_str());
}

Status ZenEagerOpRewrite::CreateGenericZenOp(
    EagerOperation *orig_op, std::unique_ptr<EagerOperation> *zen_op,
    string zen_op_name) {
  zendnnInfo(ZENDNN_FWKLOG,
             " TF-EAGER-REWRITE Info: OriginalOp=", orig_op->Name(),
             " ZenOp=", zen_op_name);

  TF_CHECK_OK(SetupNewOp(orig_op, zen_op_name, zen_op));
  return Status::OK();
}

bool ZenEagerOpRewrite::ShouldRewriteOp(EagerOperation *op) {
  // Don't rewrite the op if ZEN use is disabled at runtime.
  if (IsZenDNNDisabled()) {
    return false;
  }

  // Find and call the op's rewrite rule that determines whether we need to
  // rewrite this op or not.
  auto it = zen_eager_ops_.find(op->Name());
  if (it != zen_eager_ops_.end()) {
    // Eager op found so verify Rewrite
    if (it->second.RewriteRule(op)) {
      return true;
    }
  }
  return false;
}

Status ZenEagerOpRewrite::RewriteToZenOp(
    EagerOperation *orig_op, std::unique_ptr<EagerOperation> *zen_op) {
  // TODO(zendnn-tf): zen_eager_ops_ lookup can be reduced from twice
  // (once each in ShouldRewriteOp & RewriteToZenOp) to just once.
  TF_RETURN_IF_ERROR(zen_eager_ops_[orig_op->Name()].CreateZenOp(
      orig_op, zen_op, zen_eager_ops_[orig_op->Name()].zen_op_name));
  return Status::OK();
}

bool ZenEagerOpRewrite::RewriteConv2D(EagerOperation *op) {
  const NodeDef &ndef = op->MutableAttrs()->BuildNodeDef();
  string padding;
  TF_CHECK_OK(GetNodeAttr(ndef, "padding", &padding));
  // Right now ZEN Conv2D does not support explicit padding.
  return (padding != "EXPLICIT");
}

// Check if the data_format attribute in the node def represents 5D tensor
bool inline Check5DFormat(const NodeDef &ndef) {
  string data_format;
  TF_CHECK_OK(GetNodeAttr(ndef, "data_format", &data_format));
  if (data_format.compare("NCDHW") == 0 || data_format.compare("NDHWC") == 0) {
    return true;
  }
  return false;
}

bool ZenEagerOpRewrite::RewriteFusedBatchNormV3(EagerOperation *op) {
  const NodeDef &ndef = op->MutableAttrs()->BuildNodeDef();
  if (Check5DFormat(ndef)) {
    VLOG(1) << "Eager Op Rewrite: FusedBatchNorm(Grad)V3 op currently does not "
            << "support 5D tensors.";
    return false;
  }
  return true;
}

}  // namespace tensorflow

#endif  // AMD_ZENDNN
