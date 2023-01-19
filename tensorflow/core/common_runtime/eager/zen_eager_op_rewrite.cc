/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/graph/zen_graph_util.h"
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
    std::function<bool(EagerOperation *)> rewrite_rule;
    std::function<Status(EagerOperation *, std::unique_ptr<EagerOperation> *,
                         string)>
        create_zen_op;
  };

 private:
  // Maintain record of Zen op to rewrite.
  std::unordered_map<std::string, ZenEagerOp> zen_eager_ops_;

  // The entry point to execute the op rewrite.
  Status Run(EagerOperation *orig_op,
             std::unique_ptr<tensorflow::EagerOperation> *out_op);

  // Initializes the new op and sets up its inputs and attributes
  static Status SetupNewOp(EagerOperation *orig_op, const string zen_op_name,
                           std::unique_ptr<EagerOperation> *new_zen_op);

  // Generic rewrite that can be used for any Zen op that doesn't need
  // special processing.
  static Status CreateGenericZenOp(EagerOperation *orig_op,
                                   std::unique_ptr<EagerOperation> *zen_op,
                                   string zen_op_name);

  // Calls op-specific rewrite function to create new Zen op.
  Status RewriteToZenOp(EagerOperation *orig_op,
                        std::unique_ptr<EagerOperation> *zen_op);

  // Check whether we can rewrite the op to Zen one or not.
  bool ShouldRewriteOp(EagerOperation *op);

  // Default rewrite rule that always rewrite for float data type.
  static bool AlwaysRewriteFloat(EagerOperation *op) {
    DataType data_type;
    const NodeDef &ndef = op->MutableAttrs()->BuildNodeDef();
    TF_CHECK_OK(GetNodeAttr(ndef, "T", &data_type));
    return (data_type == DT_FLOAT);
  }

  // Helper function to insert zen_eager_ops to Map
  void InsertZenEagerOps(ZenEagerOp op);

  // List of eager ops that can be rewritten with Zen ops.
  // The list is sorted in alphabetical order
  const std::vector<string> kAlwaysRewriteOps = {
      "AvgPool",          "Conv2D", "FusedBatchNorm", "FusedBatchNormV2",
      "FusedBatchNormV3", "MatMul", "MaxPool",        "Softmax"};
};

// The priority value must be higher than MklEagerOpRewrite (10000) so that
// Zen rewrite happens before Mkl rewrite.
REGISTER_REWRITE(EagerOpRewriteRegistry::POST_PLACEMENT, 30000,
                 ZenEagerOpRewrite);

// Constructor
ZenEagerOpRewrite::ZenEagerOpRewrite(string name, string file, string line)
    : EagerOpRewrite(name, file, line) {
  for (const auto &op_name : kAlwaysRewriteOps) {
    InsertZenEagerOps({op_name, zen_op_registry::GetZenOpName(op_name),
                       AlwaysRewriteFloat, CreateGenericZenOp});
  }
};

void ZenEagerOpRewrite::InsertZenEagerOps(ZenEagerOp op) {
  zen_eager_ops_.insert(std::make_pair(op.op_name, op));
}

Status ZenEagerOpRewrite::Run(
    EagerOperation *orig_op,
    std::unique_ptr<tensorflow::EagerOperation> *out_op) {
  if (ShouldRewriteOp(orig_op)) {
    TF_CHECK_OK(RewriteToZenOp(orig_op, out_op));
  }
  return OkStatus();
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
  const NodeDef &kOrigNodeDef = orig_op->MutableAttrs()->BuildNodeDef();

  AttrSlice attr_list(kOrigNodeDef);
  for (const auto &attr : attr_list) {
    (*new_zen_op)->MutableAttrs()->Set(attr.first, attr.second);
  }

  (*new_zen_op)->MutableAttrs()->Set("is_eager", true);
  (*new_zen_op)->MutableAttrs()->Set("reorder_before", false);
  (*new_zen_op)->MutableAttrs()->Set("reorder_after", false);
  (*new_zen_op)->MutableAttrs()->Set("in_links", 1);
  (*new_zen_op)->MutableAttrs()->Set("out_links", 1);
  (*new_zen_op)->MutableAttrs()->Set("reset", true);

  string device_name = orig_op->DeviceName();
  return (*new_zen_op)->SetDeviceName(device_name.c_str());
}

Status ZenEagerOpRewrite::CreateGenericZenOp(
    EagerOperation *orig_op, std::unique_ptr<EagerOperation> *zen_op,
    string zen_op_name) {
  VLOG(1) << " TF-EAGER-REWRITE Info: OriginalOp= " << orig_op->Name()
          << " ZenOp=" << zen_op_name;

  TF_CHECK_OK(SetupNewOp(orig_op, zen_op_name, zen_op));
  return OkStatus();
}

bool ZenEagerOpRewrite::ShouldRewriteOp(EagerOperation *op) {
  // Don't rewrite the op if TF-ZenDNN use is disabled at runtime.
  if (!IsZenDnnEnabled()) {
    return false;
  }
  // Only rewrite if op is to be run on CPU device.
  if (op->GetDeviceParsedName().type != "CPU") {
    return false;
  }
  DataType data_type;
  if (op->Attrs().Get("T", &data_type) != OkStatus()) {
    return false;
  }
  // Find the op and verify the requirements for rewriting it with Zen op.
  auto it = zen_eager_ops_.find(op->Name());
  if (it != zen_eager_ops_.end()) {
    // Eager op found
    // Verify that a kernel exists for Zen op and rewrite is possible
    if (zen_op_registry::IsZenOpKernelRegistered(
            zen_op_registry::GetZenOpName(op->Name()), data_type) &&
        it->second.rewrite_rule(op)) {
      return true;
    }
  }
  return false;
}

Status ZenEagerOpRewrite::RewriteToZenOp(
    EagerOperation *orig_op, std::unique_ptr<EagerOperation> *zen_op) {
  // TODO(zendnn-tf): zen_eager_ops_ lookup can be reduced from twice
  // (once each in ShouldRewriteOp & RewriteToZenOp) to just once.
  TF_RETURN_IF_ERROR(zen_eager_ops_[orig_op->Name()].create_zen_op(
      orig_op, zen_op, zen_eager_ops_[orig_op->Name()].zen_op_name));
  return OkStatus();
}

}  // namespace tensorflow

#endif  // AMD_ZENDNN
