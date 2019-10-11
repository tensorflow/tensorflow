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
#ifdef INTEL_MKL
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/graph/mkl_layout_pass.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

class MklEagerOpRewrite : public EagerOpRewrite {
 public:
  MklEagerOpRewrite(string name, string file, string line);
  typedef struct {
    string op_name;
    std::function<bool(EagerOperation*)> RewriteRule;
    std::function<Status(EagerOperation*, std::unique_ptr<EagerOperation>*)>
        CreateMklOp;
  } MklEagerOp;

 private:
  // TODO(intel-tf): refactor with unordered_map;
  // especially when adding more ops/rewrite rules in future.
  std::vector<MklEagerOp> mkl_eager_ops_;

  // The entry point to execute the op rewrite.
  Status Run(EagerOperation* orig_op,
             std::unique_ptr<tensorflow::EagerOperation>* out_op);

  // Initializes the new op and sets up its inputs and attributes
  static Status SetupNewOp(EagerOperation* orig_op, const string mkl_op_name,
                           std::unique_ptr<EagerOperation>* new_mkl_op);

  // Generic rewrite that can be used for any mkl op that doesn't need
  // special processing.
  static Status CreateGenericMklOp(EagerOperation* orig_op,
                                   std::unique_ptr<EagerOperation>* mkl_op);

  // Creates new MKL op for Conv2D, Conv2DBackpropInput and
  // Conv2DBackpropFilter.
  static Status CreateMklConv2DOp(
      EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_conv2d_op);

  // Rewrite rule for Conv2D, Conv2DBackpropInput and Conv2DBackpropFilter.
  static bool RewriteConv2D(EagerOperation* op);

  // Calls op-specific rewrite function to create new MKL op.
  Status RewriteToMklOp(EagerOperation* orig_op,
                        std::unique_ptr<EagerOperation>* mkl_op,
                        const int op_idx);

  // Checks whether we can rewrite the op to MKL one or not.
  bool ShouldRewriteOp(EagerOperation* op, int* op_idx);

  // Default rewrite rule to be used when rewrite should happen without any
  // restriction.
  static bool AlwaysRewrite(EagerOperation* op) { return true; }
};

REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, MklEagerOpRewrite);

// Constructor
MklEagerOpRewrite::MklEagerOpRewrite(string name, string file, string line)
    : EagerOpRewrite(name, file, line) {
  mkl_eager_ops_.push_back({"BatchMatMul", AlwaysRewrite, CreateGenericMklOp});
  mkl_eager_ops_.push_back(
      {"BatchMatMulV2", AlwaysRewrite, CreateGenericMklOp});
  mkl_eager_ops_.push_back({"Conv2D", RewriteConv2D, CreateMklConv2DOp});
  mkl_eager_ops_.push_back(
      {"Conv2DBackpropInput", RewriteConv2D, CreateMklConv2DOp});
  mkl_eager_ops_.push_back(
      {"Conv2DBackpropFilter", RewriteConv2D, CreateMklConv2DOp});
  mkl_eager_ops_.push_back({"MatMul", AlwaysRewrite, CreateGenericMklOp});
}

Status MklEagerOpRewrite::Run(
    EagerOperation* orig_op,
    std::unique_ptr<tensorflow::EagerOperation>* out_op) {
  int found_op_idx = -1;
  if (ShouldRewriteOp(orig_op, &found_op_idx)) {
    TF_CHECK_OK(RewriteToMklOp(orig_op, out_op, found_op_idx));
  }
  return Status::OK();
}

Status MklEagerOpRewrite::SetupNewOp(
    EagerOperation* orig_op, const string mkl_op_name,
    std::unique_ptr<EagerOperation>* new_mkl_op) {
  const tensorflow::AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(
      tensorflow::AttrTypeMapForOp(mkl_op_name.c_str(), &types, &is_function));
  EagerContext* ctx = orig_op->EagerContext();
  new_mkl_op->reset(new tensorflow::EagerOperation(ctx, mkl_op_name.c_str(),
                                                   is_function, types));

  int num_inputs = orig_op->Inputs().size();
  // Add all inputs to the new op.
  for (int i = 0; i < num_inputs; ++i) {
    (*new_mkl_op)->AddInput(orig_op->Inputs()[i]);
  }

  // Copy all attributes to the new op.
  string name;
  const NodeDef& orig_ndef = orig_op->MutableAttrs()->BuildNodeDef();

  AttrSlice attr_list(orig_ndef);
  for (const auto& attr : attr_list) {
    (*new_mkl_op)->MutableAttrs()->Set(attr.first, attr.second);
  }

  (*new_mkl_op)
      ->MutableAttrs()
      ->Set("_kernel", mkl_op_registry::kMklNameChangeOpLabel);

  if (orig_op->Device() != nullptr) {
    (*new_mkl_op)->SetDevice(orig_op->Device());
  } else {
    string device_name =
        DeviceNameUtils::ParsedNameToString(orig_op->GetDeviceName());
    (*new_mkl_op)->SetDeviceName(device_name.c_str());
  }
  return Status::OK();
}

Status MklEagerOpRewrite::CreateGenericMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op) {
  const string mkl_op_name = mkl_op_registry::GetMklOpName(orig_op->Name());
  TF_CHECK_OK(SetupNewOp(orig_op, mkl_op_name, mkl_op));
  return Status::OK();
}

Status MklEagerOpRewrite::CreateMklConv2DOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_conv2d_op) {
  const string mkl_op_name =
      mkl_op_registry::GetMklEagerOpName(orig_op->Name());
  TF_CHECK_OK(SetupNewOp(orig_op, mkl_op_name, mkl_conv2d_op));
  return Status::OK();
}

bool MklEagerOpRewrite::ShouldRewriteOp(EagerOperation* op, int* op_idx) {
  // Don't rewrite the op if MKL use is disabled at runtime.
  if (DisableMKL()) {
    return false;
  }
  DataType data_type;
  if (op->Attrs().Get("T", &data_type) != Status::OK()) {
    return false;
  }
  // Check if we have registered MKL kernel for this op.
  if (!mkl_op_registry::IsMklNameChangeOp(
          mkl_op_registry::GetMklEagerOpName(op->Name()), data_type) &&
      !mkl_op_registry::IsMklNameChangeOp(
          mkl_op_registry::GetMklOpName(op->Name()), data_type)) {
    return false;
  }

  *op_idx = -1;
  // Find and call the op's rewrite rule that determines whether we need to
  // rewrite this op or not.
  for (auto it = mkl_eager_ops_.begin(); it != mkl_eager_ops_.end(); ++it) {
    if (it->op_name.compare(op->Name()) == 0 && it->RewriteRule(op)) {
      *op_idx = it - mkl_eager_ops_.begin();
      return true;
    }
  }
  return false;
}

Status MklEagerOpRewrite::RewriteToMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op,
    const int op_idx) {
  mkl_eager_ops_[op_idx].CreateMklOp(orig_op, mkl_op);
  return Status::OK();
}

bool MklEagerOpRewrite::RewriteConv2D(EagerOperation* op) {
  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  string padding;
  TF_CHECK_OK(GetNodeAttr(ndef, "padding", &padding));
  // Right now MKL Conv2D does not support explicit padding.
  return (padding != "EXPLICIT");
}

}  // namespace tensorflow
#endif  // INTEL_MKL
