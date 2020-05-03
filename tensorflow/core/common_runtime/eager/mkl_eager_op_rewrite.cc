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
#include <string>
#include <unordered_map>

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
  struct MklEagerOp {
    string op_name;
    std::function<bool(EagerOperation*)> RewriteRule;
    std::function<Status(EagerOperation*, std::unique_ptr<EagerOperation>*)>
        CreateMklOp;
  };

 private:
  std::unordered_map<std::string, MklEagerOp> mkl_eager_ops_;

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
                        std::unique_ptr<EagerOperation>* mkl_op);

  // Check whether we can rewrite the op to MKL one or not.
  bool ShouldRewriteOp(EagerOperation* op);

  // Default rewrite rule to be used when rewrite should happen without any
  // restriction.
  static bool AlwaysRewrite(EagerOperation* op) { return true; }

  // Check if kernel is registered for a particular op.
  bool FastCheckIfKernelRegistered(std::string op_name, DataType dt);

  // This is called by FastCheckIfKernelRegistered once per unique op name
  // and data type.
  bool SlowCheckIfKernelRegistered(std::string op_name, DataType dt);

  // Helper function to insert mkl_eager_ops to Map
  void InsertMKLEagerOps(MklEagerOp op);

  // Map used by FastCheckIfKernelRegistered.
  std::unordered_map<std::string, bool> registered_kernels_map_;
};

REGISTER_REWRITE(EagerOpRewriteRegistry::PRE_EXECUTION, MklEagerOpRewrite);

// Constructor
MklEagerOpRewrite::MklEagerOpRewrite(string name, string file, string line)
    : EagerOpRewrite(name, file, line), registered_kernels_map_() {
  InsertMKLEagerOps({"BatchMatMul", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"BatchMatMulV2", AlwaysRewrite, CreateGenericMklOp});
  InsertMKLEagerOps({"Conv2D", RewriteConv2D, CreateMklConv2DOp});
  InsertMKLEagerOps({"Conv2DBackpropInput", RewriteConv2D, CreateMklConv2DOp});
  InsertMKLEagerOps({"Conv2DBackpropFilter", RewriteConv2D, CreateMklConv2DOp});
  InsertMKLEagerOps({"MatMul", AlwaysRewrite, CreateGenericMklOp});
};

void MklEagerOpRewrite::InsertMKLEagerOps(MklEagerOp op) {
  mkl_eager_ops_.insert(std::make_pair(op.op_name, op));
}

Status MklEagerOpRewrite::Run(
    EagerOperation* orig_op,
    std::unique_ptr<tensorflow::EagerOperation>* out_op) {
  if (ShouldRewriteOp(orig_op)) {
    TF_CHECK_OK(RewriteToMklOp(orig_op, out_op));
  }
  return Status::OK();
}

Status MklEagerOpRewrite::SetupNewOp(
    EagerOperation* orig_op, const string mkl_op_name,
    std::unique_ptr<EagerOperation>* new_mkl_op) {
  bool is_remote = false;
  new_mkl_op->reset(new tensorflow::EagerOperation(&orig_op->EagerContext()));
  TF_RETURN_IF_ERROR(new_mkl_op->get()->Reset(mkl_op_name.c_str(), nullptr,
                                              is_remote, nullptr));

  int num_inputs = orig_op->Inputs().size();
  // Add all inputs to the new op.
  for (int i = 0; i < num_inputs; ++i) {
    (*new_mkl_op)->AddInput(orig_op->Inputs()[i]);
  }

  // Copy all attributes to the new op.
  const NodeDef& orig_ndef = orig_op->MutableAttrs()->BuildNodeDef();

  AttrSlice attr_list(orig_ndef);
  for (const auto& attr : attr_list) {
    (*new_mkl_op)->MutableAttrs()->Set(attr.first, attr.second);
  }

  (*new_mkl_op)
      ->MutableAttrs()
      ->Set("_kernel", mkl_op_registry::kMklNameChangeOpLabel);

  string device_name = orig_op->DeviceName();
  return (*new_mkl_op)->SetDeviceName(device_name.c_str());
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

bool MklEagerOpRewrite::ShouldRewriteOp(EagerOperation* op) {
  // Don't rewrite the op if MKL use is disabled at runtime.
  if (DisableMKL()) {
    return false;
  }
  DataType data_type;
  if (op->Attrs().Get("T", &data_type) != Status::OK()) {
    return false;
  }
  // Check if we have registered MKL kernel for this op.
  bool kernel_found = FastCheckIfKernelRegistered(op->Name(), data_type);
  if (!kernel_found) {
    return false;
  }

  // Find and call the op's rewrite rule that determines whether we need to
  // rewrite this op or not.
  auto it = mkl_eager_ops_.find(op->Name());
  if (it != mkl_eager_ops_.end()) {
    // Eager op found so verify Rewrite
    if (it->second.RewriteRule(op)) {
      return true;
    }
  }
  return false;
}

bool MklEagerOpRewrite::FastCheckIfKernelRegistered(std::string op_name,
                                                    DataType dt) {
  // Check for kernel registration only once per op name and data type
  // for performance reasons.
  string registered_kernels_key = op_name + std::to_string(dt);
  auto kernel_element = registered_kernels_map_.find(registered_kernels_key);
  bool kernel_registered = false;
  if (kernel_element == registered_kernels_map_.end()) {
    // Kernel registration is not verified even once yet.
    // So verify and store registration.
    kernel_registered = SlowCheckIfKernelRegistered(op_name, dt);
    registered_kernels_map_.insert(
        std::make_pair(registered_kernels_key, kernel_registered));
  } else {
    // Kernel is visited atleast once. return stored registration result.
    kernel_registered = kernel_element->second;
  }

  return kernel_registered;
}

bool MklEagerOpRewrite::SlowCheckIfKernelRegistered(string op_name,
                                                    DataType dt) {
  // Find if the eager op_name exists in mkl_eager_ops_ list.
  auto element = mkl_eager_ops_.find(op_name);
  if (element != mkl_eager_ops_.end() && dt == DT_FLOAT) {
    // Eager Op exists. So verify registry and return registered or not.
    return (mkl_op_registry::IsMklNameChangeOp(
                mkl_op_registry::GetMklEagerOpName(op_name), dt) ||
            mkl_op_registry::IsMklNameChangeOp(
                mkl_op_registry::GetMklOpName(op_name), dt));
  } else {
    return false;
  }
}

Status MklEagerOpRewrite::RewriteToMklOp(
    EagerOperation* orig_op, std::unique_ptr<EagerOperation>* mkl_op) {
  // TODO(intel-tf): mkl_eager_ops_ lookup can be reduced from twice
  // (once each in ShouldRewriteOp & RewriteToMklOp) to just once.
  mkl_eager_ops_[orig_op->Name()].CreateMklOp(orig_op, mkl_op);
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
