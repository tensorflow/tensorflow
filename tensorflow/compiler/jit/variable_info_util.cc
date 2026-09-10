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

#include "tensorflow/compiler/jit/variable_info_util.h"

#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace {

bool AllowHostResidentVars() {
  static const bool allow = [] {
    const char* value = std::getenv("TF_XLA_ALLOW_HOST_RESIDENT_VARS");
    return value != nullptr && absl::string_view(value) == "1";
  }();
  return allow;
}

// Copies a variable that lives on a host (CPU) device into `rm`, the resource
// manager of the executing device `dev`. The copy is made once and reused, so
// later host-side updates are not reflected. Like DeviceVariablesTable in
// core/tfrt/gpu/kernel/gpurt_kernels.cc, staged copies are never evicted.
absl::Status StageHostResidentVariable(OpKernelContext* ctx,
                                       const ResourceHandle& handle,
                                       DeviceBase* dev, ResourceMgr* rm,
                                       Var** out_variable) {
  if (ctx->function_library() == nullptr ||
      ctx->function_library()->device_mgr() == nullptr ||
      ctx->op_device_context() == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot stage host-resident variable ", handle.name(),
                     ": no DeviceMgr or device context."));
  }
  const DeviceMgr* device_mgr = ctx->function_library()->device_mgr();

  Device* src_device = nullptr;
  TF_RETURN_IF_ERROR(device_mgr->LookupDevice(handle.device(), &src_device));
  if (src_device->device_type() != DEVICE_CPU) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot stage variable ", handle.name(), " from ", handle.device(),
        ": only host-resident variables can be staged."));
  }
  Device* dst_device = nullptr;
  TF_RETURN_IF_ERROR(
      device_mgr->LookupDevice(dev->attributes().name(), &dst_device));

  Var* src_variable = nullptr;
  TF_RETURN_IF_ERROR(src_device->resource_manager()->Lookup<Var>(
      handle.container(), handle.name(), &src_variable));
  core::ScopedUnref src_unref(src_variable);

  Var* dst_variable = nullptr;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<Var>(handle.container(), handle.name(),
                                             &dst_variable, [](Var** ptr) {
                                               *ptr = new Var(DT_INVALID);
                                               return absl::OkStatus();
                                             }));

  mutex_lock dst_lock(*dst_variable->mu());
  if (dst_variable->is_initialized) {
    *out_variable = dst_variable;
    return absl::OkStatus();
  }

  Tensor host_tensor;
  {
    tf_shared_lock src_lock(*src_variable->mu());
    if (!src_variable->is_initialized) {
      dst_variable->Unref();
      return absl::FailedPreconditionError(
          absl::StrCat("Host-resident variable ", handle.name(), " on ",
                       handle.device(), " is not initialized."));
    }
    host_tensor = *src_variable->tensor();
  }

  AllocatorAttributes attr;
  Tensor device_tensor(dev->GetAllocator(attr), host_tensor.dtype(),
                       host_tensor.shape());

  absl::Status copy_status;
  absl::Notification done;
  ctx->op_device_context()->CopyCPUTensorToDevice(
      &host_tensor, dst_device, &device_tensor,
      [&copy_status, &done](const absl::Status& s) {
        copy_status = s;
        done.Notify();
      });
  done.WaitForNotification();
  if (!copy_status.ok()) {
    dst_variable->Unref();
    return copy_status;
  }

  *dst_variable->tensor() = device_tensor;
  dst_variable->is_initialized = true;
  *out_variable = dst_variable;
  return absl::OkStatus();
}

}  // namespace

absl::Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                        absl::Span<const Tensor* const> inputs,
                                        absl::Span<const int> variable_indices,
                                        std::vector<VariableInfo>* result) {
  return GetVariableInfosFromInputs(rm, dev, inputs, variable_indices, nullptr,
                                    result);
}

absl::Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                        absl::Span<const Tensor* const> inputs,
                                        absl::Span<const int> variable_indices,
                                        const std::set<int>* variables_updated,
                                        std::vector<VariableInfo>* result) {
  return GetVariableInfosFromInputs(rm, dev, inputs, variable_indices,
                                    variables_updated, /*ctx=*/nullptr, result);
}

absl::Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                        absl::Span<const Tensor* const> inputs,
                                        absl::Span<const int> variable_indices,
                                        const std::set<int>* variables_updated,
                                        OpKernelContext* ctx,
                                        std::vector<VariableInfo>* result) {
  result->clear();
  result->reserve(variable_indices.size());
  for (int var_idx : variable_indices) {
    Var* variable = nullptr;
    if (inputs[var_idx]->NumElements() == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Empty resource tensor passed  at index ", var_idx,
                       " to GetVariableInfosFromInputs."));
    }
    const ResourceHandle& handle = inputs[var_idx]->flat<ResourceHandle>()(0);
    if (handle.device() != dev->attributes().name()) {
      // With TF_XLA_ALLOW_HOST_RESIDENT_VARS=1, stage host-resident variables
      // onto `dev` instead of failing.
      if (!AllowHostResidentVars() || ctx == nullptr) {
        std::string definition_location =
            DefinitionLocationMsg(handle.definition_stack_trace());
        return absl::InvalidArgumentError(absl::StrCat(
            "Trying to access resource ", handle.name(), definition_location,
            " located in device ", handle.device(), " from device ",
            dev->attributes().name(),
            "\n Cf. "
            "https://www.tensorflow.org/xla/"
            "known_issues#tfvariable_on_a_different_device"));
      }
      TF_RETURN_IF_ERROR(
          StageHostResidentVariable(ctx, handle, dev, rm, &variable));
    } else {
      TF_RETURN_IF_ERROR(rm->LookupOrCreate<Var>(
          handle.container(), handle.name(), &variable, [](Var** ptr) {
            // This var is uninitialized for now.
            *ptr = new Var(DT_INVALID);
            return absl::OkStatus();
          }));
    }
    VariableInfo& variable_info = result->emplace_back(
        var_idx, handle.name(), variable, handle.definition_stack_trace());
    if (variables_updated != nullptr &&
        variables_updated->find(var_idx) == variables_updated->end()) {
      variable_info.set_read_only();
    }
  }
  return absl::OkStatus();
}

absl::Status LockVariables(absl::Span<VariableInfo*> variables) {
  std::vector<int> lock_order(variables.size());
  std::iota(lock_order.begin(), lock_order.end(), 0);

  // VariableInfoComparator orders all empty VariableInfo instances as
  // equivalent so it looks like we may want to stable sort these to maintain a
  // deterministic order between the empty VariableInfo instances.  However
  // since we're sorting by pointer value the sort is pretty non-deterministic
  // anyway so we don't bother using std::stable_sort for now.
  absl::c_sort(lock_order, [&](int a, int b) {
    if (variables[a]->var() && variables[b]->var()) {
      return variables[a]->var()->mu() < variables[b]->var()->mu();
    }

    // Move all the empty VariableInfo instances to the end.
    return variables[a]->var() != nullptr;
  });

  mutex* prev = nullptr;
  for (int i : lock_order) {
    Var* variable = variables[i]->var();
    if (variable == nullptr) {
      // All empty VariableInfo instances are at the end of the order
      // so we're done.
      break;
    }
    mutex* mu = variable->mu();
    if (prev == mu) {
      // It is an error to pass the same variable handle twice to the same XLA
      // cluster because we would not handle variable updates correctly.  Any
      // locks we have already acquired will be released when the VariableInfo
      // objects are destroyed.
      // TODO(b/128495870) Add support for passing aliased resource variables.
      return absl::UnimplementedError(
          "Duplicate variable passed to XLA cluster");
    }
    if (variables[i]->read_only()) {
      VLOG(4) << "Acquiring reader lock for variable "
              << reinterpret_cast<void*>(variable);
      mu->lock_shared();
      variables[i]->set_shared_lock_held();
    } else {
      VLOG(4) << "Acquiring lock for variable "
              << reinterpret_cast<void*>(variable);
      mu->lock();
      variables[i]->set_lock_held();
    }
    prev = mu;
  }
  VLOG(4) << "Finished acquiring variable locks.";
  return absl::OkStatus();
}

absl::Status LockVariables(absl::Span<VariableInfo> variables) {
  std::vector<VariableInfo*> variable_ptrs;
  variable_ptrs.reserve(variables.size());
  for (auto& var : variables) {
    variable_ptrs.push_back(&var);
  }
  return LockVariables(absl::MakeSpan(variable_ptrs));
}

absl::Status SnapshotResourceVariables(
    OpKernelContext* ctx, absl::Span<const int> variable_indices,
    absl::Span<VariableInfo const> variable_infos,
    ResourceVarsSnapshot* result) {
  for (int i = 0, end = variable_indices.size(); i < end; i++) {
    Var* var = variable_infos[i].var();
    (*result)[variable_indices[i]] =
        var ? std::make_optional(*var->tensor()) : std::nullopt;
  }
  return absl::OkStatus();
}

std::vector<int> GetResourceVariableIndicesFromContext(OpKernelContext* ctx) {
  std::vector<int> out;
  for (int64_t i = 0; i < ctx->num_inputs(); i++) {
    if (ctx->input(i).dtype() == DT_RESOURCE) {
      out.push_back(i);
    }
  }
  return out;
}

absl::Status CreateVariableInfoLookup(
    absl::Span<VariableInfo const> variable_args,
    absl::flat_hash_map<int, const VariableInfo*>& variable_info_lookup) {
  for (const VariableInfo& info : variable_args) {
    if (!(!info.var() || info.lock_held() || info.shared_lock_held())) {
      return absl::InternalError(
          "Need to hold the lock on resource variables "
          "before calling BuildXlaCompilerArguments");
    }
    variable_info_lookup.emplace(info.index(), &info);
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
