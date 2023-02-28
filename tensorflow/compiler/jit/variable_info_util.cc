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

#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                  absl::Span<const Tensor* const> inputs,
                                  absl::Span<const int> variable_indices,
                                  std::vector<VariableInfo>* result) {
  return GetVariableInfosFromInputs(rm, dev, inputs, variable_indices, nullptr,
                                    result);
}

Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                  absl::Span<const Tensor* const> inputs,
                                  absl::Span<const int> variable_indices,
                                  const std::set<int>* variables_updated,
                                  std::vector<VariableInfo>* result) {
  result->clear();
  result->reserve(variable_indices.size());
  for (int var_idx : variable_indices) {
    Var* variable = nullptr;
    ResourceHandle handle = inputs[var_idx]->flat<ResourceHandle>()(0);
    if (handle.device() != dev->attributes().name()) {
      auto pos1 = handle.device().find("STREAM");
      auto pos2 = dev->attributes().name().find("STREAM");
      auto pos3 = handle.device().rfind(":");
      auto pos4 = dev->attributes().name().rfind(":");
      if (pos1 == string::npos || pos2 == string::npos ||
          handle.device().substr(0, pos3) !=
              dev->attributes().name().substr(0, pos4)) {
        std::string definition_location =
            DefinitionLocationMsg(handle.definition_stack_trace());
        return errors::InvalidArgument(
            "Trying to access resource ", handle.name(), definition_location,
            " located in device ", handle.device(), " from device ",
            dev->attributes().name(),
            "\n Cf. "
            "https://www.tensorflow.org/xla/"
            "known_issues#tfvariable_on_a_different_device");
      }
    }
    TF_RETURN_IF_ERROR(rm->LookupOrCreate<Var>(
        handle.container(), handle.name(), &variable, [](Var** ptr) {
          // This var is uninitialized for now.
          *ptr = new Var(DT_INVALID);
          return OkStatus();
        }));
    VariableInfo& variable_info = result->emplace_back(
        var_idx, handle.name(), variable, handle.definition_stack_trace());
    if (variables_updated != nullptr &&
        variables_updated->find(var_idx) == variables_updated->end()) {
      variable_info.set_read_only();
    }
  }
  return OkStatus();
}

Status LockVariables(absl::Span<VariableInfo*> variables) {
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
      return errors::Unimplemented("Duplicate variable passed to XLA cluster");
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
  return OkStatus();
}

Status LockVariables(absl::Span<VariableInfo> variables) {
  std::vector<VariableInfo*> variable_ptrs;
  variable_ptrs.reserve(variables.size());
  for (auto& var : variables) {
    variable_ptrs.push_back(&var);
  }
  return LockVariables(absl::MakeSpan(variable_ptrs));
}

Status SnapshotResourceVariables(OpKernelContext* ctx,
                                 absl::Span<const int> variable_indices,
                                 absl::Span<VariableInfo const> variable_infos,
                                 ResourceVarsSnapshot* result) {
  for (int i = 0, end = variable_indices.size(); i < end; i++) {
    Var* var = variable_infos[i].var();
    (*result)[variable_indices[i]] =
        var ? std::make_optional(*var->tensor()) : std::nullopt;
  }
  return OkStatus();
}

std::vector<int> GetResourceVariableIndicesFromContext(OpKernelContext* ctx) {
  std::vector<int> out;
  for (int64 i = 0; i < ctx->num_inputs(); i++) {
    if (ctx->input(i).dtype() == DT_RESOURCE) {
      out.push_back(i);
    }
  }
  return out;
}

Status CreateVariableInfoLookup(
    absl::Span<VariableInfo const> variable_args,
    absl::flat_hash_map<int, const VariableInfo*>& variable_info_lookup) {
  for (const VariableInfo& info : variable_args) {
    if (!(!info.var() || info.lock_held() || info.shared_lock_held())) {
      return errors::Internal(
          "Need to hold the lock on resource variables "
          "before calling BuildXlaCompilerArguments");
    }
    variable_info_lookup.emplace(info.index(), &info);
  }
  return OkStatus();
}

}  // namespace tensorflow
