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
#include "tensorflow/core/framework/collective.h"

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace {
// A RegistrationInfo object stores a collective implementation registration
// details.  `factory` is used to create instances of the collective
// implementation.
struct RegistrationInfo {
  // This constructor also creates, and stores in `param_resolver_instance`,
  // what is effectively a static instance of the collective implementation.
  // During param resolution of collective ops we return this static instance.
  // The actual op execution gets a fresh instance using `factory`.
  RegistrationInfo(const string& n, CollectiveRegistry::Factory f)
      : name(n),
        factory(std::move(f)),
        param_resolver_instance(this->factory()) {}
  string name;
  CollectiveRegistry::Factory factory;
  CollectiveImplementationInterface* param_resolver_instance;
};

std::vector<RegistrationInfo>* MutableCollectiveRegistry() {
  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}
}  // namespace

string CollGroupRuntimeDetails::ToString() const {
  return strings::StrCat("CollGroupRuntimeDetails {communicator_key=",
                         absl::CEscape(communicator_key), "}");
}

string CollGroupParams::ToString() const {
  string v = strings::StrCat(
      "CollGroupParams {group_key=", group_key, " group_size=", group_size,
      " device_type=", device_type.type_string(), " num_tasks=", num_tasks,
      " runtime_details=", runtime_details.ToString(), " devices {");
  for (const auto& m : members) {
    strings::StrAppend(&v, m.device.name(), ",");
  }
  strings::StrAppend(&v, "} num_devices_per_task={");
  for (const auto& dpt : num_devices_per_task) {
    strings::StrAppend(&v, dpt.first, ": ", dpt.second, ", ");
  }
  strings::StrAppend(&v, "}");
  return v;
}

CollInstanceParams& CollInstanceParams::operator=(
    const CollInstanceParams& other) {
  if (this != &other) {
    instance_key = other.instance_key;
    type = other.type;
    data_type = other.data_type;
    shape = other.shape;
    impl_details.subdiv_offsets.assign(
        other.impl_details.subdiv_offsets.begin(),
        other.impl_details.subdiv_offsets.end());
    impl_details.subdiv_permutations.clear();
    for (auto p : other.impl_details.subdiv_permutations) {
      impl_details.subdiv_permutations.push_back(
          std::vector<int>(p.begin(), p.end()));
    }
    impl_details.subdiv_source_rank.assign(
        other.impl_details.subdiv_source_rank.begin(),
        other.impl_details.subdiv_source_rank.end());
    impl_details.dependencies = other.impl_details.dependencies;
    devices.assign(other.devices.begin(), other.devices.end());
    permutation.assign(other.permutation.begin(), other.permutation.end());
  }
  return *this;
}

string CollInstanceParams::ToString() const {
  string v =
      strings::StrCat("CollInstanceParams { instance_key=", instance_key,
                      " type=", type, " data_type=", DataTypeString(data_type),
                      " shape=", shape.DebugString(), " devices {");
  strings::StrAppend(&v, "}, collective_name=", impl_details.collective_name,
                     ", subdiv_offsets={");
  strings::StrAppend(&v, "}, subdiv_offsets={");
  for (const auto& d : impl_details.subdiv_offsets) {
    strings::StrAppend(&v, d, ",");
  }
  strings::StrAppend(&v, "}, subdiv_perms={");
  for (const auto& p : impl_details.subdiv_permutations) {
    strings::StrAppend(&v, "{");
    for (const auto& i : p) {
      strings::StrAppend(&v, i, ",");
    }
    strings::StrAppend(&v, "}");  // one subdiv
  }
  if (!impl_details.subdiv_source_rank.empty()) {
    strings::StrAppend(&v, " subdiv_source_rank={");
    for (const auto& r : impl_details.subdiv_source_rank) {
      strings::StrAppend(&v, r, ",");
    }
    strings::StrAppend(&v, "}");
  }  // all subdivs
  if (type == PERMUTE_COLLECTIVE) {
    strings::StrAppend(&v, "}, permute_devices {");
    for (const auto& d : devices) {
      strings::StrAppend(&v, d, ",");
    }
    strings::StrAppend(&v, "}, permute_permutation {");
    for (const auto& p : permutation) {
      strings::StrAppend(&v, p, ",");
    }
    strings::StrAppend(&v, "}");
  }
  return v;
}

string CollectiveParams::ToString() const {
  string v = strings::StrCat("CollectiveParams ", name, " {", group.ToString());
  strings::StrAppend(&v, " ", instance.ToString());
  strings::StrAppend(&v, " default_rank=", default_rank,
                     " is_source=", is_source, " source_rank=", source_rank,
                     " subdiv_rank={");
  for (const auto& r : subdiv_rank) {
    strings::StrAppend(&v, r, ",");
  }
  strings::StrAppend(&v, "}}");
  return v;
}

/*static*/ OpKernelContext::Params* CollectiveExecutor::CtxParams(
    OpKernelContext* ctx) {
  return ctx->params_;
}

CollectiveContext::CollectiveContext(
    CollectiveExecutor* col_exec, NcclCommunicatorInterface* nccl_communicator,
    const DeviceMgr* dev_mgr, OpKernelContext* ctx,
    OpKernelContext::Params* op_params, const CollectiveParams* col_params,
    const string& exec_key, int64_t step_id, const Tensor* input,
    Tensor* output)
    : col_exec(col_exec),
      nccl_communicator(nccl_communicator),
      dev_mgr(dev_mgr),
      op_ctx(ctx),
      op_params(op_params),
      col_params(col_params, /*add_ref=*/true),
      exec_key(exec_key),
      step_id(step_id),
      input(input),
      output(output),
      device(nullptr),
      device_name(
          col_params->group.members[col_params->default_rank].device.name()) {}

/*static*/
int64_t CollectiveExecutor::kInvalidId = -1;

/*static*/
Status CollectiveRegistry::Lookup(
    const string& collective_name,
    CollectiveImplementationInterface** implementation) {
  return LookupHelper(collective_name, implementation, false);
}

/*static*/
Status CollectiveRegistry::LookupParamResolverInstance(
    const string& collective_name,
    CollectiveImplementationInterface** implementation) {
  return LookupHelper(collective_name, implementation, true);
}

/*static*/
void CollectiveRegistry::GetAll(
    std::vector<CollectiveImplementationInterface*>* implementations) {
  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry)
    implementations->emplace_back(reg_info.factory());
}

/*static*/
Status CollectiveRegistry::Register(const string& collective_name,
                                    Factory factory) {
  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry) {
    if (reg_info.name == collective_name)
      return errors::Internal("Already registered collective ",
                              collective_name);
  }
  registry->emplace_back(collective_name, std::move(factory));
  return OkStatus();
}

/*static*/
Status CollectiveRegistry::LookupHelper(
    const string& collective_name,
    CollectiveImplementationInterface** implementation, bool param_resolver) {
  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry) {
    if (reg_info.name == collective_name) {
      if (param_resolver) {
        *implementation = reg_info.param_resolver_instance;
      } else {
        *implementation = reg_info.factory();
      }
      return OkStatus();
    }
  }
  return errors::Internal(
      "CollectiveRegistry::Lookup did not find collective implementation ",
      collective_name);
}

}  // namespace tensorflow
