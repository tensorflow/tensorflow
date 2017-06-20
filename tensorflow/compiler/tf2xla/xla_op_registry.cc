/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

#include <functional>
#include <memory>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace tensorflow {

const char* const DEVICE_CPU_XLA_JIT = "XLA_CPU_JIT";
const char* const DEVICE_GPU_XLA_JIT = "XLA_GPU_JIT";
const char* const DEVICE_XLA_CPU = "XLA_CPU";
const char* const DEVICE_XLA_GPU = "XLA_GPU";

// Is platform 'id' supported by XLA?
static bool IsPlatformSupported(perftools::gputools::Platform::Id id) {
  auto platform = perftools::gputools::MultiPlatformManager::PlatformWithId(id);
  if (!platform.ok()) return false;
  return xla::ClientLibrary::GetOrCreateLocalClient(platform.ValueOrDie()).ok();
}

XlaOpRegistry::XlaOpRegistry() = default;
XlaOpRegistry::~XlaOpRegistry() = default;

/* static */ void XlaOpRegistry::RegisterCompilationDevice(
    const string& device_name, const DeviceRegistration& registration) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto result =
      registry.compilation_devices_.emplace(device_name, registration);
  CHECK(result.second || result.first->second.compilation_device_name ==
                             registration.compilation_device_name);
}

/* static */ void XlaOpRegistry::RegisterBackend(
    const string& compilation_device_name,
    gtl::ArraySlice<DataType> supported_types, BackendOpFilter op_filter) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto result = registry.backends_.emplace(compilation_device_name, Backend());
  CHECK(result.second) << "Duplicate XLA backend registration "
                       << compilation_device_name;
  result.first->second.supported_types.insert(supported_types.begin(),
                                              supported_types.end());
  result.first->second.op_filter = op_filter;
}

/* static */ bool XlaOpRegistry::GetCompilationDevice(
    const string& device_name, const DeviceRegistration** registration) {
  XlaOpRegistry& registry = Instance();

  // Lazily register the CPU and GPU JIT devices the first time
  // GetCompilationDevice is called.
  static void* registration_init = [&registry]() {
    mutex_lock lock(registry.mutex_);
    if (IsPlatformSupported(perftools::gputools::host::kHostPlatformId)) {
      DeviceRegistration& registration =
          registry.compilation_devices_[DEVICE_CPU];
      registration.compilation_device_name = DEVICE_CPU_XLA_JIT;
      registration.requires_compilation = false;
      registration.enable_jit_by_default = false;
      registration.compile_resource_ops = false;
    }
    if (IsPlatformSupported(perftools::gputools::cuda::kCudaPlatformId)) {
      DeviceRegistration& registration =
          registry.compilation_devices_[DEVICE_GPU];
      registration.compilation_device_name = DEVICE_GPU_XLA_JIT;
      registration.requires_compilation = false;
      registration.enable_jit_by_default = true;
      registration.compile_resource_ops = false;
    }
    return nullptr;
  }();
  (void)registration_init;

  mutex_lock lock(registry.mutex_);
  auto it = registry.compilation_devices_.find(device_name);
  if (it == registry.compilation_devices_.end()) return false;
  *registration = &it->second;
  return true;
}

void XlaOpRegistry::RegisterCompilationKernels() {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);

  if (registry.jit_kernels_registered_) return;
  registry.jit_kernels_registered_ = true;

  OpRegistryInterface* op_registry = OpRegistry::Global();
  for (const auto& op : registry.ops_) {
    const OpDef* op_def;
    TF_CHECK_OK(op_registry->LookUpOpDef(op.first, &op_def));

    std::unordered_set<string> type_attrs;
    for (const OpDef::AttrDef& attr_def : op_def->attr()) {
      if (attr_def.type() == "type" || attr_def.type() == "list(type)") {
        type_attrs.insert(attr_def.name());
      }
    }

    // Checks there are no type constraints referring to unknown attributes.
    for (const auto& constraint : op.second->type_constraints) {
      if (type_attrs.find(constraint.first) == type_attrs.end()) {
        LOG(FATAL) << "Unknown type attribute " << constraint.first
                   << " in XLA op registration for " << op.first;
      }
    }

    for (auto& backend : registry.backends_) {
      // If the operator has a device whitelist, only register on whitelisted
      // devices.
      if (op.second->has_device_whitelist &&
          op.second->device_whitelist.find(backend.first) ==
              op.second->device_whitelist.end()) {
        continue;
      }

      std::unique_ptr<KernelDef> kdef(new KernelDef);
      kdef->set_op(op.second->name);
      kdef->set_device_type(backend.first);

      // Constrain each type attribute to the intersection of:
      // a) the types supported by the backend, and
      // b) the attribute's type constraints.
      // TODO(phawkins): it may be necessary to also take the intersection with
      // the set of types supported by the OpDef.
      for (const string& type_attr : type_attrs) {
        KernelDef::AttrConstraint* attr_constraint = kdef->add_constraint();
        attr_constraint->set_name(type_attr);
        auto* allowed_values =
            attr_constraint->mutable_allowed_values()->mutable_list();

        auto it = op.second->type_constraints.find(type_attr);
        for (DataType dtype : backend.second.supported_types) {
          if (it == op.second->type_constraints.end() ||
              (it != op.second->type_constraints.end() &&
               it->second.find(dtype) != it->second.end())) {
            allowed_values->add_type(dtype);
          }
        }
        if (op.second->allow_resource_types) {
          allowed_values->add_type(DT_RESOURCE);
        }
      }
      if (backend.second.op_filter != nullptr &&
          !backend.second.op_filter(kdef.get())) {
        continue;
      }
      VLOG(2) << "XLA op registration: device: " << backend.first
              << " op: " << op.first;
      registry.kernel_registrars_.emplace_back(
          new kernel_factory::OpKernelRegistrar(
              new KernelDef(*kdef), "XlaJitOp", op.second->factory));
      backend.second.kernel_defs.push_back(std::move(kdef));
    }
  }
}

std::vector<const KernelDef*> XlaOpRegistry::DeviceKernels(
    const string& compilation_device_name) {
  std::vector<const KernelDef*> kernels;
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto it = registry.backends_.find(compilation_device_name);
  CHECK(it != registry.backends_.end())
      << "Unknown backend " << compilation_device_name;
  for (const std::unique_ptr<KernelDef>& k : it->second.kernel_defs) {
    if (!registry.ops_.at(k->op())->compilation_only) {
      kernels.push_back(k.get());
    }
  }
  return kernels;
}

XlaOpRegistry& XlaOpRegistry::Instance() {
  static XlaOpRegistry* r = new XlaOpRegistry;
  return *r;
}

XlaOpRegistrationBuilder::XlaOpRegistrationBuilder(StringPiece name) {
  registration_.reset(new XlaOpRegistry::OpRegistration);
  registration_->name = name.ToString();
}

XlaOpRegistrationBuilder XlaOpRegistrationBuilder::Name(StringPiece name) {
  XlaOpRegistrationBuilder registration(name);
  return registration;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Device(
    gtl::ArraySlice<StringPiece> devices) {
  registration_->has_device_whitelist = true;
  for (StringPiece device : devices) {
    registration_->device_whitelist.insert(device.ToString());
  }
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Device(StringPiece device) {
  registration_->has_device_whitelist = true;
  registration_->device_whitelist.insert(device.ToString());
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::CompilationOnly() {
  registration_->compilation_only = true;
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::AllowResourceTypes() {
  registration_->allow_resource_types = true;
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::TypeConstraint(
    StringPiece attr_name, DataType allowed) {
  std::set<DataType>& types =
      registration_->type_constraints[attr_name.ToString()];
  types.insert(allowed);
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::TypeConstraint(
    StringPiece attr_name, gtl::ArraySlice<DataType> allowed) {
  std::set<DataType>& types =
      registration_->type_constraints[attr_name.ToString()];
  for (DataType t : allowed) {
    types.insert(t);
  }
  return *this;
}

std::unique_ptr<XlaOpRegistry::OpRegistration> XlaOpRegistrationBuilder::Build(
    XlaOpRegistry::Factory factory) {
  registration_->factory = factory;
  return std::move(registration_);
}

XlaOpRegistrar::XlaOpRegistrar(
    std::unique_ptr<XlaOpRegistry::OpRegistration> registration) {
  XlaOpRegistry& registry = XlaOpRegistry::Instance();
  mutex_lock lock(registry.mutex_);
  auto result = registry.ops_.emplace(registration->name, nullptr);
  if (!result.second) {
    LOG(FATAL) << "Duplicate XLA op registration " << registration->name;
  }
  result.first->second = std::move(registration);
}

XlaBackendRegistrar::XlaBackendRegistrar(
    StringPiece name, gtl::ArraySlice<DataType> types,
    XlaOpRegistry::BackendOpFilter op_filter) {
  XlaOpRegistry& registry = XlaOpRegistry::Instance();
  registry.RegisterBackend(name.ToString(), types, op_filter);
}

bool CpuOpFilter(KernelDef* kdef) {
  // TODO(b/34339814): implement inverse erf for double types and remove this
  // workaround.
  if (kdef->op() == "RandomStandardNormal") {
    kdef->clear_constraint();
    // Change the type constraint to permit only DTD_FLOAT.
    KernelDef::AttrConstraint* attr_constraint = kdef->add_constraint();
    attr_constraint->set_name("dtype");
    attr_constraint->mutable_allowed_values()->mutable_list()->add_type(
        DT_FLOAT);
    return true;
  }
  return true;
}

REGISTER_XLA_BACKEND(DEVICE_CPU_XLA_JIT, kCpuAllTypes, CpuOpFilter);

bool GpuOpFilter(KernelDef* kdef) {
  // TODO(b/31361304): The GPU backend does not parallelize PRNG ops, leading to
  // slow code.
  // TODO(b/34969189) The implementation of TruncatedNormal generates illegal
  // code on GPU.
  if (kdef->op() == "RandomStandardNormal" || kdef->op() == "RandomUniform" ||
      kdef->op() == "RandomUniformInt" || kdef->op() == "TruncatedNormal") {
    return false;
  }
  return true;
}

REGISTER_XLA_BACKEND(DEVICE_GPU_XLA_JIT, kGpuAllTypes, GpuOpFilter);

}  // namespace tensorflow
