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
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace tensorflow {

const char* const DEVICE_CPU_XLA_JIT = "XLA_CPU_JIT";
const char* const DEVICE_GPU_XLA_JIT = "XLA_GPU_JIT";
const char* const DEVICE_XLA_CPU = "XLA_CPU";
const char* const DEVICE_XLA_GPU = "XLA_GPU";

static Status LaunchOpHasKernelForDevice(const DeviceType& device_type) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef("_XlaLaunch", &op_def));
  NodeDef node_def;
  node_def.set_name("_XlaLaunch-op");
  node_def.set_op("_XlaLaunch");
  string kernel_class_name;
  TF_RETURN_IF_ERROR(FindKernelDef(device_type, node_def, /*KernelDef*/ nullptr,
                                   &kernel_class_name));
  VLOG(1) << "LaunchOpHasKernelForDevice"
          << " kernel_class_name: " << kernel_class_name;
  return Status::OK();
}

XlaOpRegistry::XlaOpRegistry() = default;
XlaOpRegistry::~XlaOpRegistry() = default;

// TODO(b/64575122) consider adding more sophisticated definitions of
// compatibility if needed by future use cases.
/* static */ bool XlaOpRegistry::IsCompatible(const OpRegistration& x,
                                              const OpRegistration& y) {
  if (x.name != y.name) return true;
  // The registrations refer to the same Op: ensures they are compatible and
  // are restricted to different device whitelists.
  if (x.compilation_only != y.compilation_only) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible compilation_only settings.";
    return false;
  }
  if (x.allow_resource_types != y.allow_resource_types) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible allow_resource_types settings.";
    return false;
  }
  if (!x.has_device_whitelist || !y.has_device_whitelist) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " do not both have device whitelists.";
    return false;
  }
  for (const auto& device : x.device_whitelist) {
    if (y.device_whitelist.count(device) != 0) {
      LOG(WARNING) << "Multiple registrations of " << x.name << " on device "
                   << device;
      return false;
    }
  }
  if (x.compile_time_constant_inputs != y.compile_time_constant_inputs) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible compile time constant inputs.";
    return false;
  }
  return true;
}

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
    if (LaunchOpHasKernelForDevice(DeviceType(DEVICE_CPU)).ok()) {
      DeviceRegistration& registration =
          registry.compilation_devices_[DEVICE_CPU];
      registration.compilation_device_name = DEVICE_CPU_XLA_JIT;
      registration.requires_compilation = false;
      registration.enable_jit_by_default = false;
      registration.compile_resource_ops = false;
    }
    if (LaunchOpHasKernelForDevice(DeviceType(DEVICE_GPU)).ok()) {
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
    const string& op_name = op.first;
    const std::unique_ptr<OpRegistration>& op_registration = op.second;
    const OpDef* op_def;
    Status lookup_status = op_registry->LookUpOpDef(op_name, &op_def);
    if (!lookup_status.ok()) {
      LOG(ERROR) << lookup_status.error_message();
      XLA_LOG_LINES(
          ERROR, "Ops registered: \n" +
                     dynamic_cast<OpRegistry*>(op_registry)->DebugString(true));
    }
    TF_CHECK_OK(lookup_status);

    std::unordered_set<string> type_attrs;
    for (const OpDef::AttrDef& attr_def : op_def->attr()) {
      if (attr_def.type() == "type" || attr_def.type() == "list(type)") {
        type_attrs.insert(attr_def.name());
      }
    }

    // Checks there are no type constraints referring to unknown attributes.
    for (const auto& constraint : op_registration->type_constraints) {
      if (type_attrs.find(constraint.first) == type_attrs.end()) {
        LOG(FATAL) << "Unknown type attribute " << constraint.first
                   << " in XLA op registration for " << op_name;
      }
    }

    for (auto& backend : registry.backends_) {
      // If the operator has a device whitelist, only register on whitelisted
      // devices.
      if (op_registration->has_device_whitelist &&
          op_registration->device_whitelist.find(backend.first) ==
              op_registration->device_whitelist.end()) {
        continue;
      }

      std::unique_ptr<KernelDef> kdef(new KernelDef);
      kdef->set_op(op_registration->name);
      kdef->set_device_type(backend.first);

      // Constrain each type attribute to the intersection of:
      // a) the types supported by the backend, and
      // b) the types allowed by the OpDef, and
      // c) the type constraints.
      for (const string& type_attr : type_attrs) {
        KernelDef::AttrConstraint* attr_constraint = kdef->add_constraint();
        attr_constraint->set_name(type_attr);
        auto* allowed_values =
            attr_constraint->mutable_allowed_values()->mutable_list();

        const OpDef::AttrDef& op_def_attr = *FindAttr(type_attr, *op_def);
        const auto* op_def_allowed_types =
            op_def_attr.has_allowed_values()
                ? &op_def_attr.allowed_values().list().type()
                : nullptr;
        auto constraint_it = op_registration->type_constraints.find(type_attr);
        const std::set<DataType>* type_constraints =
            constraint_it != op_registration->type_constraints.end()
                ? &constraint_it->second
                : nullptr;
        for (DataType dtype : backend.second.supported_types) {
          // Filter out types that aren't allowed by the OpDef.
          if (op_def_allowed_types != nullptr &&
              std::find(op_def_allowed_types->begin(),
                        op_def_allowed_types->end(),
                        dtype) == op_def_allowed_types->end()) {
            continue;
          }
          // Filter out types based on the type constraints.
          if (type_constraints != nullptr &&
              type_constraints->find(dtype) == type_constraints->end()) {
            continue;
          }
          // Passed all the filters, this type is allowed.
          allowed_values->add_type(dtype);
        }
        if (op_registration->allow_resource_types) {
          allowed_values->add_type(DT_RESOURCE);
        }
      }
      if (backend.second.op_filter != nullptr &&
          !backend.second.op_filter(kdef.get())) {
        continue;
      }
      VLOG(2) << "XLA op registration: device: " << backend.first
              << " op: " << op_name;
      registry.kernel_registrars_.emplace_back(
          new kernel_factory::OpKernelRegistrar(
              new KernelDef(*kdef), "XlaJitOp", op_registration->factory));
      backend.second.kernel_defs.push_back(std::move(kdef));
    }
  }
}

std::vector<const KernelDef*> XlaOpRegistry::DeviceKernels(
    const string& compilation_device_name,
    bool include_compilation_only_kernels) {
  // Ensure compilation kernels registered.
  RegisterCompilationKernels();
  std::vector<const KernelDef*> kernels;
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto it = registry.backends_.find(compilation_device_name);
  CHECK(it != registry.backends_.end())
      << "Unknown backend " << compilation_device_name;
  for (const std::unique_ptr<KernelDef>& k : it->second.kernel_defs) {
    auto op_iter = registry.ops_.find(k->op());
    CHECK(op_iter != registry.ops_.end());
    // The test in IsCompatible ensures that if there are multiple matching
    // registrations for this op name, they all have the same value of
    // compilation_only, so only the first match needs to be tested.
    if (include_compilation_only_kernels ||
        !op_iter->second->compilation_only) {
      kernels.push_back(k.get());
    }
  }
  return kernels;
}

/* static */ const std::unordered_set<string>*
XlaOpRegistry::CompileTimeConstantInputs(const string& op) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto it = registry.ops_.find(op);
  if (it == registry.ops_.end()) {
    return nullptr;
  }
  return &it->second->compile_time_constant_inputs;
}

std::vector<string> XlaOpRegistry::BackendNames() {
  std::vector<string> names;
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  for (const auto& backend_pair : registry.backends_) {
    names.push_back(backend_pair.first);
  }
  return names;
}

bool XlaOpRegistry::IsBackendRegistered(const string& name) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  return registry.backends_.find(name) != registry.backends_.end();
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

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::CompileTimeConstInput(
    StringPiece input_name) {
  registration_->compile_time_constant_inputs.insert(input_name.ToString());
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
  auto existing_ops = registry.ops_.equal_range(registration->name);
  for (auto existing = existing_ops.first; existing != existing_ops.second;
       ++existing) {
    if (!XlaOpRegistry::IsCompatible(*existing->second, *registration)) {
      LOG(FATAL)
          << "XLA op registration " << registration->name
          << " is incompatible with existing registration of the same name.";
    }
  }
  registry.ops_.emplace(registration->name, std::move(registration));
}

XlaBackendRegistrar::XlaBackendRegistrar(
    StringPiece name, gtl::ArraySlice<DataType> types,
    XlaOpRegistry::BackendOpFilter op_filter) {
  XlaOpRegistry& registry = XlaOpRegistry::Instance();
  registry.RegisterBackend(name.ToString(), types, op_filter);
}

}  // namespace tensorflow
