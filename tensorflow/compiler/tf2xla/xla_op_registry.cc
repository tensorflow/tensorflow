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
#include <string>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "xla/util.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_factory.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {

const char* const DEVICE_CPU_XLA_JIT = "XLA_CPU_JIT";
const char* const DEVICE_GPU_XLA_JIT = "XLA_GPU_JIT";
const char* const DEVICE_XLA_CPU = "XLA_CPU";
const char* const DEVICE_XLA_GPU = "XLA_GPU";

static absl::Status LaunchOpHasKernelForDevice(const DeviceType& device_type) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef("XlaLaunch", &op_def));
  NodeDef node_def;
  node_def.set_name("_XlaLaunch-op");
  node_def.set_op("XlaLaunch");
  string kernel_class_name;
  TF_RETURN_IF_ERROR(FindKernelDef(device_type, node_def, /*KernelDef*/ nullptr,
                                   &kernel_class_name));
  VLOG(1) << "LaunchOpHasKernelForDevice"
          << " kernel_class_name: " << kernel_class_name;
  return absl::OkStatus();
}

XlaOpRegistry::XlaOpRegistry() = default;
XlaOpRegistry::~XlaOpRegistry() = default;

// TODO(b/64575122) consider adding more sophisticated definitions of
// compatibility if needed by future use cases.
/* static */ bool XlaOpRegistry::IsCompatible(const OpRegistration& x,
                                              const OpRegistration& y) {
  if (x.name != y.name) return true;
  if (x.label != y.label) return true;
  // The registrations refer to the same Op: ensures they are compatible and
  // are restricted to different device allowlists.
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
  if (x.allow_variant_types != y.allow_variant_types) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible allow_variant_types settings.";
    return false;
  }
  if (x.allow_string_type != y.allow_string_type) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible allow_string_type settings.";
    return false;
  }
  if (!x.has_device_allowlist && !y.has_device_allowlist) {
    LOG(WARNING) << "Duplicate registrations of " << x.name
                 << " with no device allowlists.";
    return false;
  }
  if (x.has_device_allowlist && y.has_device_allowlist) {
    for (const auto& device : x.device_allowlist) {
      if (y.device_allowlist.count(device) != 0) {
        LOG(WARNING) << "Multiple registrations of " << x.name << " on device "
                     << device;
        return false;
      }
    }
  }
  if (x.compile_time_constant_inputs != y.compile_time_constant_inputs) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible compile time constant inputs.";
    return false;
  }
  if (x.is_metadata_op != y.is_metadata_op) {
    LOG(WARNING) << "Registrations of " << x.name
                 << " have incompatible values for is_metadata_op.";
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
    absl::Span<const DataType> supported_types, BackendOpFilter op_filter) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto result = registry.backends_.emplace(compilation_device_name, Backend());
  CHECK(result.second) << "Duplicate XLA backend registration "
                       << compilation_device_name;
  result.first->second.supported_types.insert(supported_types.begin(),
                                              supported_types.end());
  result.first->second.op_filter = op_filter;
}

/* static */ bool XlaOpRegistry::IsCompilationDevice(
    const string& device_name) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  return registry.backends_.find(device_name) != registry.backends_.end();
}

/* static */ bool XlaOpRegistry::GetCompilationDevice(
    const string& device_name, const DeviceRegistration** registration) {
  XlaOpRegistry& registry = Instance();

  // Lazily register the CPU and GPU JIT devices the first time
  // GetCompilationDevice is called.
  static void* registration_init = [&registry]() {
    MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
    bool cpu_global_jit = flags->tf_xla_cpu_global_jit;
    VLOG(2) << "tf_xla_cpu_global_jit = " << cpu_global_jit;

    mutex_lock lock(registry.mutex_);
    if (LaunchOpHasKernelForDevice(DeviceType(DEVICE_CPU)).ok()) {
      DeviceRegistration& registration =
          registry.compilation_devices_[DEVICE_CPU];
      registration.compilation_device_name = DEVICE_CPU_XLA_JIT;
      registration.autoclustering_policy =
          cpu_global_jit
              ? XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally
              : XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested;
    }
    if (LaunchOpHasKernelForDevice(DeviceType(DEVICE_GPU)).ok()) {
      DeviceRegistration& registration =
          registry.compilation_devices_[DEVICE_GPU];
      registration.compilation_device_name = DEVICE_GPU_XLA_JIT;
      registration.autoclustering_policy =
          XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally;
    }
    return nullptr;
  }();
  (void)registration_init;

  // Register GPU JIT devices for NextPluggableDevice if its jit_device_type is
  // `XLA_GPU_JIT`.
  if (DeviceFactory::IsPluggableDevice(device_name) &&
      GetPjRtClient(DeviceType(device_name)).ok()) {
    mutex_lock lock(registry.mutex_);

    NextPluggableDeviceFactory* device_factory =
        static_cast<NextPluggableDeviceFactory*>(
            DeviceFactory::GetFactory(device_name));
    if (device_factory != nullptr &&
        DeviceType(device_factory->compilation_device_name()) ==
            DeviceType(DEVICE_GPU_XLA_JIT) &&
        registry.compilation_devices_.find(device_name) ==
            registry.compilation_devices_.end()) {
      DeviceRegistration& registration =
          registry.compilation_devices_[device_name];
      registration.compilation_device_name = DEVICE_GPU_XLA_JIT;
      registration.autoclustering_policy =
          XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally;
    }
  }

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
  // Order of op registration:
  // The goal is to allow the co-existence of backend-specific kernels and
  // generic kernels. To achieve this, we enforce the following order of
  // registrations for one op:
  // 1. Process op registration with device allowlists:
  //      this pass registers backend-specific kernels for this op.
  // 2. Process op registration without device allowlists:
  //      this pass registers the kernels for all the other supported backends.
  for (auto& ops : registry.ops_) {
    const string& op_name = ops.first;
    std::vector<std::unique_ptr<OpRegistration>>& op_registrations = ops.second;
    // Partition the op registration so that the ones with device allowlists
    // precede the one without device allowlist.
    std::partition(op_registrations.begin(), op_registrations.end(),
                   [](const std::unique_ptr<OpRegistration>& op_reg) {
                     return op_reg->has_device_allowlist;
                   });

    // Collect a set of backend registered by ops with device allowlists.
    // The op registration without allowlists will register a generic kernel
    // for all other backends not in this set.
    std::unordered_set<string> allowlisted_backend;
    for (auto& op_registration : op_registrations) {
      if (op_registration->has_device_allowlist) {
        allowlisted_backend.insert(op_registration->device_allowlist.begin(),
                                   op_registration->device_allowlist.end());
      }
    }

    for (auto& op_registration : op_registrations) {
      const OpDef* op_def;
      absl::Status lookup_status = op_registry->LookUpOpDef(op_name, &op_def);
      if (!lookup_status.ok()) {
        LOG(ERROR) << lookup_status.message();
        XLA_LOG_LINES(
            ERROR,
            "Ops registered: \n" +
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
        // If the operator has a device allowlist, only register on allowlisted
        // devices.
        if (op_registration->has_device_allowlist &&
            op_registration->device_allowlist.find(backend.first) ==
                op_registration->device_allowlist.end()) {
          continue;
        }

        // If the operator does NOT has a device allowlist, skip all devices
        // that has already been registered.
        if (!op_registration->has_device_allowlist &&
            allowlisted_backend.find(backend.first) !=
                allowlisted_backend.end()) {
          continue;
        }

        std::unique_ptr<KernelDef> kdef(new KernelDef);
        kdef->set_op(op_registration->name);
        kdef->set_device_type(backend.first);
        kdef->set_label(op_registration->label);

        // Constrain each type attribute to the intersection of:
        // a) the types supported by the backend, and
        // b) the types allowed by the OpDef, and
        // c) the type constraints.
        bool unsatisfiable_type_constraint = false;
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
          auto constraint_it =
              op_registration->type_constraints.find(type_attr);
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
          if (op_registration->allow_variant_types) {
            allowed_values->add_type(DT_VARIANT);
          }
          if (op_registration->allow_string_type) {
            allowed_values->add_type(DT_STRING);
          }
          // Don't build KernelDefs that have unsatisfiable type constraints.
          if (allowed_values->type().empty()) {
            unsatisfiable_type_constraint = true;
            break;
          }
        }
        if (unsatisfiable_type_constraint) continue;

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
}

std::vector<const KernelDef*> XlaOpRegistry::DeviceKernels(
    const string& compilation_device_name,
    bool include_compilation_only_kernels) {
  // Ensure compilation kernels registered.
  RegisterCompilationKernels();
  std::vector<const KernelDef*> kernels;
  XlaOpRegistry& registry = Instance();
  std::string registered_backends =
      absl::StrJoin(registry.BackendNames(), ", ");
  mutex_lock lock(registry.mutex_);
  auto it = registry.backends_.find(compilation_device_name);

  CHECK(it != registry.backends_.end())
      << "Unknown backend " << compilation_device_name
      << "; Known backends are: " << registered_backends;
  for (const std::unique_ptr<KernelDef>& k : it->second.kernel_defs) {
    auto op_iter = registry.ops_.find(k->op());
    CHECK(op_iter != registry.ops_.end() && !op_iter->second.empty());
    // The test in IsCompatible ensures that if there are multiple matching
    // registrations for this op name, they all have the same value of
    // compilation_only, so only the first match needs to be tested.
    if (include_compilation_only_kernels ||
        !op_iter->second.front()->compilation_only) {
      kernels.push_back(k.get());
    }
  }
  return kernels;
}

/*static*/ std::vector<string> XlaOpRegistry::GetAllRegisteredOps() {
  std::vector<string> ops;
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  ops.reserve(registry.ops_.size());
  for (const auto& pair : registry.ops_) {
    ops.push_back(pair.first);
  }
  std::sort(ops.begin(), ops.end());
  return ops;
}

/*static*/ const std::unordered_set<std::string>*
XlaOpRegistry::CompileTimeConstantInputArgNames(const string& op) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto it = registry.ops_.find(op);
  static auto empty_set = new std::unordered_set<std::string>;
  if (it == registry.ops_.end() || it->second.empty()) {
    return empty_set;
  } else {
    return &it->second.front()->compile_time_constant_inputs;
  }
}

/* static */ absl::Status XlaOpRegistry::CompileTimeConstantInputs(
    const NodeDef& node_def, const OpKernel* op_kernel, const OpDef* op_def,
    std::vector<int>* result) {
  result->clear();

  DCHECK(op_def != nullptr || op_kernel != nullptr);

  std::unordered_set<string> compile_time_constant_inputs_from_attr;
  std::vector<string> compile_time_constant_inputs_vect_from_attr;

  const std::unordered_set<string>* compile_time_constant_inputs;

  if (TryGetNodeAttr(node_def, kXlaCompileTimeConstantInputsAttr,
                     &compile_time_constant_inputs_vect_from_attr)) {
    absl::c_copy(compile_time_constant_inputs_vect_from_attr,
                 std::inserter(compile_time_constant_inputs_from_attr,
                               compile_time_constant_inputs_from_attr.end()));
    compile_time_constant_inputs = &compile_time_constant_inputs_from_attr;
  } else {
    compile_time_constant_inputs =
        CompileTimeConstantInputArgNames(node_def.op());
    if (compile_time_constant_inputs->empty()) {
      return absl::OkStatus();
    }
  }

  VLOG(3) << "For operation "
          << (op_def != nullptr ? op_def->name() : op_kernel->name())
          << " required constants are: "
          << absl::StrJoin(*compile_time_constant_inputs, ", ");

  for (const string& input : *compile_time_constant_inputs) {
    if (op_def) {
      NameRangeMap input_name_ranges;
      TF_RETURN_IF_ERROR(
          NameRangesForNode(node_def, *op_def, &input_name_ranges, nullptr));
      auto name_range = input_name_ranges.find(input);
      if (name_range == input_name_ranges.end()) {
        continue;
      }

      for (int i = name_range->second.first; i < name_range->second.second;
           i++) {
        result->push_back(i);
      }
    } else {
      int start, stop;
      TF_CHECK_OK(op_kernel->InputRange(input, &start, &stop));
      for (int i = start; i < stop; ++i) {
        result->push_back(i);
      }
    }
  }

  absl::c_sort(*result);
  return absl::OkStatus();
}

/*static*/ bool XlaOpRegistry::IsMetadataOp(const string& op) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto it = registry.ops_.find(op);
  if (it == registry.ops_.end() || it->second.empty()) {
    return false;
  }

  // The test in IsCompatible ensures that if there are multiple matching
  // registrations for this op name, they all have the same value of
  // is_metadata_op, so only the first match is returned.
  return it->second.front()->is_metadata_op;
}

std::vector<string> XlaOpRegistry::BackendNames() {
  std::vector<string> names;
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  names.reserve(registry.backends_.size());
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

XlaOpRegistrationBuilder::XlaOpRegistrationBuilder(absl::string_view name) {
  registration_.reset(new XlaOpRegistry::OpRegistration);
  registration_->name = string(name);
}

XlaOpRegistrationBuilder XlaOpRegistrationBuilder::Name(
    absl::string_view name) {
  XlaOpRegistrationBuilder registration(name);
  return registration;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Device(
    absl::Span<const absl::string_view> devices) {
  registration_->has_device_allowlist = true;
  for (absl::string_view device : devices) {
    registration_->device_allowlist.emplace(device);
  }
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Device(
    absl::string_view device) {
  registration_->has_device_allowlist = true;
  registration_->device_allowlist.emplace(device);
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

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::AllowVariantTypes() {
  registration_->allow_variant_types = true;
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::AllowStringType() {
  registration_->allow_string_type = true;
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::TypeConstraint(
    absl::string_view attr_name, DataType allowed) {
  std::set<DataType>& types =
      registration_->type_constraints[string(attr_name)];
  types.insert(allowed);
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::TypeConstraint(
    absl::string_view attr_name, absl::Span<const DataType> allowed) {
  std::set<DataType>& types =
      registration_->type_constraints[string(attr_name)];
  for (DataType t : allowed) {
    types.insert(t);
  }
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::CompileTimeConstantInput(
    absl::string_view input_name) {
  registration_->compile_time_constant_inputs.emplace(input_name);
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::IsMetadataOp() {
  registration_->is_metadata_op = true;
  return *this;
}

XlaOpRegistrationBuilder& XlaOpRegistrationBuilder::Label(std::string label) {
  registration_->label = label;
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
  auto& existing_ops = registry.ops_[registration->name];
  for (auto& existing : existing_ops) {
    if (!XlaOpRegistry::IsCompatible(*existing, *registration)) {
      LOG(FATAL)
          << "XLA op registration " << registration->name
          << " is incompatible with existing registration of the same name.";
    }
  }
  existing_ops.emplace_back(std::move(registration));
}

XlaBackendRegistrar::XlaBackendRegistrar(
    absl::string_view name, absl::Span<const DataType> types,
    XlaOpRegistry::BackendOpFilter op_filter) {
  XlaOpRegistry& registry = XlaOpRegistry::Instance();
  registry.RegisterBackend(string(name), types, op_filter);

  AddSymbolicExecutionDevice(name);
}

}  // namespace tensorflow
