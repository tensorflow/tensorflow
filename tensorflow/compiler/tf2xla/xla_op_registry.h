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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_OP_REGISTRY_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_OP_REGISTRY_H_

#include <functional>
#include <memory>
#include <set>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// Names of the XLA compilation devices. These are not user-visible, and are
// used internally by the Tensorflow/XLA bridge to perform symbolic execution of
// a Tensorflow graph.

extern const char* const DEVICE_CPU_XLA_JIT;  // "CPU_XLA_JIT"
extern const char* const DEVICE_GPU_XLA_JIT;  // "GPU_XLA_JIT"

extern const char* const DEVICE_XLA_CPU;
extern const char* const DEVICE_XLA_GPU;

constexpr std::array<DataType, 2> kIntTypes = {{DT_INT32, DT_INT64}};
constexpr std::array<DataType, 3> kFloatTypes = {
    {DT_HALF, DT_FLOAT, DT_DOUBLE}};
constexpr std::array<DataType, 5> kNumericTypes = {
    {DT_INT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE}};

constexpr std::array<DataType, 5> kCpuAllTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};

constexpr std::array<DataType, 5> kGpuAllTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};

// Class that manages registrations of operators and devices for the XLA JIT.
// Not thread-safe.
class XlaOpRegistry {
 public:
  typedef OpKernel* (*Factory)(OpKernelConstruction*);

  // Describes how to compile operators assigned to a device.
  struct DeviceRegistration {
    // The name of the an XLA compilation device to use to compile code.
    string compilation_device_name;

    // Do operators assigned to this device require compilation?
    bool requires_compilation;

    // If !requires_compilation, should we try to JIT operators on this device
    // when XLA JIT compilation is enabled globally via the SessionOptions?
    // (It is still possible to explicitly mark operators to JIT compile, even
    // if enable_jit_by_default is false.)
    bool enable_jit_by_default;

    // Enable compilation of operators that use DT_RESOURCE types?
    bool compile_resource_ops = false;
  };

  // Registers an XLA backend. `compilation_device_name` is the name of the
  // device used for symbolic execution during compilation. `supported_types`
  // is the list of non-resource types supported by the device. Each operators
  // will be registered for the intersection of the operator's supported types
  // and the device's supported types. `backend_op_filter` is a function used
  // to exclude or modify operator registrations on the device; it may be
  // nullptr, in which case all ops are included.
  // `backend_op_filter` should return true if the op should be registered on
  // the device; it may optionally modify the KernelDef.
  typedef bool (*BackendOpFilter)(KernelDef* kdef);
  static void RegisterBackend(const string& compilation_device_name,
                              gtl::ArraySlice<DataType> supported_types,
                              BackendOpFilter op_filter);

  // Registers `device_name` for XLA compilation, using information from
  // `registration`.
  static void RegisterCompilationDevice(const string& device_name,
                                        const DeviceRegistration& registration);

  // Returns the JIT device name associated with 'device_name', setting
  // 'jit_device_name', 'requires_jit', and 'enabled_jit_by_default', if they
  // are not null. Returns false and leaves the outputs unchanged if no matching
  // JIT device is registered.
  // '*enable_jit_by_default' is set to true if we should try to JIT using this
  // device when the JIT is enabled via the Session OptimizerOptions.
  static bool GetCompilationDevice(const string& device_name,
                                   const DeviceRegistration** registration);

  // Registers all JIT kernels on JIT devices, if not already registered.
  // Does nothing otherwise.
  static void RegisterCompilationKernels();

  // Returns KernelDefs for compilation ops registered on
  // 'compilation_device_name'.
  // Does not include kernels registered as CompilationOnly.
  static std::vector<const KernelDef*> DeviceKernels(
      const string& compilation_device_name);

 private:
  friend class XlaBackendRegistrar;
  friend class XlaOpRegistrar;
  friend class XlaOpRegistrationBuilder;

  static XlaOpRegistry& Instance();

  XlaOpRegistry();
  ~XlaOpRegistry();

  mutex mutex_;

  // Describes an XLA backend.
  struct Backend {
    // Which types are supported by this device?
    std::set<DataType> supported_types;

    // The per-backend operator filter function. See the comment on
    // RegisterBackend() for details.
    BackendOpFilter op_filter;

    // KernelDefs built by RegisterCompilationKernels() for each op supported
    // by the device.
    std::vector<std::unique_ptr<KernelDef>> kernel_defs;
  };

  // Map from compilation device names to a description of the backend.
  std::unordered_map<string, Backend> backends_ GUARDED_BY(mutex_);

  // Map from Tensorflow device names to the corresponding JIT device metadata.
  std::unordered_map<string, DeviceRegistration> compilation_devices_
      GUARDED_BY(mutex_);

  // A description of a Tensorflow operator that can be compiled to XLA.
  struct OpRegistration {
    string name;

    // Should this operator be registered only on compilation devices, without a
    // dummy kernel registered on the corresponding XLA device?
    bool compilation_only = false;

    // Should we allow resource types for type attributes? Used by _Arg to
    // allow DT_RESOURCE.
    bool allow_resource_types = false;

    // Mapping from attribute name to a list of supported types.
    std::unordered_map<string, std::set<DataType>> type_constraints;

    // An optional whitelist of devices. If there is no whitelist, all devices
    // are permitted.
    bool has_device_whitelist = false;
    std::unordered_set<string> device_whitelist;

    // Factory used to build OpKernels that perform symbolic execution.
    Factory factory;
  };

  // Returns true if registrations x and y can both be added to the registry.
  // This is always the case if they refer to different ops. If they refer to
  // the same op name, they must: have the same values for compilation_only and
  // allow_resource_types; use a device_whitelist; and their
  // whitelists must not intersect.
  static bool IsCompatible(const OpRegistration& x, const OpRegistration& y);

  // Map from operator name to OpRegistrations, populated by REGISTER_XLA_OP.
  // Registrations present under the same key must satisfy IsCompatible above,
  // and this is checked during registration.
  std::unordered_multimap<string, std::unique_ptr<OpRegistration>> ops_
      GUARDED_BY(mutex_);

  // Have we already registered the JIT kernels on the JIT devices?
  bool jit_kernels_registered_ = false;

  // Holds ownership of OpKernelRegistrars that represent the Tensorflow kernel
  // registrations created by RegisterCompilationKernels() and
  // RegisterDeviceKernels().
  std::vector<std::unique_ptr<kernel_factory::OpKernelRegistrar>>
      kernel_registrars_ GUARDED_BY(mutex_);
};

// REGISTER_XLA_OP() registers an XLA OpKernel by name, for example:
// REGISTER_XLA_OP(Name("Add"), AddOp);
// where 'AddOp' is the name of a JIT OpKernel class that implements "Add".
//
// We don't use a variadic macro here because we don't expect JIT operators to
// be templated.

#define REGISTER_XLA_OP(NAME, OP) \
  REGISTER_XLA_OP_UNIQ_HELPER(__COUNTER__, NAME, OP)

class XlaOpRegistrationBuilder {
 public:
  // Starts an operator registration chain.
  static XlaOpRegistrationBuilder Name(StringPiece name);

  // Specifies a whitelist of devices on which the operator may run.
  XlaOpRegistrationBuilder& Device(StringPiece devices);
  XlaOpRegistrationBuilder& Device(gtl::ArraySlice<StringPiece> devices);

  // Specifies a type constraint for a type variable attribute. Each constraint
  // specifies the set of types that the type variable may assume.
  XlaOpRegistrationBuilder& TypeConstraint(StringPiece attr_name,
                                           DataType allowed);

  XlaOpRegistrationBuilder& TypeConstraint(StringPiece attr_name,
                                           gtl::ArraySlice<DataType> allowed);

  // Specifies that a dummy copy of this operator should not be registered on
  // XLA_* devices, but may be used during compilation.
  XlaOpRegistrationBuilder& CompilationOnly();

  // Allow DT_RESOURCE types for type parameters.
  XlaOpRegistrationBuilder& AllowResourceTypes();

  std::unique_ptr<XlaOpRegistry::OpRegistration> Build(
      XlaOpRegistry::Factory factory);

 private:
  XlaOpRegistrationBuilder(StringPiece name);

  std::unique_ptr<XlaOpRegistry::OpRegistration> registration_;
};

// REGISTER_XLA_BACKEND() registers an XLA backend. Example usage:
// REGISTER_XLA_BACKEND(DEVICE_GPU_XLA_JIT, kGpuAllTypes, GpuOpFilter);
#define REGISTER_XLA_BACKEND(NAME, ...) \
  REGISTER_XLA_BACKEND_UNIQ_HELPER(__COUNTER__, NAME, __VA_ARGS__)

// Implementation details.

class XlaOpRegistrar {
 public:
  XlaOpRegistrar(std::unique_ptr<XlaOpRegistry::OpRegistration> registration);
};

#define REGISTER_XLA_OP_UNIQ_HELPER(COUNTER, BUILDER, OP) \
  REGISTER_XLA_OP_UNIQ(COUNTER, BUILDER, OP)

#define REGISTER_XLA_OP_UNIQ(CTR, BUILDER, OP)                                 \
  static ::tensorflow::XlaOpRegistrar xla_op_registrar__body__##CTR##__object( \
      XlaOpRegistrationBuilder::BUILDER.Build(                                 \
          [](::tensorflow::OpKernelConstruction* context)                      \
              -> ::tensorflow::OpKernel* { return new OP(context); }));

class XlaBackendRegistrar {
 public:
  XlaBackendRegistrar(StringPiece name, gtl::ArraySlice<DataType> types,
                      XlaOpRegistry::BackendOpFilter op_filter = nullptr);
};

#define REGISTER_XLA_BACKEND_UNIQ_HELPER(COUNTER, NAME, ...) \
  REGISTER_XLA_BACKEND_UNIQ(COUNTER, NAME, __VA_ARGS__)

#define REGISTER_XLA_BACKEND_UNIQ(CTR, NAME, ...) \
  static ::tensorflow::XlaBackendRegistrar        \
      xla_backend_registrar__body__##CTR##__object(NAME, __VA_ARGS__);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_OP_REGISTRY_H_
