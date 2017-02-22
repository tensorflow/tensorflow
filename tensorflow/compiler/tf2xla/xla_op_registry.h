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

#include <map>
#include <memory>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
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

constexpr std::array<DataType, 5> kCpuAllTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};
constexpr std::array<DataType, 2> kCpuIntTypes = {{DT_INT32, DT_INT64}};
constexpr std::array<DataType, 2> kCpuFloatTypes = {{DT_FLOAT, DT_DOUBLE}};
constexpr std::array<DataType, 4> kCpuNumericTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}};

constexpr std::array<DataType, 5> kGpuAllTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};
constexpr std::array<DataType, 2> kGpuIntTypes = {{DT_INT32, DT_INT64}};
constexpr std::array<DataType, 2> kGpuFloatTypes = {{DT_FLOAT, DT_DOUBLE}};
constexpr std::array<DataType, 4> kGpuNumericTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}};

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

  // Returns KernelDefs for JIT ops registered on 'jit_device_type'.
  // Does not include kernels registered using REGISTER_XLA_JIT_ONLY_KERNEL.
  static std::vector<const KernelDef*> DeviceKernels(
      const string& jit_device_type);

 private:
  friend class XlaKernelRegistrar;
  friend class XlaOpRegistrar;

  static XlaOpRegistry& Instance();

  XlaOpRegistry();
  ~XlaOpRegistry();

  mutex mutex_;

  // Map from Tensorflow device names to the corresponding JIT device metadata.
  std::unordered_map<string, DeviceRegistration> compilation_devices_
      GUARDED_BY(mutex_);

  // Map from operator name to OpKernel factory, populated by REGISTER_XLA_OP.
  std::unordered_map<string, Factory> ops_ GUARDED_BY(mutex_);

  // Have we already registered the JIT kernels on the JIT devices?
  bool jit_kernels_registered_ = false;

  struct XlaKernel {
    // Should this kernel be registered only on JIT devices, without a dummy
    // kernel registered on the corresponding XLA device?
    bool jit_only;

    // KernelDef as built by REGISTER_XLA_KERNEL.
    std::unique_ptr<const KernelDef> kernel_def;
  };

  // Map from JIT device name to a vector of XLA kernel descriptors.
  std::unordered_map<string, std::vector<XlaKernel>> kernels_
      GUARDED_BY(mutex_);

  // Holds ownership of OpKernelRegistrars that represent the Tensorflow kernel
  // registrations created by RegisterCompilationKernels() and
  // RegisterDeviceKernels().
  std::vector<std::unique_ptr<kernel_factory::OpKernelRegistrar>>
      kernel_registrars_ GUARDED_BY(mutex_);
};

// REGISTER_XLA_OP() registers an XLA OpKernel by name, for example:
// REGISTER_XLA_OP("Add", AddOp);
// where 'AddOp' is the name of a JIT OpKernel class that implements "Add".
//
// We don't use a variadic macro here because we don't expect JIT operators to
// be templated.

#define REGISTER_XLA_OP(NAME, OP) \
  REGISTER_XLA_OP_UNIQ_HELPER(__COUNTER__, NAME, OP)

// REGISTER_XLA_KERNEL() associates an XLA OpKernel with a particular device and
// set of type constraints, e.g.,
// REGISTER_XLA_KERNEL(DEVICE_XLA_CPU_JIT,
//                     Name("Relu").TypeConstraint("T", DT_FLOAT));
//
// REGISTER_XLA_JIT_ONLY_KERNEL is similar to REGISTER_XLA_KERNEL(), but causes
// XlaOpRegistry::RegisterDeviceKernels() to ignore the kernel.

#define REGISTER_XLA_KERNEL(DEVICE, BUILDER) \
  REGISTER_XLA_KERNEL_UNIQ_HELPER(__COUNTER__, DEVICE, BUILDER, false)

#define REGISTER_XLA_JIT_ONLY_KERNEL(DEVICE, BUILDER) \
  REGISTER_XLA_KERNEL_UNIQ_HELPER(__COUNTER__, DEVICE, BUILDER, true)

// Implementation details.

class XlaOpRegistrar {
 public:
  XlaOpRegistrar(StringPiece name, XlaOpRegistry::Factory factory);
};

#define REGISTER_XLA_OP_UNIQ_HELPER(COUNTER, NAME, OP) \
  REGISTER_XLA_OP_UNIQ(COUNTER, NAME, OP)

#define REGISTER_XLA_OP_UNIQ(CTR, NAME, OP)                                    \
  static ::tensorflow::XlaOpRegistrar xla_op_registrar__body__##CTR##__object( \
      NAME,                                                                    \
      [](::tensorflow::OpKernelConstruction* context)                          \
          -> ::tensorflow::OpKernel* { return new OP(context); });

// Implementation details.
class XlaKernelRegistrar {
 public:
  XlaKernelRegistrar(bool jit_only, const KernelDef* def);
};

#define REGISTER_XLA_KERNEL_UNIQ_HELPER(COUNTER, DEVICE, BUILDER, JIT_ONLY) \
  REGISTER_XLA_KERNEL_UNIQ(COUNTER, DEVICE, BUILDER, JIT_ONLY)

#define REGISTER_XLA_KERNEL_UNIQ(CTR, DEVICE, BUILDER, JIT_ONLY) \
  static ::tensorflow::XlaKernelRegistrar                        \
      xla_kernel_registrar__body__##CTR##__object(               \
          JIT_ONLY,                                              \
          ::tensorflow::register_kernel::BUILDER.Device(DEVICE).Build());

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_OP_REGISTRY_H_
