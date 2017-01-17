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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILATION_DEVICE_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILATION_DEVICE_H_

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

// Names of the XLA JIT devices. These are not user-visible, and are used
// internally by the JIT to perform symbolic execution of a Tensorflow graph.

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

// Class is declared and defined in tla_jit_device.cc, reference
// included here only so the XlaCompilationDevice allocator_ member can be
// defined.
class XlaCompilationAllocator;

// Deliberately don't register the device factory because we *never*
// want soft placement to put Ops on an JIT device. Tests can include
// the tla_jit_test_deps target which registers the factory, and when
// using JIT in practice, the device is created manually not using a
// factory.

// This is a 'dummy' TensorFlow device that is only used to execute a
// subgraph of XLA compilation Ops to construct a compiled version
// of the subgraph's computation. It has a 'dummy' allocator that
// backs each Tensor with metadata indicating the computation the
// Tensor represents.
class XlaCompilationDevice : public LocalDevice {
 public:
  XlaCompilationDevice(const SessionOptions& options, DeviceType type);

  ~XlaCompilationDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;

  Status Sync() override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

 private:
  std::unique_ptr<XlaCompilationAllocator> allocator_;
};

// Class that manages registrations of operators and devices for the XLA JIT.
// Not thread-safe.
class XlaOpRegistry {
 public:
  typedef OpKernel* (*Factory)(OpKernelConstruction*);

  // Registers 'jit_device_name' as the JIT device corresponding to
  // 'device_name'. If 'requires_jit' is true, then operators placed on this
  // device must be JIT-compiled. Dies if a conflicting registration already
  // exists.
  static void RegisterJitDevice(const string& device_name,
                                const string& jit_device_name,
                                bool requires_jit);

  // Returns the JIT device name associated with 'device_name', setting
  // 'jit_device_name' and 'requires_jit', if they are not null. Returns false
  // and leaves 'jit_device_name' and 'requires_jit' unchanged if no matching
  // JIT device is registered.
  static bool GetJitDevice(const string& device_name,
                           const string** jit_device_name, bool* requires_jit);

  // Registers all JIT kernels on JIT devices, if not already registered.
  // Does nothing otherwise.
  static void RegisterJitKernels();

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

  // Map from Tensorflow device names to the corresponding JIT device names.
  std::unordered_map<string, std::pair<string, bool>> jit_devices_
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
  // registrations created by RegisterJitKernels() and RegisterDeviceKernels().
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
      NAME, [](::tensorflow::OpKernelConstruction* context)                    \
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

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILATION_DEVICE_H_
