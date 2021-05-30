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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace jit {
class DeviceInfoCache;
class DeviceSet;

// Instances of DeviceId represent TensorFlow devices as integers.
//
// This helps avoid having to manipulate device names as strings when
// auto-clustering.
class DeviceId {
 public:
  DeviceId(DeviceId&&) = default;
  DeviceId(const DeviceId&) = default;
  DeviceId& operator=(const DeviceId&) = default;

  bool operator==(const DeviceId& other) const { return id() == other.id(); }
  bool operator!=(const DeviceId& other) const { return !(*this == other); }

 private:
  int id_;

  explicit DeviceId(int id) : id_(id) {}

  int id() const { return id_; }

  friend class DeviceInfoCache;
  friend class DeviceSet;
};

// A set of DeviceIds, represented as a bitmap.
class DeviceSet {
 public:
  void Insert(DeviceId device_id);
  void UnionWith(const DeviceSet& other);
  bool IsEmpty() const;

  // Calls `func` on each DeviceId in the set.  Stops iterating early if `func`
  // return false.
  //
  // TODO(sanjoy): Change this to take a typed std::function if that's
  // performance neutral.
  template <typename FnTy>
  void ForEach(FnTy func) const {
    // This is really a poor man's iterator, we should consider writing a proper
    // iterator if this ends up being used widely.
    for (int word_index = 0, end = storage_.size(); word_index < end;
         word_index++) {
      uint64 word = storage_[word_index];
      while (word != 0) {
        uint64 only_lowest_bit_set = word & -word;
        // The number of trailing zeros in a non-zero word is the index of the
        // least significant 1.
        int bit_index = ctz_uint64(word);
        if (!func(DeviceId(word_index * kWordSize + bit_index))) {
          return;
        }
        word ^= only_lowest_bit_set;
      }
    }
  }

 private:
  static int ctz_uint64(uint64 x) {
    DCHECK_NE(x, 0);
#ifdef __GNUC__
    return __builtin_ctzl(x);
#else
    int result = 0u;
    while ((x & 1u) == 0u) {
      x >>= 1;
      ++result;
    }
    return result;
#endif
  }

  absl::InlinedVector<uint64, 1> storage_;

  const int kWordSize = 64;
};

// Caches some miscellaneous information about TF devices.  Thread compatible.
class DeviceInfoCache {
 public:
  bool IsGpu(DeviceId device) const { return is_gpu_[device.id()]; }
  bool IsCpu(DeviceId device) const { return is_cpu_[device.id()]; }

  absl::string_view GetNameFor(DeviceId device) const {
    return names_[device.id()];
  }

  StatusOr<DeviceId> GetIdFor(absl::string_view name);

  using DeviceRegistration = const XlaOpRegistry::DeviceRegistration;

  DeviceRegistration* GetCompilationDevice(DeviceId device) const {
    return id_to_compilation_device_[device.id()];
  }

  StatusOr<DeviceRegistration*> GetCompilationDevice(absl::string_view name) {
    TF_ASSIGN_OR_RETURN(DeviceId device_id, GetIdFor(name));
    return GetCompilationDevice(device_id);
  }

  const DeviceType& GetDeviceTypeFor(DeviceId device) const {
    return *id_to_device_type_[device.id()];
  }

  using DeviceTypeConstRef = std::reference_wrapper<const DeviceType>;

  StatusOr<DeviceTypeConstRef> GetDeviceTypeFor(absl::string_view device_name) {
    TF_ASSIGN_OR_RETURN(DeviceId device_id, GetIdFor(device_name));
    return std::cref(*id_to_device_type_[device_id.id()]);
  }

  string DebugString(const DeviceSet& device_set) const;

 private:
  absl::flat_hash_map<string, DeviceId> name_to_id_;

  // These fields are populated for a device in GetIdFor, *before* we give out a
  // DeviceId.
  std::vector<const XlaOpRegistry::DeviceRegistration*>
      id_to_compilation_device_;
  std::vector<std::unique_ptr<DeviceType>> id_to_device_type_;
  std::vector<string> names_;
  std::vector<bool> is_cpu_;
  std::vector<bool> is_gpu_;
};

}  // namespace jit

// Returns the DeviceType corresponding to 'device'.
Status DeviceNameToDeviceType(const string& device, DeviceType* device_type);

// Picks the device for which XLA should compile a cluster that contains
// operations placed in devices in `devices`.  For instance a cluster that
// contains operations solely placed on the CPU will be compiled into a CPU
// executable by XLA, whereas a cluster that contains operations placed on the
// CPU and also operations placed on the GPU will be compiled into a GPU
// executable.
//
// Returns a non-OK Status if no unambiguous choice of device exists.
//
// We choose the device using the following rules:
//
//  - It is an error for `device_names` to contain more than one device of the
//    same type.
//  - GPU is preferred over CPU.
//  - If `allow_mixing_unknown_and_cpu` is true then unknown devices are
//    preferred over CPU.
//  - XLA devices count as "unrecognized devices".
//
// This set of rules above implicitly assume that XLA:GPU can compile all
// operations in the cluster that XLA:CPU can compile, and if
// `allow_mixing_unknown_and_cpu` then the unrecognized device can also compile
// all operations in the cluster that XLA:CPU can compile.
//
// We provide the `allow_mixing_unknown_and_cpu` knob so that we can do both of
// the following things:
//
// - Let MarkForCompilationPass not inject CPU-placed operations into clusters
//   that will run on unknown devices (because the unknown XLA backend may not
//   support every operation supported by CPU).
// - Let BuildXlaOpsPass successfully infer a compilation device for a cluster
//   that contains nodes placed on both the CPU and on unknown devices.  In this
//   case it is the responsibility of the optimization pass that injected the
//   CPU nodes into the cluster to ensure that these nodes can be compiled by
//   the unknown XLA backend.
StatusOr<jit::DeviceId> PickDeviceForXla(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu);

// This is like `PickDeviceForXla` except that it returns nullopt (instead of a
// non-OK Status) if no unambiguous choice of device exists.
//
// We return a failing Status for errors unrelated to the device choice
// algorithm itself.
StatusOr<absl::optional<jit::DeviceId>> MaybePickDeviceForXla(
    const jit::DeviceInfoCache& device_info_cache,
    const jit::DeviceSet& devices, bool allow_mixing_unknown_and_cpu);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
