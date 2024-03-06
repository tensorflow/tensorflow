/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_SHARDING_H_
#define XLA_PYTHON_SHARDING_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// placeholder for index annotation headers
#include "absl/types/span.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device_list.h"
#include "xla/python/sharded_device_array.h"
#include "xla/xla_data.pb.h"

namespace jax {

class Sharding {
 public:
  Sharding() = default;

  // This constructor is used in the fast path to retrieve the number of devices
  // without falling back to python. This is only used in the cpp path.
  explicit Sharding(int num_devices) : num_devices_(num_devices) {}

  virtual ~Sharding() = default;

  static int SafeNumDevices(pybind11::handle sharding);

 private:
  std::optional<int> num_devices_;
};

extern bool (*GetEnableMemories)();

// Checks if the memory kind is valid, and canonicalizes the
// memory kind to default memory on backends that support memories.
pybind11::object CheckAndCanonicalizeMemoryKind(pybind11::object memory_kind,
                                                PyDeviceList* device_list);

// Returns a hash that may sometimes return different hashes for equal values.
// It is not a correct implementation of `__hash__` in python, but it's fine
// for jit/pjit dispatch since it only causes spurious cache misses.
size_t ShardingHash(const pybind11::object& sharding);

bool ShardingEqual(const pybind11::object& a, const pybind11::object& b);

xla::ClientAndPtr<xla::PjRtMemorySpace> GetMemory(
    const xla::ClientAndPtr<xla::PjRtDevice>& device, const std::string& kind);

class XLACompatibleSharding : public Sharding {
 public:
  using Sharding::Sharding;

  ~XLACompatibleSharding() override = default;
};

class NamedSharding : public XLACompatibleSharding {
 public:
  NamedSharding(pybind11::object mesh, pybind11::object spec,
                pybind11::object memory_kind, pybind11::object parsed_pspec,
                pybind11::object manual_axes);

  const pybind11::object& mesh() const { return mesh_; }
  const pybind11::object& spec() const { return spec_; }
  const pybind11::object& memory_kind() const { return memory_kind_; }
  const pybind11::object& parsed_pspec() const { return parsed_pspec_; }
  const pybind11::object& manual_axes() const { return manual_axes_; }
  void set_parsed_pspec(pybind11::object parsed_pspec) {
    parsed_pspec_ = std::move(parsed_pspec);
  }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<NamedSharding>();
    return type;
  }

  std::shared_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  pybind11::object mesh_;
  pybind11::object spec_;
  pybind11::object memory_kind_;
  pybind11::object parsed_pspec_;
  pybind11::object manual_axes_;
  std::shared_ptr<PyDeviceList> internal_device_list_;
};

class SingleDeviceSharding : public XLACompatibleSharding {
 public:
  explicit SingleDeviceSharding(
      pybind11::object device, pybind11::object memory_kind = pybind11::none());

  // Used only in C++ to accelerate `PyArray::MakeFromSingleDeviceArray()`.
  SingleDeviceSharding(std::shared_ptr<xla::PyClient> client,
                       xla::ifrt::DeviceList device_list,
                       pybind11::object memory_kind);

  const pybind11::object& device() const { return device_; }
  const pybind11::object& memory_kind() const { return memory_kind_; }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<SingleDeviceSharding>();
    return type;
  }

  std::shared_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  pybind11::object device_;
  pybind11::object memory_kind_;
  std::shared_ptr<PyDeviceList> internal_device_list_;
};

// The C++ implementation of jax.PmapSharding in python. It contains a few key
// data members and methods that are performance-critical.
class PmapSharding : public XLACompatibleSharding {
 public:
  PmapSharding(pybind11::array devices, ShardingSpec sharding_spec);

  ~PmapSharding() override = default;

  pybind11::array devices() const { return devices_; }

  const ShardingSpec& sharding_spec() const { return sharding_spec_; }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<PmapSharding>();
    return type;
  }

  std::shared_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  pybind11::array devices_;
  ShardingSpec sharding_spec_;
  std::shared_ptr<PyDeviceList> internal_device_list_;
};

class GSPMDSharding : public XLACompatibleSharding {
 public:
  GSPMDSharding(pybind11::sequence devices, xla::OpSharding op_sharding,
                pybind11::object memory_kind, pybind11::object device_list)
      : GSPMDSharding(
            std::move(devices),
            xla::ValueOrThrow(xla::HloSharding::FromProto(op_sharding)),
            std::move(memory_kind), std::move(device_list)) {}

  GSPMDSharding(pybind11::sequence devices, xla::HloSharding op_sharding,
                pybind11::object memory_kind, pybind11::object device_list);

  const pybind11::tuple& devices() const { return devices_; }
  const pybind11::object& memory_kind() const { return memory_kind_; }

  size_t Hash() {
    if (!hash_.has_value()) {
      hash_ = CalculateHash();
    }
    return *hash_;
  }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<GSPMDSharding>();
    return type;
  }

  const xla::HloSharding& hlo_sharding() const { return hlo_sharding_; }

  bool operator==(const GSPMDSharding& other) const {
    return AreOpShardingsEqual(*this, other) &&
           this->devices().equal(other.devices()) &&
           this->memory_kind().equal(other.memory_kind());
  }

  std::shared_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  size_t CalculateHash() const {
    // We only hash `hlo_sharding_` here for performance.
    return absl::Hash<xla::HloSharding>()(hlo_sharding_);
  }

  static bool AreOpShardingsEqual(const GSPMDSharding& a,
                                  const GSPMDSharding& b) {
    // If the OpSharding object is the same, return true
    if (&a.hlo_sharding() == &b.hlo_sharding()) {
      return true;
    }
    // If both OpShardings are replicated, return true
    if (a.IsOpShardingReplicated() && b.IsOpShardingReplicated()) {
      return true;
    }
    return a.hlo_sharding() == b.hlo_sharding();
  }

  bool IsOpShardingReplicated() const {
    // For JAX, shardings with 1 device are considered as replicated in its
    // semantics so that downstream things continue to work.
    if (hlo_sharding_.tile_assignment().num_elements() == 1) {
      return true;
    }
    return hlo_sharding().IsReplicated();
  }

  pybind11::tuple devices_;
  xla::HloSharding hlo_sharding_;
  pybind11::object memory_kind_;
  std::optional<size_t> hash_;
  std::shared_ptr<PyDeviceList> internal_device_list_;
};

void RegisterSharding(pybind11::module& m);

}  // namespace jax

#endif  // XLA_PYTHON_SHARDING_H_
