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

#include <cstddef>
#include <optional>
#include <utility>

// placeholder for index annotation headers
#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_numpy.h"
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

  static int SafeNumDevices(nanobind::handle sharding);

 private:
  std::optional<int> num_devices_;
};

// Checks if the memory kind is valid, and canonicalizes the
// memory kind to default memory on backends that support memories.
nanobind::object CheckAndCanonicalizeMemoryKind(
    nanobind::object memory_kind,
    const xla::nb_class_ptr<PyDeviceList>& device_list);

// Returns a hash that may sometimes return different hashes for equal values.
// It is not a correct implementation of `__hash__` in python, but it's fine
// for jit/pjit dispatch since it only causes spurious cache misses.
size_t ShardingHash(nanobind::handle sharding);

bool ShardingEqual(nanobind::handle a, nanobind::handle b);

class NamedSharding : public Sharding {
 public:
  NamedSharding(nanobind::object mesh, nanobind::object spec,
                nanobind::object memory_kind, nanobind::object parsed_pspec,
                nanobind::object manual_axes,
                nanobind::object logical_device_ids);

  const nanobind::object& mesh() const { return mesh_; }
  const nanobind::object& spec() const { return spec_; }
  const nanobind::object& memory_kind() const { return memory_kind_; }
  const nanobind::object& parsed_pspec() const { return parsed_pspec_; }
  const nanobind::object& manual_axes() const { return manual_axes_; }
  const nanobind::object& logical_device_ids() const {
    return logical_device_ids_;
  }
  void set_parsed_pspec(nanobind::object parsed_pspec) {
    parsed_pspec_ = std::move(parsed_pspec);
  }

  static nanobind::handle type() {
    static auto type = nanobind::type<NamedSharding>();
    return type;
  }

  absl::StatusOr<xla::nb_class_ptr<PyDeviceList>> internal_device_list() const {
    if (internal_device_list_) {
      return *internal_device_list_;
    }
    return xla::InvalidArgument(
        "internal_device_list is not implemented for "
        "`jax.sharding.AbstractMesh`");
  }

 private:
  nanobind::object mesh_;
  nanobind::object spec_;
  nanobind::object memory_kind_;
  nanobind::object parsed_pspec_;
  nanobind::object manual_axes_;
  nanobind::object logical_device_ids_;
  std::optional<xla::nb_class_ptr<PyDeviceList>> internal_device_list_;
};

class SingleDeviceSharding : public Sharding {
 public:
  explicit SingleDeviceSharding(
      nanobind::object device, nanobind::object memory_kind = nanobind::none());

  // Used only in C++ to accelerate `PyArray::MakeFromSingleDeviceArray()`.
  SingleDeviceSharding(xla::nb_class_ptr<xla::PyClient> client,
                       tsl::RCReference<xla::ifrt::DeviceList> device_list,
                       nanobind::object memory_kind);

  const nanobind::object& device() const { return device_; }
  const nanobind::object& memory_kind() const { return memory_kind_; }

  static nanobind::handle type() {
    static auto type = nanobind::type<SingleDeviceSharding>();
    return type;
  }

  xla::nb_class_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  nanobind::object device_;
  nanobind::object memory_kind_;
  xla::nb_class_ptr<PyDeviceList> internal_device_list_;
};

// The C++ implementation of jax.PmapSharding in python. It contains a few key
// data members and methods that are performance-critical.
class PmapSharding : public Sharding {
 public:
  PmapSharding(xla::nb_numpy_ndarray devices, ShardingSpec sharding_spec);

  ~PmapSharding() override = default;

  xla::nb_numpy_ndarray devices() const { return devices_; }

  const ShardingSpec& sharding_spec() const { return sharding_spec_; }

  static nanobind::handle type() {
    static auto type = nanobind::type<PmapSharding>();
    return type;
  }

  xla::nb_class_ptr<PyDeviceList> internal_device_list() const {
    return internal_device_list_;
  }

 private:
  xla::nb_numpy_ndarray devices_;
  ShardingSpec sharding_spec_;
  xla::nb_class_ptr<PyDeviceList> internal_device_list_;
};

class GSPMDSharding : public Sharding {
 public:
  GSPMDSharding(nanobind::sequence devices, xla::OpSharding op_sharding,
                nanobind::object memory_kind, nanobind::object device_list)
      : GSPMDSharding(
            std::move(devices),
            xla::ValueOrThrow(xla::HloSharding::FromProto(op_sharding)),
            std::move(memory_kind), std::move(device_list)) {}

  GSPMDSharding(nanobind::sequence devices, xla::HloSharding op_sharding,
                nanobind::object memory_kind, nanobind::object device_list);

  const nanobind::tuple& devices() const { return devices_; }
  const nanobind::object& memory_kind() const { return memory_kind_; }

  size_t Hash() {
    if (!hash_.has_value()) {
      hash_ = CalculateHash();
    }
    return *hash_;
  }

  static nanobind::handle type() {
    static auto type = nanobind::type<GSPMDSharding>();
    return type;
  }

  const xla::HloSharding& hlo_sharding() const { return hlo_sharding_; }

  bool operator==(const GSPMDSharding& other) const {
    return AreOpShardingsEqual(*this, other) &&
           this->devices().equal(other.devices()) &&
           this->memory_kind().equal(other.memory_kind());
  }

  xla::nb_class_ptr<PyDeviceList> internal_device_list() const {
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

  nanobind::tuple devices_;
  xla::HloSharding hlo_sharding_;
  nanobind::object memory_kind_;
  std::optional<size_t> hash_;
  xla::nb_class_ptr<PyDeviceList> internal_device_list_;
};

void RegisterSharding(nanobind::module_& m);

}  // namespace jax

#endif  // XLA_PYTHON_SHARDING_H_
