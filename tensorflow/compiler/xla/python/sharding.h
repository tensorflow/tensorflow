/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_SHARDING_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_SHARDING_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/sharded_device_array.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace jax {

class Sharding {
 public:
  Sharding() = default;

  // This constructor is used in the fast path to retrieve the number of devices
  // without falling back to python. This is only used in the cpp path.
  explicit Sharding(int num_devices) : num_devices_(num_devices) {}

  virtual ~Sharding() = default;

  int num_devices() const {
    if (num_devices_.has_value()) {
      return *num_devices_;
    }

    auto self = pybind11::cast(this);
    pybind11::set device_set = self.attr("device_set");
    return device_set.size();
  }

 private:
  std::optional<int> num_devices_;
};

// Returns a hash that may sometimes return different hashes for equal values.
// It is not a correct implementation of `__hash__` in python, but it's fine
// for jit/pjit dispatch since it only causes spurious cache misses.
size_t ShardingHash(const pybind11::object& obj);

bool ShardingEqual(const pybind11::object& a, const pybind11::object& b);

class XLACompatibleSharding : public Sharding {
 public:
  using Sharding::Sharding;

  ~XLACompatibleSharding() override = default;
};

class NamedSharding : public XLACompatibleSharding {
 public:
  NamedSharding(pybind11::object mesh, pybind11::object spec,
                pybind11::object parsed_pspec);

  const pybind11::object& mesh() const { return mesh_; }
  const pybind11::object& spec() const { return spec_; }
  const pybind11::object& parsed_pspec() const { return parsed_pspec_; }
  void set_parsed_pspec(pybind11::object parsed_pspec) {
    parsed_pspec_ = std::move(parsed_pspec);
  }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<NamedSharding>();
    return type;
  }

 private:
  pybind11::object mesh_;
  pybind11::object spec_;
  pybind11::object parsed_pspec_;
};

class SingleDeviceSharding : public XLACompatibleSharding {
 public:
  explicit SingleDeviceSharding(pybind11::object device)
      : XLACompatibleSharding(/*num_devices=*/1), device_(std::move(device)) {}

  const pybind11::object& device() const { return device_; }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<SingleDeviceSharding>();
    return type;
  }

 private:
  pybind11::object device_;
};

// The C++ implementation of jax.PmapSharding in python. It contains a few key
// data members and methods that are performance-critical.
class PmapSharding : public XLACompatibleSharding {
 public:
  PmapSharding(pybind11::array devices, ShardingSpec sharding_spec)
      : XLACompatibleSharding(/*num_devices=*/devices.size()),
        devices_(std::move(devices)),
        sharding_spec_(std::move(sharding_spec)) {}

  ~PmapSharding() override = default;

  pybind11::array devices() const { return devices_; }

  const ShardingSpec& sharding_spec() const { return sharding_spec_; }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<PmapSharding>();
    return type;
  }

 private:
  pybind11::array devices_;
  ShardingSpec sharding_spec_;
};

class GSPMDSharding : public XLACompatibleSharding {
 public:
  GSPMDSharding(pybind11::list devices, xla::OpSharding op_sharding)
      : XLACompatibleSharding(/*num_devices=*/devices.size()),
        devices_(std::move(devices)),  // Implicitly converts a list to a tuple.
        op_sharding_(std::move(op_sharding)) {}

  GSPMDSharding(pybind11::tuple devices, xla::OpSharding op_sharding)
      : XLACompatibleSharding(/*num_devices=*/devices.size()),
        devices_(std::move(devices)),
        op_sharding_(std::move(op_sharding)) {}

  const pybind11::tuple& devices() const { return devices_; }
  const xla::OpSharding& op_sharding() const { return op_sharding_; }

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

  xla::HloSharding hlo_sharding() const {
    auto hlo_sharding = xla::HloSharding::FromProto(op_sharding_);
    if (!hlo_sharding.ok()) {
      throw xla::XlaRuntimeError(hlo_sharding.status().error_message());
    }
    return hlo_sharding.value();
  }

  bool operator==(const GSPMDSharding& other) const {
    return AreOpShardingsEqual(*this, other) &&
           this->devices().equal(other.devices());
  }

 private:
  size_t CalculateHash() const {
    // We only hash `op_sharding_` here for performance.
    auto hlo_sharding = xla::HloSharding::FromProto(op_sharding_);
    if (!hlo_sharding.ok()) {
      throw xla::XlaRuntimeError(hlo_sharding.status().error_message());
    }
    return absl::Hash<xla::HloSharding>()(*hlo_sharding);
  }

  bool IsOpShardingReplicated() const {
    if (op_sharding_.tile_assignment_devices().size() == 1) {
      return true;
    } else {
      return hlo_sharding().IsReplicated();
    }
  }

  static bool AreOpShardingsEqual(const GSPMDSharding& a,
                                  const GSPMDSharding& b) {
    // If the OpSharding object is the same, return true
    if (&a.op_sharding() == &b.op_sharding()) {
      return true;
    }
    // If both OpShardings are replicated, return true
    if (a.IsOpShardingReplicated() && b.IsOpShardingReplicated()) {
      return true;
    }
    return a.hlo_sharding() == b.hlo_sharding();
  }

  pybind11::tuple devices_;
  xla::OpSharding op_sharding_;

  std::optional<size_t> hash_;
};

void RegisterSharding(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARDING_H_
