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

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/sharded_device_array.h"

namespace jax {

class Sharding {
 public:
  virtual ~Sharding() = default;
};

class XLACompatibleSharding : public Sharding {
 public:
  ~XLACompatibleSharding() override = default;
};

// The C++ implementation of jax.PmapSharding in python. It contains a few key
// data members and methods that are performance-critical.
class PmapSharding : public XLACompatibleSharding {
 public:
  PmapSharding(pybind11::object devices, ShardingSpec sharding_spec)
      : devices_(std::move(devices)),
        sharding_spec_(std::move(sharding_spec)) {}

  ~PmapSharding() override = default;

  pybind11::object devices() const { return devices_; }

  const ShardingSpec& sharding_spec() const { return sharding_spec_; }

  static pybind11::handle type() {
    static auto type = pybind11::type::handle_of<PmapSharding>();
    return type;
  }

 private:
  pybind11::object devices_;
  ShardingSpec sharding_spec_;
};

void RegisterSharding(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARDING_H_
