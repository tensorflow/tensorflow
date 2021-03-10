/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PMAP_LIB_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PMAP_LIB_H_

#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/core/platform/logging.h"

// TODO(jblespiau): The current implementation moves the Python logic to C++,
// as a preliminary step to executing the `pmap` execution path from C++.
// It implements the current Python behavior (thus, it may not be optimal, and
// we will be able to modify it later).

namespace jax {

// High level introduction.
//
// pmap and other parallel computation functions distribute some computation on
// several devices. On December 2020, the devices mesh (i.e. N-dimentional array
// of devices on which we map the computation) is defined by the user.
//
// We describe how to shard the inputs, and how to map it to the mesh of devices
// using `ShardingSpec`. It's mainly based on 2 components:
// - `sharding`, which specifies how to shard the inputs.
// - `mesh_mapping`, which specifies how to map shards to devices.
//

// The 3 following structs define how to shard one dimension of an ndarry.
//
// `NoSharding` (`None` in Python) means no sharding.
struct NoSharding {
  bool operator==(const NoSharding& other) const { return true; }
  bool operator!=(const NoSharding& other) const { return false; }
};

// `Chunked` means that the dimension is split into np.prod(chunks) chunks
// and the split dimension itself is preserved inside the map.
// Those chunks are distributed over `len(chunks)` ShardedAxes axes
// (major-to-minor).
// For example, for a tensor `t` or shape [N] sharded using [Chunked([p])] (with
// p  dividing N, let S = N // p) the tensor will be split into p chunks of
// shape [S], such sharded_t[k] = t[k * S: (k+1)*S] (left included, right
// excluded) for k in {0, ... p-1}.
struct Chunked {
 public:
  explicit Chunked(std::vector<int> chunks_) : chunks(std::move(chunks_)) {}
  // The number of chunks per axis.
  std::vector<int> chunks;

  bool operator==(const Chunked& other) const { return chunks == other.chunks; }
  bool operator!=(const Chunked& other) const { return chunks != other.chunks; }
};

// `Unstacked` means that the dimension is split into chunks of size 1, and
// doesn't appear inside the map. `size` is always the dimension size.
// For example, a Tensor t of shape [N] will be sharded into N tensors of shape
// [], when using `Unstacked(N)`.
struct Unstacked {
 public:
  explicit Unstacked(int sz) : size(sz) {}
  int size;

  bool operator==(const Unstacked& other) const { return size == other.size; }
  bool operator!=(const Unstacked& other) const { return size != other.size; }
};

using AvalDimSharding = absl::variant<NoSharding, Chunked, Unstacked>;

// Assigns sharded axes to mesh dimensions.
//
// The devices will be for each dimension which has a sharded `AvalDimSharding`
// When no axis is assigned, the data is replicated.
// As indices are 0-indexed, `ShardedAxis(1)` refers to the second actually
// sharded axis (i.e. counting as if the None dimensions of sharding were
// filtered out).
// For example, given the sharding `[Unstacked(n), None, Chunked(m)]`, an entry
// of `ShardedAxis(1)` refers to the `Chunked(m)` axis, not the `None`.

struct ShardedAxis {
  int axis;
  bool operator==(const ShardedAxis& other) const { return axis == other.axis; }
  bool operator!=(const ShardedAxis& other) const { return axis != other.axis; }
};
struct Replicated {
  int replicas;
  bool operator==(const Replicated& other) const {
    return replicas == other.replicas;
  }
  bool operator!=(const Replicated& other) const {
    return replicas != other.replicas;
  }
};

using MeshDimAssignment = absl::variant<ShardedAxis, Replicated>;

// Describes how each axis is sharded (if it is), and how it'smapped to the
// devices mesh.
class ShardingSpec {
 public:
  ShardingSpec(std::vector<AvalDimSharding> sharding,
               std::vector<MeshDimAssignment> mesh_mapping)
      : sharding_(std::move(sharding)),
        mesh_mapping_(std::move(mesh_mapping)) {}

  const std::vector<AvalDimSharding>& GetSharding() const { return sharding_; }
  const std::vector<MeshDimAssignment>& GetMeshMapping() const {
    return mesh_mapping_;
  }

  bool operator==(const ShardingSpec& other) const {
    return sharding_ == other.sharding_ && mesh_mapping_ == other.mesh_mapping_;
  }

  bool operator!=(const ShardingSpec& other) const { return !(*this == other); }

 private:
  //  `sharding` specifies how the array is supposed to get partitioned into
  //  chunks. Its length matchs the rank of the array. See the docstring
  //  of `AvalDimSharding` for the supported partitioning schemes.
  std::vector<AvalDimSharding> sharding_;
  //  `mesh_mapping` describes an assignments of the array chunks created by
  //  `sharding` to a logical device mesh. The length of the tuple is equal to
  //  the rank of the mesh. Each mesh dimension can either get partitions of
  //  data varying along one of the sharded dimensions, or the data can be
  //  replicated.
  std::vector<MeshDimAssignment> mesh_mapping_;
};

// A ShardedDeviceArray is an ndarray sharded across devices.
//
// The purpose of a ShardedDeviceArray is to reduce the number of transfers when
// executing replicated computations, by allowing results to persist on the
// devices that produced them. That way dispatching a similarly replicated
// computation that consumes the same sharded memory layout does not incur any
// transfers.

// A ShardedDeviceArray represents one logical ndarray value, and simulates the
// behavior of an ndarray so that it can be treated by user code as an ndarray;
// that is, it is only an optimization to reduce transfers.

// Design note: We move to C++, only what will need to be accessed by C++ to
// execute a pmap computation. A large part of the logic is still in Python.
class ShardedDeviceArray : xla::DeviceArrayBase {
 public:
  ShardedDeviceArray(
      pybind11::handle aval, ShardingSpec sharding_spec,
      // Buffers are expected to be xla::PyBuffer objects, but as there are
      // alternative backend implementations, this may not be guaranteed.
      // TODO(jblespiau): As soon as PjRtBuffer is supported by all
      // implementations, we should be able to store this with the C++ objects.
      pybind11::list device_buffers)
      : DeviceArrayBase(),
        aval_(pybind11::cast<pybind11::object>(aval)),
        sharding_spec_(std::move(sharding_spec)),
        device_buffers_(device_buffers) {}

  pybind11::object GetAval() const { return aval_; }
  const ShardingSpec& GetShardingSpec() const { return sharding_spec_; }
  pybind11::list GetDeviceBuffers() const { return device_buffers_; }

 private:
  // A ShapedArray indicating the shape and dtype of this array.
  pybind11::object aval_;
  // Describes how this array is sharded across `device_buffers`.
  ShardingSpec sharding_spec_;
  // The buffers containing the data for this array. Each buffer is the same
  // shape and on a different device. Buffers are in row-major order, with
  // replication treated as an extra innermost dimension.
  pybind11::list device_buffers_;
};

void BuildPmapSubmodule(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PMAP_LIB_H_
