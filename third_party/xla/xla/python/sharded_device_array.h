/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
#define XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_

#include <utility>
#include <variant>
#include <vector>

#include "absl/types/variant.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "xla/python/types.h"

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

template <typename H>
H AbslHashValue(H h, const NoSharding& key) {
  return h;
}

// `Chunked` means that the dimension is split into np.prod(chunks) chunks
// and the split dimension itself is preserved inside the map.
// Those chunks are distributed over `len(chunks)` ShardedAxes axes
// (major-to-minor).
// For example, for a tensor `t` of shape [N] sharded using [Chunked([p])] (with
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

template <typename H>
H AbslHashValue(H h, const Chunked& key) {
  h = H::combine(std::move(h), key.chunks);
  return h;
}

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

template <typename H>
H AbslHashValue(H h, const Unstacked& key) {
  h = H::combine(std::move(h), key.size);
  return h;
}

using AvalDimSharding = std::variant<NoSharding, Chunked, Unstacked>;

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

template <typename H>
H AbslHashValue(H h, const ShardedAxis& key) {
  h = H::combine(std::move(h), key.axis);
  return h;
}

struct Replicated {
  int replicas;
  bool operator==(const Replicated& other) const {
    return replicas == other.replicas;
  }
  bool operator!=(const Replicated& other) const {
    return replicas != other.replicas;
  }
};

template <typename H>
H AbslHashValue(H h, const Replicated& key) {
  h = H::combine(std::move(h), key.replicas);
  return h;
}

using MeshDimAssignment = std::variant<ShardedAxis, Replicated>;

// Describes how each axis is sharded (if it is), and how it's mapped to the
// devices mesh. See Jax pxla.py for the documentation.
//
// ShardingSpec is shared across pmap, pjit and xpmap. For pmap, an input
// `sharding`  is composed of `NoSharding` and at most one `Unstacked`.
// If `axis_size=None`, at least one the inputs has a dimension associated to
// `Unstacked`.
//
// Examples:
//
// 1. For pmap, with a tensor of shape [8, 2, 2], to unstack along the first
//    dimension into [8] devices:
//
//    sharding = [Unstacked(8), NoSharding, NoSharding]
//    mesh_mapping = [ShardedAxis(0)]
//
// 2. With an input array of shape [6], that we want to chunk into [2, 3]
//    Assuming an device mesh [3, 4, 2] of devices, we will have:
//
//    sharding = [Chunked([2, 3])]
//    mesh_mapping = [ShardedAxis(1), Replicated, ShardedAxis(0)]
//
//    In particular, in the above example, the ShardedAxis refers to indices
//    of the sharded shape [2, 3]. (only the `Chunked` sharding can produce more
//    than one dimension).
class ShardingSpec {
 public:
  ShardingSpec(std::vector<AvalDimSharding> sharding,
               std::vector<MeshDimAssignment> mesh_mapping)
      : sharding_(std::move(sharding)),
        mesh_mapping_(std::move(mesh_mapping)) {}
  ShardingSpec(nanobind::iterable py_sharding,
               nanobind::iterable py_mesh_mapping)
      : sharding_(xla::IterableToVector<AvalDimSharding>(py_sharding)),
        mesh_mapping_(
            xla::IterableToVector<MeshDimAssignment>(py_mesh_mapping)) {}

  const std::vector<AvalDimSharding>& GetSharding() const { return sharding_; }
  const std::vector<MeshDimAssignment>& GetMeshMapping() const {
    return mesh_mapping_;
  }

  bool operator==(const ShardingSpec& other) const {
    return sharding_ == other.sharding_ && mesh_mapping_ == other.mesh_mapping_;
  }

  bool operator!=(const ShardingSpec& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const ShardingSpec& key);

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

template <typename H>
H AbslHashValue(H h, const ShardingSpec& key) {
  h = H::combine(std::move(h), key.sharding_);
  h = H::combine(std::move(h), key.mesh_mapping_);
  return h;
}

}  // namespace jax

#endif  // XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
