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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/variant.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/types.h"

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
  ShardingSpec(pybind11::iterable py_sharding,
               pybind11::iterable py_mesh_mapping)
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
class ShardedDeviceArray {
 public:
  ShardedDeviceArray(const ShardedDeviceArray&) = delete;
  ShardedDeviceArray& operator=(const ShardedDeviceArray&) = delete;
  ShardedDeviceArray(ShardedDeviceArray&&) = default;
  ShardedDeviceArray& operator=(ShardedDeviceArray&&) = default;

  // Delete all the underlying buffers (freeing memory on device).
  // The Numpy value on the host, if it exists, will also be deleted.
  void Delete();
  const ShardingSpec& GetShardingSpec() const { return sharding_spec_; }

  // Returns an error status iff the object has been deleted.
  xla::StatusOr<xla::ifrt::Array*> ifrt_array();

  // Returns an error status iff the object has been deleted.
  xla::StatusOr<absl::Span<xla::PjRtBuffer* const>> pjrt_buffers();

  bool is_deleted() const { return is_deleted_; }
  bool weak_type() const { return weak_type_; }
  std::optional<pybind11::list> device_buffers() const {
    return device_buffers_;
  }
  pybind11::object aval() const { return aval_; }
  pybind11::object indices() const { return indices_; }

  std::optional<pybind11::object> npy_value() const { return npy_value_; }
  void set_npy_value(pybind11::object npy_value) { npy_value_ = npy_value; }

  std::optional<pybind11::object> one_replica_buffer_indices() const {
    return one_replica_buffer_indices_;
  }
  void set_one_replica_buffer_indices(pybind11::object obj) {
    one_replica_buffer_indices_ = obj;
  }

  // Python-wrapper definitions.

  // pybind11::object typed subclass for PyBuffer objects.
  class pyobject : public pybind11::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    pybind11::object, ShardedDeviceArray::IsShardedDeviceArray);
    pyobject() = default;
    ShardedDeviceArray* sda() const {
      return ShardedDeviceArray::AsShardedDeviceArrayUnchecked(*this);
    }
  };
  using object = pyobject;

  // Returns true if `handle` is a IsShardedDeviceArray.
  static bool IsShardedDeviceArray(pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Does not do any checking.
  static ShardedDeviceArray* AsShardedDeviceArrayUnchecked(
      pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Returns an error status if
  // !IsPyBuffer(handle)
  static xla::StatusOr<ShardedDeviceArray*> AsShardedDeviceArray(
      pybind11::handle handle);

  // Gets a Python handle to an existing ShardedDeviceArray. Assumes the
  // PyObject was allocated on the Python heap, which is the case if Make() was
  // used.
  pybind11::handle AsHandle();

  static object Make(pybind11::object aval, ShardingSpec sharding_spec,
                     pybind11::list device_buffers, pybind11::object indices,
                     bool weak_type);

  static xla::Status RegisterTypes(pybind11::module& m);
  static PyObject* base_type() { return base_type_; }
  static PyObject* type() { return type_; }

 private:
  // Buffers are expected to be xla::PyBuffer objects, but as there are
  // alternative backend implementations, this may not be guaranteed.
  // TODO(jblespiau): As soon as PjRtBuffer is supported by all
  // implementations, we should be able to store this with the C++ objects.
  ShardedDeviceArray(pybind11::object aval, ShardingSpec sharding_spec,
                     pybind11::list device_buffers, pybind11::object indices,
                     bool weak_type)
      : aval_(std::move(aval)),
        sharding_spec_(std::move(sharding_spec)),
        indices_(std::move(indices)),
        device_buffers_(std::move(device_buffers)),
        weak_type_(weak_type) {}
  static PyObject* base_type_;
  static PyObject* type_;

  // A ShapedArray indicating the shape and dtype of this array.
  pybind11::object aval_;
  // Describes how this array is sharded across `device_buffers`.
  ShardingSpec sharding_spec_;
  // The `indices` used to slice numpy array into the underlying list of
  // buffers. See the Python pxla.py:spec_to_indices function.
  pybind11::object indices_;
  // The buffers containing the data for this array. Each buffer is the same
  // shape and on a different device. Buffers are in row-major order, with
  // replication treated as an extra innermost dimension.
  std::optional<pybind11::list> device_buffers_;

  std::optional<pybind11::object> npy_value_ = std::nullopt;
  std::optional<pybind11::object> one_replica_buffer_indices_ = std::nullopt;

  std::optional<tsl::RCReference<xla::ifrt::Array>> ifrt_array_ = std::nullopt;

  // The device_buffers as a C++ object. As this is what we consume from C++
  // and this is also what we generate from C++, cache the result so that
  // we don't have to perform casts.
  // TODO(jblespiau): Make this the default, and have `device_buffers_` the
  // the optional Python value if it's accessed from Python.
  std::optional<std::vector<xla::PjRtBuffer*>> cpp_device_buffers_ =
      std::nullopt;

  // The weak_type to prevent accessing the "aval_.weak_type" attribute which
  // is significantly slower.
  bool weak_type_;
  bool is_deleted_ = false;
};

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_SHARDED_DEVICE_ARRAY_H_
