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

#ifndef TENSORFLOW_DTENSOR_CC_TENSOR_LAYOUT_H_
#define TENSORFLOW_DTENSOR_CC_TENSOR_LAYOUT_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

// Definitions for DTensor mesh & layout.
//
// A mesh describes how a set of devices is partitioned.
// A layout describes how a distributed tensor is partitioned across a mesh (and
// thus across devices). Defining tensor layouts in terms of mesh dimensions
// allows us to efficiently determine the communication required when computing
// an operation with tensors of different layouts.
namespace tensorflow {
namespace dtensor {

// Returns true if `size` is a dynamic size based on either MLIR and TF
// standards.
bool IsDynamicSize(int64_t size);

// The location of a device in a mesh.
//
// Each device has a unique location in the mesh, which is indicated by the
// offset in each mesh dimension. e.g. a mesh:
//
// [x:4, y:3, z:2]
//
// Must consist of 24 devices placed densely into the corresponding 3D space.
using DeviceLocation = absl::InlinedVector<int64, 4>;

// A shard refers to a partition of a tensor. Shards are arranged in
// ShardVectors that contains a list of Shards and a list of integers
// representing the number of shards in each dimension.
//
// Example: layout = sharding_specs:x,y, mesh:|x=2,y=2|. This can be represented
// with a ShardVector:
//          - shards = (1,1), (1,2), (2,1), (2,2)
//          - num_shards_per_dim = (2,2).
//
// The number of elements in each shard matches the tensor rank.
using Shard = std::vector<int>;

struct ShardVector {
  bool operator==(const ShardVector& other) const;
  bool operator!=(const ShardVector& other) const { return !(*this == other); }
  std::string ToString() const;

  bool ContainsShard(const Shard& shard) const;

  std::vector<Shard> shards;
  std::vector<int> num_shards_per_dim;
};

struct MeshDimension {
  MeshDimension(const std::string& name, int64 size)
      : name(std::move(name)), size(size) {}
  MeshDimension() = default;

  std::string name;
  int64 size;
};

class Mesh {
 public:
  // Failed serialized strings are represented with an empty string, therefore
  // we use this string representation of an empty mesh instead to avoid
  // confusion.
  static constexpr const char* kEmptyMeshString = "empty_mesh";
  static constexpr const char* kUseXLASPMDString = "use_xla_spmd";
  static constexpr bool kUseXLASPMD = false;
  enum class MeshType {
    kTile,
    kSingleDevice,
  };

  static Mesh Empty();
  bool IsEmpty() const;
  Mesh() { mesh_type_ = MeshType::kTile; }

  inline bool IsTile() const { return mesh_type_ == MeshType::kTile; }
  inline bool IsSingleDevice() const {
    return mesh_type_ == MeshType::kSingleDevice;
  }

  // Creates fully defined mesh.
  //
  // When `use_xla_spmd` is true, all ops running on this mesh will use XLA SPMD
  // instead of DTensor SPMD.
  static Mesh CreateMesh(const std::string& mesh_name,
                         const std::vector<std::string>& dim_names,
                         const std::vector<std::int64_t>& mesh_shape,
                         const std::vector<std::int64_t>& global_device_ids,
                         const std::vector<std::string>& global_devices_str,
                         const std::vector<std::int64_t>& local_device_ids,
                         const std::vector<std::string>& local_devices_str,
                         bool use_xla_spmd = Mesh::kUseXLASPMD);

  // Parses from MeshProto.
  static StatusOr<Mesh> ParseFromProto(const MeshProto& proto);
  // Parses from a human readable string version of the mesh, currently used
  // to represent meshes in MLIR:
  //  mesh = <name|List[MeshDim]|List[GlobalId]|List[LocalId]|List[Devices]>
  //
  // Example:
  //  mesh =
  //  <name|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  static StatusOr<Mesh> FromString(absl::string_view str);
  std::string ToString() const;
  StatusOr<MeshProto> ToProto() const;

  // Creates mesh without specific devices associated to it (aka abstract mesh).
  // This is an experimental API. Use only if strictly needed.
  static StatusOr<Mesh> GetAbstractMesh(
      const std::string& name, const std::vector<MeshDimension>& mesh_dims);
  // Creates fully defined mesh.
  static StatusOr<Mesh> GetMesh(
      const std::string& name, const std::vector<MeshDimension>& mesh_dims,
      const std::vector<std::int64_t>& global_device_ids,
      const std::vector<std::int64_t>& local_device_ids,
      const std::vector<std::string>& local_devices,
      const std::vector<std::string>& global_devices,
      bool use_xla_spmd = Mesh::kUseXLASPMD);
  static StatusOr<Mesh> GetSingleDeviceMesh(absl::string_view single_device);

  bool is_cpu_mesh() const { return device_type() == "CPU"; }
  bool is_epu_mesh() const { return device_type() == "EPU"; }
  bool is_tpu_mesh() const { return device_type() == "TPU"; }
  // Returns whether the mesh is a remote mesh.
  bool is_remote() const {
    return local_device_ids_.empty() && !global_device_ids_.empty();
  }

  // Device information methods.
  std::string device_type() const;
  // Takes an index in the flattened list of devices and returns a location
  // in the mesh.
  StatusOr<const DeviceLocation> device_location(int offset) const;
  int64 num_devices() const;
  absl::Span<const std::string> local_devices() const { return local_devices_; }
  absl::Span<const int64_t> local_device_ids() const {
    return local_device_ids_;
  }
  // Parses names of local_devices according to TF's Device Name Utils.
  StatusOr<const std::vector<DeviceNameUtils::ParsedName>> ParsedDevices()
      const;
  // Convert to given device type.
  StatusOr<Mesh> ToDeviceType(const std::string& device_type) const;
  std::vector<std::string> hosts() const;

  // Consumes a location in the mesh and returns its corresponding index in
  // the flattened list of devices.
  int64 GetFlattenedCoordinate(const DeviceLocation& loc) const;

  const MeshDimension& dim(int64 index) const { return mesh_dims_[index]; }
  std::vector<MeshDimension> dims() const { return mesh_dims_; }
  // Returns size of mesh dimension.
  StatusOr<int64> dim_size(absl::string_view name) const;
  // Returns list of mesh dimension sizes.
  std::vector<int64> dim_sizes() const;
  const std::string& dim_name(int64 index) const {
    return mesh_dims_[index].name;
  }
  int64_t min_global_device_id() const {
    DCHECK(!global_device_ids_.empty());
    return *std::min_element(global_device_ids_.begin(),
                             global_device_ids_.end());
  }
  int64_t num_local_devices() const { return local_devices_.size(); }

  absl::Span<const int64_t> global_device_ids() const {
    return global_device_ids_;
  }

  const std::vector<std::string>& global_devices() const {
    return global_devices_;
  }
  // Returns index of given dim_name in the mesh.
  StatusOr<int32> idx_for_dim(absl::string_view dim_name) const;

  // Returns the index of MeshDimension in mesh where the mesh dimension name is
  // `mesh_name`.
  int GetMeshDimIndexWithName(const std::string& mesh_name) const;
  bool IsMeshDim(const std::string& dim_name) const;
  std::vector<std::string> MeshDimNames() const;

  int64 rank() const;
  int64 size() const;
  bool use_xla_spmd() const { return use_xla_spmd_; }
  const std::string& name() const { return name_; }
  absl::string_view single_device() const { return single_device_; }

  // Global unique fingerprint. Same on different workers.
  uint64 GlobalFingerprint() const;

  // Uses proto to compare the equality. If any conversion to proto fails,
  // returns false.
  bool operator==(const Mesh& b) const;
  bool operator!=(const Mesh& b) const { return !((*this) == b); }
  bool operator<(const Mesh& b) const {
    return this->ToString() < b.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const Mesh& m) {
    return H::combine(std::move(h), m.ToString());
  }

  // A map from mesh names to their corresponding core ID mappings. The core ID
  // mapping is stored as a vector. The i-th element in the vector is the ID of
  // the core represented by global device ID of i in this mesh.
  //
  // The entry stored under the empty name key (the so-called "default mapping"
  // in some comments) is special. It is always set at the end of TPU
  // initialization. It represents the mapping for any mesh whose global device
  // IDs follow TF task-device ordinals. Legacy and test meshes created without
  // using the `create_tpu_mesh` helper follow that rule and can use this entry.
  static std::map<std::string, std::vector<int>>& tpu_core_ids();

  // The host mesh associated with any user-defined TPU mesh.
  static std::string& tpu_host_mesh();

 private:
  MeshType mesh_type_;
  std::string name_;
  // The following fields store the information for tile sharding. Usable only
  // when the mesh has type `kTile`.
  std::vector<MeshDimension> mesh_dims_;
  std::vector<std::string> local_devices_;
  std::vector<int64_t> local_device_ids_;
  std::vector<int64_t> global_device_ids_;
  std::vector<std::string> global_devices_;
  bool use_xla_spmd_ = Mesh::kUseXLASPMD;

  // Stores the device when mesh is used for representing the state of a tensor
  // on one device. Usable only when the mesh has type `kSingleDevice`.
  std::string single_device_;
};

// Obtain all possible forms of indexing a mesh.
//
// e.g. given a mesh with dimensions [x=2, y=3], returns {
//   [0, 0], [0, 1], [0, 2],
//   [1, 0], [1, 1], [1, 2]
// }
std::vector<DeviceLocation> ComputeDeviceLocations(const Mesh& mesh);

class Layout {
 public:
  static constexpr const char* kUnshardedDim = "unsharded";
  // This spec should only be used to express no preferred sharding in the
  // Layout propagation algorithm.
  static constexpr const char* kAny = "any";
  // Failed serialized strings are represented with an empty string, therefore
  // we use this string representation of an empty layout instead to avoid
  // confusion.
  static constexpr const char* kEmptyLayoutString = "empty_layout";
  // Used for the relayout operation, to allow relayout act as an identity on
  // the layout for the given dimension.
  static constexpr const char* kMatch = "match";

  inline bool IsSingleDevice() const { return mesh_.IsSingleDevice(); }

  // Returns empty layout.
  static Layout Empty();

  // Parses from LayoutProto.
  static StatusOr<Layout> FromProto(const LayoutProto& proto);
  // Parses from a human readable string version of the layout, currently used
  // to represent layouts in MLIR:
  //  layout = <sharding_specs:List[specs] mesh:name|List[MeshDim]|
  //  List[GlobalId]|List[LocalId]|List[Devices]>
  //
  // Example:
  //  layout = <sharding_specs:x,not_sharded mesh:name|x=2,y=2|0,1,2,3|0,1,2,3|
  //  /job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,
  //  /job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  static StatusOr<Layout> FromString(absl::string_view layout_str);
  // Creates human readable string version of a layout.
  std::string ToString() const;
  StatusOr<LayoutProto> ToProto() const;

  const Mesh& mesh() const { return mesh_; }
  static Layout ReplicatedOnMesh(const Mesh& mesh, int rank);
  static Layout BatchShardedOnMesh(const Mesh& mesh, int rank,
                                   const string& mesh_dim, int axis = 0);
  static Layout ReplicatedLike(const Layout& layout);
  static Layout BatchShardedLike(const Layout& layout, const string& mesh_dim,
                                 int axis = 0);
  static Layout AnyOnMesh(const Mesh& mesh, int rank);
  // Creates a mesh of unique shards.
  Mesh ReducedMesh() const;
  void set_mesh(Mesh mesh) { mesh_ = mesh; }

  // Returns a layout for the transposed matrix for given layout. This assumes
  // that only the last two dimensions are used for matrix computation and all
  // dimensions before are batch dimensions.
  static StatusOr<Layout> Transposed2D(const Layout& layout);
  static bool IsUnshardedDimension(const absl::string_view name) {
    return name == kUnshardedDim;
  }
  static bool IsShardedDimension(const absl::string_view name) {
    return !IsUnshardedDimension(name);
  }
  static bool IsUnshardedSpec(const ShardingSpec& spec) {
    return IsUnshardedDimension(spec.sharding_spec());
  }
  static bool IsShardedSpec(const ShardingSpec& spec) {
    return !IsUnshardedDimension(spec.sharding_spec());
  }
  static StatusOr<Layout> GetLayout(
      const std::vector<std::string>& sharding_spec_strs, const Mesh& mesh);
  static StatusOr<Layout> GetLayout(
      const std::vector<ShardingSpec>& sharding_specs, const Mesh& mesh);
  static StatusOr<Layout> GetSingleDeviceLayout(const Mesh& mesh);

  // Makes a new layout from this one dropping the given dimensions.
  // If keep_dims is true, the dimensions are replicated rather than
  // deleted.
  StatusOr<Layout> GetLayoutWithReducedDims(
      const absl::flat_hash_set<int>& reduced_dims, bool keep_dims) const;

  // Truncates a layout at the front or back, depending on the value of end.
  // end = false returns the layout up to the split point,
  // end = true returns the layout from the split point.
  StatusOr<Layout> Truncate(int64 split_point, bool end = false) const;

  // Left or right pad the layout to a max rank.
  Layout LeftPad(int64 rank) const;

  bool IsFullyReplicated() const;
  bool IsLastDimReplicated() const;
  // Checks that the last N-1 dimensions are replicated
  bool IsBatchParallel() const;
  // Checks that the dimensions from [-non_batch_rank, end) are replicated.
  bool IsBatchParallel(int non_batch_rank) const;
  bool IsEmpty() const;

  // Compute global shape using the layout and provided local_shape.
  std::vector<int64_t> GlobalShapeFromLocalShape(
      absl::Span<const int64_t> local_shape) const;

  std::vector<int64_t> LocalShapeFromGlobalShape(
      absl::Span<const int64_t> global_shape) const;
  PartialTensorShape LocalShapeFromGlobalShape(
      const PartialTensorShape& global_shape) const;

  int64 rank() const { return sharding_specs_.size(); }
  size_t num_shards_for_dim(const ShardingSpec& dim) const;
  size_t num_shards_for_dim(int) const;
  std::vector<int32> num_shards() const;

  const ShardingSpec& dim(int64 idx) const { return sharding_specs_[idx]; }
  absl::Span<const ShardingSpec> sharding_specs() const {
    return sharding_specs_;
  }

  // Computes the corresponding shard vector to this layout.
  ShardVector GetShardVector() const;

  // Returns sharding specs in string form.
  std::vector<std::string> sharding_spec_strs() const;

  int64 num_devices() const { return mesh_.num_devices(); }

  // Map hosts to shards.
  std::map<std::string, ShardVector> HostShardMap() const;

  const std::string& sharding_spec(int idx) const;

  // Two layouts are equivalent if they would result in the same sharding for
  // the tensor. E.g. if one is unsharded and the other is sharded on a mesh
  // dimension of size 1.
  bool IsEquivalent(const Layout& b) const;
  // Uses proto to compare the equality. If any conversion to proto fails,
  // returns false.
  bool operator==(const Layout& b) const;
  bool operator!=(const Layout& b) const { return !((*this) == b); }
  bool operator<(const Layout& b) const {
    return this->ToString() < b.ToString();
  }

 private:
  std::vector<ShardingSpec> sharding_specs_;
  Mesh mesh_;
};

// Takes two layouts and concatenates their TensorDimensions. If the meshes for
// the two layouts are different or both layouts are using the same mesh
// dimension returns an error rather than a layout.
StatusOr<Layout> ConcatenateLayouts(const Layout& layout_a,
                                    const Layout& layout_b);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_TENSOR_LAYOUT_H_
