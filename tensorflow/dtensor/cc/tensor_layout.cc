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

#include "tensorflow/dtensor/cc/tensor_layout.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

constexpr const char* Layout::kUnshardedDim;
constexpr const char* Layout::kAny;
constexpr const char* Layout::kEmptyLayoutString;
constexpr const char* Layout::kMatch;
constexpr const char* Mesh::kEmptyMeshString;
constexpr const char* Mesh::kUseXLASPMDString;
constexpr bool Mesh::kUseXLASPMD;

namespace {
// Expands a ShardVector into the size defined in new_num_shards_per_dim.
//
// For example, the inputs:
//    - shard_vec: shards = [(1,1)] num_shards_per_dim = [1,1]
//    - new_num_shards_per_dim = [2,2]
//
// Would lead to:
// shard_vec: shards = [(1,1),(1,2),(2,1),(2,2)] num_shards_per_dim = [2,2]
//
// This is used to check whether two ShardVectors contain the same information
// while having different number of shards per dimension. The two ShardVectors
// above are an example of this.
ShardVector ExpandShardVector(const ShardVector& shard_vec,
                              const std::vector<int>& new_num_shards_per_dim) {
  if (shard_vec.shards.empty()) return shard_vec;

  // Takes a single shard and expands it into multiple shards.
  auto ExpandShard = [shard_vec, new_num_shards_per_dim](
                         const Shard& shard,
                         int dim_ind) -> std::vector<Shard> {
    int original_dim_size = shard_vec.num_shards_per_dim[dim_ind];
    int new_dim_size = new_num_shards_per_dim[dim_ind];
    int size_ratio = new_dim_size / original_dim_size;

    std::vector<Shard> expanded_shards;
    expanded_shards.reserve(size_ratio);
    for (int i = 0; i < size_ratio; ++i) {
      int original_coord = shard[dim_ind];
      int shifted_coord = (original_coord - 1) * size_ratio + 1 + i;
      // Copy original shard, then modify it.
      Shard new_shard = shard;
      new_shard[dim_ind] = shifted_coord;
      expanded_shards.push_back(new_shard);
    }
    return expanded_shards;
  };
  // Iterates over the dimensions of the shard, expanding at each
  // dimension.
  std::vector<Shard> total_expanded_shards = shard_vec.shards;
  for (int dim_ind = 0; dim_ind < new_num_shards_per_dim.size(); ++dim_ind) {
    std::vector<Shard> dim_expanded_shards;
    for (const auto& shard : total_expanded_shards) {
      std::vector<Shard> expanded_shards = ExpandShard(shard, dim_ind);
      // Concatenate newly created shards.
      dim_expanded_shards.insert(dim_expanded_shards.end(),
                                 expanded_shards.begin(),
                                 expanded_shards.end());
    }
    // Copy newly created shards and delete old ones.
    total_expanded_shards = dim_expanded_shards;
  }
  std::sort(total_expanded_shards.begin(), total_expanded_shards.end());
  ShardVector expanded_shard_vec;
  expanded_shard_vec.shards = total_expanded_shards;
  expanded_shard_vec.num_shards_per_dim = new_num_shards_per_dim;
  return expanded_shard_vec;
}
}  // namespace

std::vector<DeviceLocation> ComputeDeviceLocations(const Mesh& mesh) {
  std::vector<DeviceLocation> mesh_locs(mesh.size());
  for (size_t i = 0; i < mesh.size(); ++i)
    mesh_locs[i] = *(mesh.device_location(i));
  return mesh_locs;
}

bool ShardVector::operator==(const ShardVector& other) const {
  // Check same number of shards.
  if (this->shards.empty() && other.shards.empty()) return true;
  if (this->shards.empty() || other.shards.empty()) return false;

  // Check number of shard dimensions match.
  if (this->num_shards_per_dim.size() != other.num_shards_per_dim.size())
    return false;

  // Compute lowest common multiple for each of the shard dimensions.
  Shard first_shard_this = this->shards[0];
  Shard first_shard_other = other.shards[0];
  std::vector<int> new_sizes;
  for (size_t i = 0; i < first_shard_this.size(); ++i) {
    int lcm = this->num_shards_per_dim[i] * other.num_shards_per_dim[i] /
              MathUtil::GCD(static_cast<unsigned>(this->num_shards_per_dim[i]),
                            static_cast<unsigned>(other.num_shards_per_dim[i]));
    new_sizes.push_back(lcm);
  }

  // Expand and compare.
  return ExpandShardVector(*this, new_sizes).shards ==
         ExpandShardVector(other, new_sizes).shards;
}

std::string ShardVector::ToString() const {
  std::string string = "shards:[";
  // Convert each Shard into string.
  std::vector<std::string> shard_strs;
  shard_strs.reserve(shards.size());
  for (const Shard& shard : shards)
    shard_strs.push_back("(" + absl::StrJoin(shard, ",") + ")");
  // Join shards, and append dimensions.
  absl::StrAppend(&string, absl::StrJoin(shard_strs, ","));
  absl::StrAppend(&string, "] num_shards_per_dim:(");
  absl::StrAppend(&string, absl::StrJoin(num_shards_per_dim, ",") + ")");
  return string;
}

bool ShardVector::ContainsShard(const Shard& shard) const {
  for (const auto& shard_in_vec : shards)
    if (shard_in_vec == shard) return true;
  return false;
}

bool IsDynamicSize(int64_t size) {
  return mlir::ShapedType::isDynamic(size) || size == -1;
}

// static
std::map<std::string, std::vector<int>>& Mesh::tpu_core_ids() {
  static auto tpu_core_ids = new std::map<std::string, std::vector<int>>();
  return *tpu_core_ids;
}

// static
std::string& Mesh::tpu_host_mesh() {
  static auto tpu_host_mesh = new std::string;
  return *tpu_host_mesh;
}

// static
StatusOr<Mesh> Mesh::ParseFromProto(const MeshProto& proto) {
  Mesh mesh;
  mesh.name_ = proto.name();

  for (const auto& device : proto.local_devices()) {
    mesh.local_devices_.push_back(device);
  }

  // Define local device ids.
  for (const auto& device_id : proto.local_device_ids()) {
    mesh.local_device_ids_.push_back(device_id);
  }

  for (const auto& device_id : proto.global_device_ids()) {
    mesh.global_device_ids_.push_back(device_id);
  }

  for (const auto& device : proto.global_devices()) {
    mesh.global_devices_.push_back(device);
  }

  // Assign Mesh Dimensions.
  mesh.mesh_dims_.resize(proto.mesh_dimensions_size());
  for (int i = 0; i < proto.mesh_dimensions_size(); ++i) {
    const MeshDimensionProto& dim = proto.mesh_dimensions(i);
    mesh.mesh_dims_[i].name = dim.name();
    mesh.mesh_dims_[i].size = dim.size();
  }

  mesh.use_xla_spmd_ = proto.use_xla_spmd();

  // Check invariants.
  int64 mesh_size = mesh.size();
  int num_devices = proto.global_device_ids_size();
  if (mesh_size > 0 && mesh_size != num_devices) {
    TF_RETURN_WITH_CONTEXT(
        errors::InvalidArgument("Number of devices ", num_devices,
                                " not matching mesh size ", mesh_size));
  }
  return mesh;
}

// static
StatusOr<Mesh> Mesh::GetAbstractMesh(
    const std::string& name, const std::vector<MeshDimension>& mesh_dims) {
  Mesh mesh;
  mesh.name_ = name;
  mesh.mesh_dims_ = mesh_dims;

  // Check no repeated mesh dimension names.
  std::set<std::string> dims_set;
  for (const MeshDimension& dim : mesh.dims()) {
    if (dims_set.find(dim.name) != dims_set.end())
      TF_RETURN_WITH_CONTEXT(
          errors::InvalidArgument("repeated mesh dimension"));
    if (dim.name == Layout::kAny || dim.name == Layout::kMatch ||
        dim.name == Layout::kUnshardedDim)
      TF_RETURN_WITH_CONTEXT(errors::InvalidArgument("mesh dimension name ",
                                                     dim.name, " is reserved"));
    dims_set.insert(dim.name);
  }

  return mesh;
}

// static
StatusOr<Mesh> Mesh::GetMesh(const std::string& name,
                             const std::vector<MeshDimension>& mesh_dims,
                             const std::vector<std::int64_t>& global_device_ids,
                             const std::vector<std::int64_t>& local_device_ids,
                             const std::vector<std::string>& local_devices,
                             const std::vector<std::string>& global_devices,
                             bool use_xla_spmd) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, GetAbstractMesh(name, mesh_dims));
  mesh.global_device_ids_ = global_device_ids;
  mesh.local_device_ids_ = local_device_ids;
  mesh.local_devices_ = local_devices;
  mesh.global_devices_ = global_devices;
  mesh.use_xla_spmd_ = use_xla_spmd;

  // Check number of devices matches conditions.
  size_t global_n = mesh.global_device_ids_.size();
  size_t local_n = mesh.local_device_ids_.size();
  size_t dev_n = mesh.local_devices_.size();

  if (!(global_n >= local_n && dev_n == local_n))
    TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
        "number of global_device_ids ", std::to_string(global_n),
        " local_devices ids ", std::to_string(local_n), " and local devices ",
        std::to_string(dev_n), "not meeting requirements"));

  // If empty device list, return empty mesh.
  if (global_n == 0) return Mesh::Empty();

  if (local_n && !(global_n % local_n == 0))
    TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
        "Uneven local clusters with global_ids ", std::to_string(global_n),
        " and local_devices ids ", std::to_string(local_n)));

  // Check mesh size matches number of devices.
  if (mesh.size() != global_n)
    TF_RETURN_WITH_CONTEXT(errors::InvalidArgument("mesh size doesn't match",
                                                   "number of devices"));

  // Check local device invariants.
  TF_ASSIGN_OR_RETURN(const auto& parsed_devs, mesh.ParsedDevices());
  std::set<std::string> types_set;
  for (const DeviceNameUtils::ParsedName& dev : parsed_devs) {
    if (!dev.has_job || !dev.has_task || !dev.has_type)
      return errors::InvalidArgument(
          "Failed to either identify host or device type");
    types_set.insert(dev.type);
    if (types_set.size() > 1)
      return errors::InvalidArgument(
          "More than one device type per mesh not supported. Found ",
          types_set.size());
  }

  return mesh;
}

StatusOr<int64_t> Mesh::dim_size(absl::string_view name) const {
  for (const auto& mesh_dim : dims()) {
    if (name == mesh_dim.name) {
      return mesh_dim.size;
    }
  }

  std::vector<std::string> dim_names;
  for (const auto& mesh_dim : dims()) dim_names.push_back(mesh_dim.name);

  return errors::NotFound(
      "Dimension ", name, " does not exist in mesh.",
      "Available dimensions: ", absl::StrJoin(dim_names, ","));
}

std::vector<int64_t> Mesh::dim_sizes() const {
  std::vector<int64_t> dim_sizes;
  if (mesh_dims_.empty()) return dim_sizes;
  for (const auto& mesh_dim : mesh_dims_) dim_sizes.push_back(mesh_dim.size);
  return dim_sizes;
}

bool Mesh::operator==(const Mesh& b) const {
  return protobuf::util::MessageDifferencer::Equals(ToProto(), b.ToProto());
}

bool Mesh::IsEmpty() const { return global_device_ids_.empty(); }

StatusOr<const std::vector<DeviceNameUtils::ParsedName>> Mesh::ParsedDevices()
    const {
  std::vector<DeviceNameUtils::ParsedName> parsed_devices(
      local_devices_.size());
  for (std::size_t i = 0; i < local_devices_.size(); ++i)
    if (!DeviceNameUtils::ParseFullOrLocalName(
            absl::string_view(local_devices_[i]), &parsed_devices[i]))
      return errors::InvalidArgument("Failed to parse local_devices");

  return parsed_devices;
}

StatusOr<Mesh> Mesh::ToDeviceType(const std::string& device_type) const {
  std::vector<std::string> to_local_devices;
  DeviceNameUtils::ParsedName parsed_dev;
  for (const std::string& local_dev : local_devices_) {
    if (!DeviceNameUtils::ParseFullOrLocalName(absl::string_view(local_dev),
                                               &parsed_dev)) {
      return errors::InvalidArgument("Failed to parse local devices");
    }
    // Converted mesh using full task name with job, replica and task ids.
    to_local_devices.push_back(
        DeviceNameUtils::FullName(parsed_dev.job, parsed_dev.replica,
                                  parsed_dev.task, device_type, parsed_dev.id));
    parsed_dev.Clear();
  }
  return GetMesh(name_, mesh_dims_, global_device_ids_, local_device_ids_,
                 to_local_devices, /*global_devices=*/{});
}

namespace {
std::string HostFromParsedDev(const DeviceNameUtils::ParsedName& dev) {
  return "/job:" + dev.job + "/task:" + std::to_string(dev.task);
}
}  //  namespace

std::vector<std::string> Mesh::hosts() const {
  std::vector<std::string> host_list;
  if (IsEmpty()) return host_list;

  const auto parsed_devices = ParsedDevices().value();
  for (const DeviceNameUtils::ParsedName& dev : parsed_devices) {
    std::string host = HostFromParsedDev(dev);
    if (std::find(host_list.begin(), host_list.end(), host) == host_list.end())
      host_list.push_back(host);
  }
  return host_list;
}

std::string Mesh::device_type() const {
  if (IsEmpty()) return std::string();
  std::string device;
  if (!global_devices_.empty()) {
    device = global_devices_[0];
  } else {
    device = local_devices_[0];
  }
  DeviceNameUtils::ParsedName dev;
  DeviceNameUtils::ParseFullOrLocalName(device, &dev);
  return dev.type;
}

bool Mesh::IsMeshDim(const std::string& dim_name) const {
  for (const auto& mesh_dim : dims())
    if (dim_name == mesh_dim.name) return true;
  return false;
}

std::vector<std::string> Mesh::MeshDimNames() const {
  std::vector<std::string> mesh_names;
  for (const auto& mesh_dim : dims()) mesh_names.push_back(mesh_dim.name);
  return mesh_names;
}

int Mesh::GetMeshDimIndexWithName(const std::string& mesh_name) const {
  int mesh_index = -1;
  for (int i = 0; i < dims().size(); ++i) {
    const auto mesh_dim = dim(i);
    if (mesh_dim.name == mesh_name) mesh_index = i;
  }
  assert(mesh_index >= 0);
  return mesh_index;
}

int64 Mesh::rank() const { return mesh_dims_.size(); }

int64 Mesh::size() const {
  if (mesh_dims_.empty()) return 0;

  int64 size = 1;
  for (const MeshDimension& dim : mesh_dims_) size *= dim.size;
  return size;
}

Mesh Mesh::Empty() { return Mesh(); }

MeshProto Mesh::ToProto() const {
  MeshProto mesh_proto;
  mesh_proto.set_name(name());
  mesh_proto.set_use_xla_spmd(use_xla_spmd());

  for (const auto& d : local_devices_) {
    mesh_proto.add_local_devices(d);
  }

  for (const auto& i : local_device_ids_) {
    mesh_proto.add_local_device_ids(i);
  }

  for (const auto& i : global_device_ids_) {
    mesh_proto.add_global_device_ids(i);
  }

  for (const auto& dim : mesh_dims_) {
    MeshDimensionProto* mesh_dim_proto = mesh_proto.add_mesh_dimensions();
    mesh_dim_proto->set_name(dim.name);
    mesh_dim_proto->set_size(dim.size);
  }

  for (const auto& d : global_devices_) {
    mesh_proto.add_global_devices(d);
  }
  return mesh_proto;
}

std::string Mesh::ToString() const {
  if (Mesh::IsEmpty()) return kEmptyMeshString;

  // We use "|" to separate name, mesh dimensions and devices.
  std::string mesh_str = absl::StrCat(Mesh::name(), "|");

  // Add mesh dimensions
  absl::InlinedVector<std::string, 4> mesh_dim_lst;
  for (const auto& dim : mesh_dims_)
    mesh_dim_lst.push_back(absl::StrCat(dim.name, "=", dim.size));
  mesh_str += absl::StrJoin(mesh_dim_lst, ",") + "|";

  // Add flattened list of global device ids
  mesh_str += absl::StrJoin(global_device_ids_, ",") + "|";

  // Add flattened list of local device ids
  mesh_str += absl::StrJoin(local_device_ids_, ",") + "|";

  // Add flattened list of local devices
  mesh_str += absl::StrJoin(local_devices_, ",");

  if (!global_devices_.empty()) {
    // Add flattened list of global devices
    mesh_str += "|";
    mesh_str += absl::StrJoin(global_devices_, ",");
  }

  if (use_xla_spmd()) {
    // Add use_xla_spmd
    mesh_str += "|";
    mesh_str += Mesh::kUseXLASPMDString;
  }
  return mesh_str;
}

uint64 Mesh::GlobalFingerprint() const {
  if (Mesh::IsEmpty()) return Fingerprint64(kEmptyMeshString);

  std::string mesh_str;
  // Add mesh dimensions
  absl::InlinedVector<std::string, 4> mesh_dim_lst;
  for (const auto& dim : mesh_dims_)
    mesh_dim_lst.push_back(absl::StrCat(dim.name, "=", dim.size));
  mesh_str += absl::StrJoin(mesh_dim_lst, ",") + "|";

  // Ignore local_device_ids_, local_devices and name which might be not global
  // unique.
  // Add flattened list of global device ids
  mesh_str += absl::StrJoin(global_device_ids_, ",") + "|";

  if (!global_devices_.empty()) {
    // Add flattened list of global devices
    mesh_str += "|";
    mesh_str += absl::StrJoin(global_devices_, ",");
  }
  // mesh dims | global device ids (| global devices)
  return Fingerprint64(mesh_str);
}

namespace {
MeshDimension StrToMeshDimension(const std::string& str) {
  MeshDimension mesh_dim;
  if (str.empty()) return mesh_dim;

  std::vector<std::string> mesh_dim_parts = absl::StrSplit(str, '=');

  mesh_dim.name = mesh_dim_parts[0];
  mesh_dim.size = std::stoi(mesh_dim_parts[1]);
  return mesh_dim;
}

StatusOr<Mesh> GenerateMeshDevicesForTests(
    const std::string& name, const std::vector<MeshDimension>& mesh_dims,
    const std::string& mesh_gen_instruction) {
  // Parse mesh generation instruction.
  std::vector<std::string> instruction_parts =
      absl::StrSplit(mesh_gen_instruction, '*');
  if (instruction_parts.size() != 2)
    TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
        "Expected a * in mesh_gen_instructions but found ",
        mesh_gen_instruction));
  std::string device_type = instruction_parts[1];

  // Get Mesh Size.
  int64 mesh_size = 0;
  if (!mesh_dims.empty()) {
    mesh_size = 1;
    for (const MeshDimension& mesh_dim : mesh_dims) mesh_size *= mesh_dim.size;
  }

  // Generate device ids.
  std::vector<int64_t> global_device_ids;
  std::vector<int64_t> local_device_ids;
  std::vector<std::string> local_devices;
  for (std::size_t i = 0; i < mesh_size; ++i) {
    global_device_ids.push_back(i);
    local_device_ids.push_back(i);
    local_devices.push_back("/job:localhost/task:0/device:" + device_type +
                            ":" + std::to_string(i));
  }

  TF_ASSIGN_OR_RETURN(
      Mesh mesh,
      Mesh::GetMesh(name, mesh_dims, global_device_ids, local_device_ids,
                    local_devices, /*global_devices=*/{}));
  return mesh;
}
}  // namespace

// static
StatusOr<Mesh> Mesh::FromString(absl::string_view str) {
  if (str == kEmptyMeshString) return Mesh::Empty();

  std::vector<std::string> mesh_parts = absl::StrSplit(str, '|');

  // Check formatting error.
  if (mesh_parts.size() != 3 && mesh_parts.size() != 5 &&
      mesh_parts.size() != 6 && mesh_parts.size() != 7)
    TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
        "Expected either 5, 6, 7 or 3 mesh parts but found",
        mesh_parts.size()));

  // Populate mesh.
  std::string name = mesh_parts[0];

  // Add mesh dimensions.
  std::vector<MeshDimension> mesh_dims;
  if (!mesh_parts[1].empty()) {
    std::vector<std::string> mesh_dim_strs = absl::StrSplit(mesh_parts[1], ',');
    mesh_dims.reserve(mesh_dim_strs.size());
    for (const std::string& mesh_dim_str : mesh_dim_strs)
      mesh_dims.push_back(StrToMeshDimension(mesh_dim_str));
  }

  // Check if mesh is set to be autogenerated.
  if (mesh_parts.size() == 3)
    return GenerateMeshDevicesForTests(name, mesh_dims, mesh_parts[2]);

  // Add global device ids list.
  std::vector<int64_t> global_device_ids;
  if (!mesh_parts[2].empty()) {
    std::vector<std::string> global_device_ids_strs =
        absl::StrSplit(mesh_parts[2], ',');

    global_device_ids.reserve(global_device_ids_strs.size());
    for (const std::string& id : global_device_ids_strs)
      global_device_ids.push_back(std::stoi(id));
  }

  // Add local device ids list.
  std::vector<int64_t> local_device_ids;
  if (!mesh_parts[3].empty()) {
    std::vector<std::string> local_device_ids_strs =
        absl::StrSplit(mesh_parts[3], ',');

    local_device_ids.reserve(local_device_ids_strs.size());
    for (const std::string& id : local_device_ids_strs)
      local_device_ids.push_back(std::stoi(id));
  }
  // Add local devices.
  std::vector<std::string> local_devices;
  if (!mesh_parts[4].empty())
    local_devices = absl::StrSplit(mesh_parts[4], ',');

  bool use_xla_spmd = Mesh::kUseXLASPMD;
  std::vector<std::string> global_devices;
  if (mesh_parts.size() == 6 && !mesh_parts[5].empty()) {
    // Add global devices.
    if (mesh_parts[5] == Mesh::kUseXLASPMDString) {
      use_xla_spmd = true;
    } else {
      global_devices = absl::StrSplit(mesh_parts[5], ',');
    }
  }
  // Add use_xla_spmd.
  if (mesh_parts.size() == 7 && !mesh_parts[6].empty()) {
    if (mesh_parts[6] == Mesh::kUseXLASPMDString) {
      use_xla_spmd = true;
    } else {
      return errors::InvalidArgument(
          "Expected string ", Mesh::kUseXLASPMDString,
          "as the 7th argument but got: ", mesh_parts[6]);
    }
  }

  TF_ASSIGN_OR_RETURN(
      Mesh mesh,
      Mesh::GetMesh(name, mesh_dims, global_device_ids, local_device_ids,
                    local_devices, global_devices, use_xla_spmd));
  return mesh;
}

int64 Mesh::num_devices() const { return global_device_ids_.size(); }

StatusOr<const DeviceLocation> Mesh::device_location(int offset) const {
  if (offset < 0 || offset > size() - 1)
    return errors::InvalidArgument(
        "Mesh offset cannot be negative or exceed Mesh's size. Offset size:",
        offset, " and Mesh size:", size());

  DeviceLocation dev_loc;
  std::vector<int64> mesh_dim_lengths = dim_sizes();
  int64 i = mesh_dim_lengths.size() - 1;
  while (i >= 0) {
    dev_loc.insert(dev_loc.begin(), offset % mesh_dim_lengths[i]);
    offset /= mesh_dim_lengths[i];
    --i;
  }
  return dev_loc;
}

int64 Mesh::GetFlattenedCoordinate(const DeviceLocation& loc) const {
  const std::vector<int64> mesh_dim_sizes = dim_sizes();
  int64 i = mesh_dim_sizes.size() - 1;
  int64 acc = 1;
  int64 device_pos = 0;
  while (i >= 0) {
    device_pos += loc[i] * acc;
    acc *= mesh_dim_sizes[i];
    --i;
  }
  return device_pos;
}

StatusOr<int32> Mesh::idx_for_dim(absl::string_view dim_name) const {
  for (int i = 0; i < mesh_dims_.size(); ++i) {
    if (mesh_dims_[i].name == dim_name) return i;
  }
  return errors::InvalidArgument("dim name :", dim_name,
                                 " does not exist on mesh : ", ToString());
}

Mesh Mesh::CreateMesh(const std::string& mesh_name,
                      const std::vector<std::string>& dim_names,
                      const std::vector<std::int64_t>& mesh_shape,
                      const std::vector<std::int64_t>& global_device_ids,
                      const std::vector<std::string>& global_devices_str,
                      const std::vector<std::int64_t>& local_device_ids,
                      const std::vector<std::string>& local_devices_str,
                      const bool use_xla_spmd) {
  Mesh mesh;
  mesh.name_ = mesh_name;
  mesh.use_xla_spmd_ = use_xla_spmd;
  mesh.mesh_dims_.resize(dim_names.size());

  for (int i = 0; i < dim_names.size(); ++i) {
    mesh.mesh_dims_[i].name = dim_names[i];
    mesh.mesh_dims_[i].size = mesh_shape[i];
  }

  for (const auto& id : global_device_ids) {
    mesh.global_device_ids_.push_back(id);
  }

  for (const auto& d : global_devices_str) {
    mesh.global_devices_.push_back(d);
  }

  for (const auto& id : local_device_ids) {
    mesh.local_device_ids_.push_back(id);
  }

  for (const auto& d : local_devices_str) {
    mesh.local_devices_.push_back(d);
  }

  return mesh;
}

StatusOr<Layout> Layout::GetLayout(
    const std::vector<std::string>& sharding_spec_strs, const Mesh& mesh) {
  // Re-format sharding specs.
  std::vector<ShardingSpec> sharding_specs;
  sharding_specs.reserve(sharding_spec_strs.size());
  for (const std::string& spec_str : sharding_spec_strs) {
    ShardingSpec spec;
    spec.set_sharding_spec(spec_str);
    sharding_specs.push_back(spec);
  }
  return GetLayout(sharding_specs, mesh);
}

StatusOr<Layout> Layout::GetLayout(
    const std::vector<ShardingSpec>& sharding_specs, const Mesh& mesh) {
  Layout layout;
  // Append mesh, then check sharding_specs are legal.
  layout.mesh_ = mesh;

  // Check sharding_specs are either mesh dimension or special value.
  for (const auto& dim : sharding_specs) {
    const std::string& sharding_spec = dim.sharding_spec();
    if (!(sharding_spec == kUnshardedDim || sharding_spec == kAny ||
          sharding_spec == kMatch || mesh.IsMeshDim(sharding_spec) ||
          sharding_spec == "scalar"))
      TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
          "sharding spec (", sharding_spec,
          ") refers to mesh dimension not contained in mesh ",
          mesh.ToString()));
  }
  // Check same tensor dimensions not sharded over same mesh dimension twice.
  std::set<std::string> dims_set;
  for (const auto& dim : sharding_specs) {
    const std::string& sharding_spec = dim.sharding_spec();
    if (sharding_spec == kUnshardedDim || sharding_spec == kAny) continue;
    // If scalar, delete all sharding specs.
    if (sharding_spec == "scalar") {
      if (sharding_specs.size() > 1)
        TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
            "A scalar sharding_spec can only be used as a single sharding_spec "
            "instruction, not as part of list of sharding_specs as attempted "
            "here with ",
            sharding_specs.size(), " sharding_specs"))
      // Return layout with empty spec to represent scalar behavior.
      return layout;
    }
    if (dims_set.find(sharding_spec) != dims_set.end())
      TF_RETURN_WITH_CONTEXT(
          errors::InvalidArgument("Attempted to shard two or more tensor "
                                  "dimensions over mesh dimension ",
                                  sharding_spec))
    dims_set.insert(sharding_spec);
  }
  // After checking sharding_specs are legal, append and return layout.
  layout.sharding_specs_ = sharding_specs;
  return layout;
}

Layout Layout::Empty() {
  Layout result;
  return result;
}

bool Layout::IsEmpty() const { return mesh_.IsEmpty(); }

namespace {
Mesh ReducedAbstractMesh(const Layout* layout) {
  const std::vector<std::string>& shard_spec_strs =
      layout->sharding_spec_strs();
  std::vector<MeshDimension> reduced_mesh_dims;
  reduced_mesh_dims.reserve(layout->mesh().dims().size());
  for (const MeshDimension& mesh_dim : layout->mesh().dims()) {
    bool IsMeshDimInShardingSpecs =
        std::find(shard_spec_strs.begin(), shard_spec_strs.end(),
                  mesh_dim.name) != shard_spec_strs.end();
    // If dimension not in sharding_spec, flip size to 1.
    MeshDimension reduced_dim =
        IsMeshDimInShardingSpecs ? mesh_dim : MeshDimension(mesh_dim.name, 1);
    reduced_mesh_dims.push_back(reduced_dim);
  }
  return Mesh::GetAbstractMesh("", reduced_mesh_dims).value();
}

}  // namespace

Mesh Layout::ReducedMesh() const {
  // Set replicated mesh dimensions to size 1, and create reduced abstract mesh.
  Mesh reduced_mesh = ReducedAbstractMesh(this);

  // Populate reduced mesh with global devices from original mesh.
  std::vector<int64_t> reduced_global_device_ids;
  std::vector<std::string> reduced_global_devs;
  for (const DeviceLocation& loc : ComputeDeviceLocations(reduced_mesh)) {
    int64 pos = mesh().GetFlattenedCoordinate(loc);
    reduced_global_device_ids.push_back(mesh().global_device_ids().at(pos));
    if (!mesh().global_devices().empty()) {
      reduced_global_devs.push_back(mesh().global_devices().at(pos));
    }
  }

  // Track the set of global device IDs in the abstract mesh.
  std::set<int64_t> reduced_global_device_ids_set(
      reduced_global_device_ids.begin(), reduced_global_device_ids.end());

  // Populate reduced mesh with local devices in the same order as the original
  // mesh.
  std::vector<int64_t> reduced_local_device_ids;
  std::vector<std::string> reduced_local_devs;
  for (size_t i = 0; i < mesh().local_device_ids().size(); ++i) {
    int64_t device_id = mesh().local_device_ids().at(i);
    if (reduced_global_device_ids_set.find(device_id) !=
        reduced_global_device_ids_set.end()) {
      reduced_local_device_ids.push_back(device_id);
      reduced_local_devs.push_back(mesh().local_devices().at(i));
    }
  }

  return Mesh::GetMesh(reduced_mesh.name(), reduced_mesh.dims(),
                       reduced_global_device_ids, reduced_local_device_ids,
                       reduced_local_devs, reduced_global_devs)
      .value();
}

namespace {
Layout ReducedLayout(const Layout* layout) {
  // Change format sharding specs.
  std::vector<ShardingSpec> shard_specs(layout->sharding_specs().size());
  for (size_t i = 0; i < shard_specs.size(); ++i)
    shard_specs[i] = layout->dim(i);
  // Retrieve layout.
  return Layout::GetLayout(shard_specs, layout->ReducedMesh()).value();
}

// Returns index of the given mesh dimension or mesh dim size if not found.
StatusOr<int> IndexOfMeshDimension(const Mesh& mesh,
                                   const std::string& dim_name) {
  for (size_t i = 0; i < mesh.dims().size(); ++i)
    if (dim_name == mesh.dims()[i].name) return i;
  return errors::InvalidArgument("Mesh dimension not found");
}
}  // namespace

ShardVector Layout::GetShardVector() const {
  // Change format sharding specs.
  std::vector<ShardingSpec> shard_specs(sharding_specs().size());
  for (size_t i = 0; i < shard_specs.size(); ++i) shard_specs[i] = dim(i);
  // Obtain a shard position (i.e. sharded section of a tensor) from a mesh
  // location, using the sharding specs.
  auto GetShardFromDeviceLocation = [&](const DeviceLocation& loc) -> Shard {
    Shard shard;
    for (size_t i = 0; i < shard_specs.size(); ++i) {
      // If unsharded, there is only one shard, that is 1.
      std::string spec = shard_specs[i].sharding_spec();
      if (spec == Layout::kUnshardedDim) {
        shard.push_back(1);
      } else {
        int mesh_index = IndexOfMeshDimension(mesh(), sharding_spec(i)).value();
        int shard_number = loc[mesh_index] + 1;
        shard.push_back(shard_number);
      }
    }
    return shard;
  };
  // Obtain dims of shard vector.
  auto ShardVectorDims = [&]() -> std::vector<int> {
    std::vector<int> num_shards_per_dim(shard_specs.size());
    for (size_t i = 0; i < sharding_specs().size(); ++i) {
      ShardingSpec spec = sharding_specs()[i];
      if (Layout::IsShardedSpec(spec)) {
        StatusOr<int64> dim_size = mesh().dim_size(spec.sharding_spec());
        num_shards_per_dim[i] = dim_size.value();
      } else {
        num_shards_per_dim[i] = 1;
      }
    }
    return num_shards_per_dim;
  };
  // Compute mesh locations and obtain shards from them.
  ShardVector shard_vec;
  for (const DeviceLocation& mesh_loc : ComputeDeviceLocations(mesh()))
    shard_vec.shards.push_back(GetShardFromDeviceLocation(mesh_loc));
  // Calculate dims.
  shard_vec.num_shards_per_dim = ShardVectorDims();
  return shard_vec;
}

std::map<std::string, ShardVector> Layout::HostShardMap() const {
  Layout reduced_layout = ReducedLayout(this);
  Mesh reduced_mesh = reduced_layout.mesh();
  using HostName = std::string;

  // Build a map: {Host : Shards}
  std::map<HostName, ShardVector> host_shards_map;
  ShardVector shard_vec_in_red_layout = reduced_layout.GetShardVector();

  const auto parsed_devs = reduced_mesh.ParsedDevices().value();
  for (size_t i = 0; i < parsed_devs.size(); ++i) {
    HostName host = HostFromParsedDev(parsed_devs[i]);
    Shard shard_in_device = shard_vec_in_red_layout.shards[i];

    // Check if host in hashtable and append shard.
    auto it = host_shards_map.find(host);
    if (it == host_shards_map.end()) {
      ShardVector shard_vec_in_host;
      shard_vec_in_host.shards.push_back(shard_in_device);
      shard_vec_in_host.num_shards_per_dim =
          shard_vec_in_red_layout.num_shards_per_dim;
      host_shards_map.insert(
          std::pair<HostName, ShardVector>(host, shard_vec_in_host));
    } else {
      bool isShardInShardVector = it->second.ContainsShard(shard_in_device);
      if (!isShardInShardVector) {
        it->second.shards.push_back(shard_in_device);
      }
    }
  }
  // Sort shards inside each host.
  for (auto it = host_shards_map.begin(); it != host_shards_map.end(); ++it) {
    std::sort(it->second.shards.begin(), it->second.shards.end());
  }
  return host_shards_map;
}

const std::string& Layout::sharding_spec(int idx) const {
  return sharding_specs_[idx].sharding_spec();
}

std::vector<int32> Layout::num_shards() const {
  std::vector<int32> num_shards;
  num_shards.reserve(sharding_specs_.size());
  for (const auto& sharding_spec : sharding_specs_) {
    num_shards.push_back(num_shards_for_dim(sharding_spec));
  }
  return num_shards;
}

size_t Layout::num_shards_for_dim(const ShardingSpec& dim) const {
  absl::string_view name = dim.sharding_spec();
  if (name == Layout::kUnshardedDim) return 1;
  if (name == Layout::kMatch) return -1;

  return mesh().dim_size(name).value();
}

bool Layout::IsFullyReplicated() const {
  for (const auto& sharding_spec : sharding_specs_) {
    if (num_shards_for_dim(sharding_spec) > 1) {
      return false;
    }
  }
  return true;
}

bool Layout::IsLastDimReplicated() const {
  return (sharding_specs_.empty()) ||
         (num_shards_for_dim(sharding_specs_.back()) == 1);
}

bool Layout::IsBatchParallel() const {
  if (sharding_specs_.empty()) {
    return true;
  }

  for (int i = 1; i < sharding_specs_.size(); ++i) {
    const auto& dim = sharding_specs_[i];
    if (num_shards_for_dim(dim) != 1) {
      return false;
    }
  }
  return true;
}

// TODO(samuelslee) Replace this with the IsBatchParallel() everywhere
bool Layout::IsBatchParallel(int non_batch_rank) const {
  if (sharding_specs_.empty()) return true;
  for (int i = rank() - non_batch_rank; i < rank(); ++i) {
    if (num_shards_for_dim(sharding_specs_[i]) != 1) return false;
  }
  return true;
}

LayoutProto Layout::ToProto() const {
  LayoutProto proto;
  *proto.mutable_mesh_config() = mesh_.ToProto();
  for (const auto& dim : sharding_specs_) {
    *proto.add_sharding_specs() = dim;
  }
  return proto;
}

bool Layout::IsEquivalent(const Layout& b) const {
  if (this->rank() != b.rank()) return false;
  if (this->mesh() != b.mesh()) return false;
  for (int i = 0; i < this->rank(); ++i) {
    if (this->sharding_specs_[i].sharding_spec() !=
        b.sharding_specs_[i].sharding_spec()) {
      if ((this->num_shards_for_dim(this->sharding_specs_[i]) != 1) ||
          (b.num_shards_for_dim(b.sharding_specs_[i]) != 1))
        return false;
    }
  }
  return true;
}

bool Layout::operator==(const Layout& b) const {
  return protobuf::util::MessageDifferencer::Equals(ToProto(), b.ToProto());
}

std::vector<int64_t> Layout::GlobalShapeFromLocalShape(
    absl::Span<const int64_t> local_shape) const {
  if (IsFullyReplicated()) {
    return std::vector<int64_t>(local_shape.begin(), local_shape.end());
  }
  std::vector<int64_t> global_shape;
  global_shape.reserve(sharding_specs().size());
  for (int i = 0; i < sharding_specs().size(); ++i) {
    int64_t l_shape = local_shape.empty() ? 1 : local_shape[i];
    int64_t dim_shards = num_shards()[i];
    int64_t global_size =
        IsDynamicSize(l_shape) ? l_shape : l_shape * dim_shards;
    global_shape.emplace_back(global_size);
  }
  return global_shape;
}

std::vector<int64_t> Layout::LocalShapeFromGlobalShape(
    absl::Span<const int64_t> global_shape) const {
  if (IsFullyReplicated()) {
    return std::vector<int64_t>(global_shape.begin(), global_shape.end());
  }
  std::vector<int32> shards = num_shards();
  std::vector<int64_t> local_shape;
  for (int i = 0; i < sharding_specs().size(); ++i) {
    int64_t dim_shards = shards[i];
    // TODO(hthu): Shape might not be always divisible.
    int64_t local_size = IsDynamicSize(global_shape[i])
                             ? global_shape[i]
                             : global_shape[i] / dim_shards;
    local_shape.emplace_back(local_size);
  }
  return local_shape;
}

PartialTensorShape Layout::LocalShapeFromGlobalShape(
    const PartialTensorShape& global_shape) const {
  if (IsFullyReplicated() || global_shape.dims() == -1) {
    return global_shape;
  }
  std::vector<int32> shards = num_shards();
  PartialTensorShape local_shape({});
  for (int spec_index = 0; spec_index < sharding_specs().size(); ++spec_index) {
    int64_t dim_size = global_shape.dim_size(spec_index);
    int64_t local_size =
        IsDynamicSize(dim_size) ? dim_size : dim_size / shards[spec_index];
    local_shape.AddDim(local_size);
  }
  return local_shape;
}

StatusOr<Layout> Layout::FromProto(const LayoutProto& proto) {
  Layout layout;
  for (const auto& spec : proto.sharding_specs())
    layout.sharding_specs_.push_back(spec);

  TF_ASSIGN_OR_RETURN(auto mesh, Mesh::ParseFromProto(proto.mesh_config()));
  layout.mesh_ = std::move(mesh);

  return GetLayout(layout.sharding_specs_, layout.mesh_);
}

Layout Layout::ReplicatedOnMesh(const Mesh& mesh, int rank) {
  std::vector<std::string> specs(rank, kUnshardedDim);
  return Layout::GetLayout(specs, mesh).value();
}

Layout Layout::ReplicatedLike(const Layout& layout) {
  std::vector<std::string> specs(layout.rank(), kUnshardedDim);
  return Layout::GetLayout(specs, layout.mesh()).value();
}

Layout Layout::BatchShardedOnMesh(const Mesh& mesh, int rank,
                                  const string& mesh_dim, int axis) {
  std::vector<std::string> specs(rank, kUnshardedDim);
  specs[axis] = mesh_dim;
  return Layout::GetLayout(specs, mesh).value();
}

Layout Layout::BatchShardedLike(const Layout& layout, const string& mesh_dim,
                                int axis) {
  std::vector<std::string> specs(layout.rank(), kUnshardedDim);
  specs[axis] = mesh_dim;
  return Layout::GetLayout(specs, layout.mesh()).value();
}

Layout Layout::AnyOnMesh(const Mesh& mesh, int rank) {
  std::vector<std::string> specs(rank, kAny);
  return Layout::GetLayout(specs, mesh).value();
}

StatusOr<Layout> Layout::Transposed2D(const Layout& layout) {
  if (layout.rank() < 2) {
    return errors::InvalidArgument("Transposed2D requires rank to be >= 2");
  }
  std::vector<std::string> transposed_specs = layout.sharding_spec_strs();
  std::iter_swap(transposed_specs.end() - 2, transposed_specs.end() - 1);
  return Layout::GetLayout(transposed_specs, layout.mesh()).value();
}

// static
StatusOr<Layout> Layout::FromString(absl::string_view layout_str) {
  if (layout_str == kEmptyLayoutString) return Layout::Empty();

  // Print sharding specs.
  std::vector<absl::string_view> layout_parts = absl::StrSplit(layout_str, ' ');
  // Check formatting error.
  if (layout_parts.size() != 2) {
    TF_RETURN_WITH_CONTEXT(errors::InvalidArgument(
        "Expected 2 items but found ", layout_parts.size(), layout_parts[0]));
  }
  // Substract prefixes.
  absl::string_view sharding_spec_str = layout_parts[0];
  absl::ConsumePrefix(&sharding_spec_str, "sharding_specs:");

  absl::string_view mesh_str = layout_parts[1];
  absl::ConsumePrefix(&mesh_str, "mesh:");

  // Add sharding specs.
  std::vector<std::string> sharding_spec_strs =
      absl::StrSplit(sharding_spec_str, ',');
  sharding_spec_strs.pop_back();

  // Add mesh.
  TF_ASSIGN_OR_RETURN(Mesh mesh, Mesh::FromString(mesh_str));
  // Try to create layout.
  TF_ASSIGN_OR_RETURN(Layout layout,
                      Layout::GetLayout(sharding_spec_strs, mesh));
  return layout;
}

std::vector<std::string> Layout::sharding_spec_strs() const {
  std::vector<std::string> sharding_spec_strs(sharding_specs().size());
  for (size_t i = 0; i < sharding_specs().size(); ++i)
    sharding_spec_strs[i] = sharding_spec(i);
  return sharding_spec_strs;
}

std::string Layout::ToString() const {
  if (Layout::IsEmpty()) return kEmptyLayoutString;

  std::string layout_str = "sharding_specs:";
  // Print sharding specs.
  for (const ShardingSpec& dim : sharding_specs_) {
    std::string dim_name = dim.sharding_spec();
    absl::StrAppend(&layout_str, dim_name + ",");
  }
  // Append mesh.
  absl::StrAppend(&layout_str, " mesh:", mesh_.ToString());
  return layout_str;
}

Layout Layout::GetLayoutWithReducedDims(
    const absl::flat_hash_set<int>& reduced_dims, bool keep_dims) const {
  dtensor::LayoutProto output_layout;
  *output_layout.mutable_mesh_config() = mesh().ToProto();

  for (int i = 0; i < rank(); ++i) {
    // reduced_dims may contain negative values.
    if (!reduced_dims.contains(i) && !reduced_dims.contains(i - rank())) {
      *output_layout.add_sharding_specs() = dim(i);
    } else if (keep_dims) {
      auto* replicated_dim = output_layout.add_sharding_specs();
      replicated_dim->set_sharding_spec(kUnshardedDim);
    }
  }
  return Layout::FromProto(output_layout).value();
}

Layout Layout::Truncate(int64 split_point, bool end) const {
  if ((split_point == 0 && end) || (split_point == rank() && !end))
    return *this;

  dtensor::LayoutProto output_layout;
  *output_layout.mutable_mesh_config() = mesh().ToProto();

  if (end) {
    for (int i = split_point; i < rank(); ++i)
      *output_layout.add_sharding_specs() = dim(i);
  } else {
    for (int i = 0; i < split_point; ++i)
      *output_layout.add_sharding_specs() = dim(i);
  }
  return Layout::FromProto(output_layout).value();
}

namespace {
// Adds unsharded sharding specs to layout.
Layout PadLayout(const int64 rank, const bool is_padding_before,
                 const Layout& layout) {
  if (rank <= layout.rank()) return layout;

  // Create list of padding sharding specs.
  const int n = rank - layout.rank();
  std::vector<ShardingSpec> new_specs(n);
  for (int i = 0; i < n; ++i)
    new_specs[i].set_sharding_spec(Layout::kUnshardedDim);

  // Define concatenation point of layout specs.
  auto concat_point = is_padding_before ? new_specs.end() : new_specs.begin();

  // Concatenate old layout specs and new unsharded specs.
  new_specs.insert(concat_point, layout.sharding_specs().begin(),
                   layout.sharding_specs().end());
  return Layout::GetLayout(new_specs, layout.mesh()).value();
}
}  // namespace

Layout Layout::LeftPad(int64 rank) const {
  bool is_padding_before = true;
  return PadLayout(rank, is_padding_before, *this);
}

StatusOr<Layout> ConcatenateLayouts(const Layout& layout_a,
                                    const Layout& layout_b) {
  if (layout_a.mesh() != layout_b.mesh())
    return errors::InvalidArgument(
        "unable to concatenate layouts as they are on different meshes.");

  absl::flat_hash_set<std::string> layout_a_mesh_dims;
  for (int i = 0; i < layout_a.rank(); ++i)
    if (layout_a.sharding_spec(i) != Layout::kUnshardedDim)
      layout_a_mesh_dims.emplace(layout_a.sharding_spec(i));

  for (int i = 0; i < layout_b.rank(); ++i)
    if (layout_b.sharding_spec(i) != Layout::kUnshardedDim &&
        layout_a_mesh_dims.contains(layout_b.sharding_spec(i)))
      return errors::InvalidArgument(
          "unable to concatenate layouts as they use the same meshes "
          "dimension: ",
          layout_b.sharding_spec(i), " is used in both layouts.");

  LayoutProto layout_proto_a = layout_a.ToProto();
  LayoutProto layout_proto_b = layout_b.ToProto();
  LayoutProto output_layout_proto;

  *output_layout_proto.mutable_mesh_config() = layout_proto_a.mesh_config();
  for (int i = 0; i < layout_proto_a.sharding_specs_size(); ++i)
    *output_layout_proto.add_sharding_specs() =
        layout_proto_a.sharding_specs(i);
  for (int i = 0; i < layout_proto_b.sharding_specs_size(); ++i)
    *output_layout_proto.add_sharding_specs() =
        layout_proto_b.sharding_specs(i);
  return Layout::FromProto(output_layout_proto);
}

}  // namespace dtensor
}  // namespace tensorflow
