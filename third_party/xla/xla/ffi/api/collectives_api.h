/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_FFI_API_COLLECTIVES_API_H_
#define XLA_FFI_API_COLLECTIVES_API_H_

#include <cstdint>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/collectives_c_api.h"

// C++ wrapper for the XLA FFI Collectives API.
namespace xla::ffi {

// Mirrors `XLA_FFI_CollectiveGroupMode` / `xla::CollectiveOpGroupMode`.
enum class GroupMode {
  kCrossReplica = XLA_FFI_GROUP_CROSS_REPLICA,
  kCrossPartition = XLA_FFI_GROUP_CROSS_PARTITION,
  kCrossReplicaAndPartition = XLA_FFI_GROUP_CROSS_REPLICA_AND_PARTITION,
  kFlattenedId = XLA_FFI_GROUP_FLATTENED_ID,
};

namespace internal {

// C++ wrapper for the XLA FFI Collectives extension API.
// Unified implementation for internal and external FFI modules.
template <typename ErrorPolicy>
class CommunicatorContextBase {
 public:
  using Status = typename ErrorPolicy::Status;
  template <typename T>
  using StatusOr = typename ErrorPolicy::template StatusOr<T>;

  CommunicatorContextBase(const XLA_FFI_Api* api,
                          const XLA_FFI_Collectives_Extension* ext)
      : api_(api), ext_(ext) {}

  // Requests the clique for `groups` so it is acquired before execution.
  // Prepare stage only.
  Status RequestCommunicator(GroupMode group_mode,
                             const std::vector<std::vector<int64_t>>& groups,
                             int64_t communication_id) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_Communicator_Request_Args args;
    args.struct_size = XLA_FFI_Communicator_Request_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    if (XLA_FFI_Error* err = ext_->request_communicator(ext_, &args)) {
      return ErrorPolicy::TakeError(api_, err);
    }
    return ErrorPolicy::Ok();
  }

  // Returns the non-owning communicator handle for `groups`. The handle is
  // backend-defined; the caller reinterprets it (e.g. as `ncclComm_t`).
  StatusOr<XLA_FFI_Communicator*> GetCommunicator(
      GroupMode group_mode, const std::vector<std::vector<int64_t>>& groups,
      int64_t communication_id) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups = ToRawGroups(groups);
    XLA_FFI_Communicator_Get_Args args;
    args.struct_size = XLA_FFI_Communicator_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.group_mode = static_cast<XLA_FFI_CollectiveGroupMode>(group_mode);
    args.groups = raw_groups.data();
    args.num_groups = raw_groups.size();
    args.communication_id = communication_id;
    args.communicator = nullptr;
    if (XLA_FFI_Error* err = ext_->get_communicator(ext_, &args)) {
      return StatusOr<XLA_FFI_Communicator*>(ErrorPolicy::TakeError(api_, err));
    }
    return args.communicator;
  }

 private:
  // Converts a vector of replica groups to a vector of `XLA_FFI_ReplicaGroup`.
  // The results reference the id storage in `groups`, which must outlive them.
  static std::vector<XLA_FFI_ReplicaGroup> ToRawGroups(
      const std::vector<std::vector<int64_t>>& groups) {
    std::vector<XLA_FFI_ReplicaGroup> raw_groups;
    raw_groups.reserve(groups.size());
    for (const std::vector<int64_t>& group : groups) {
      raw_groups.push_back(XLA_FFI_ReplicaGroup{group.data(), group.size()});
    }
    return raw_groups;
  }

  const XLA_FFI_Api* api_;
  const XLA_FFI_Collectives_Extension* ext_;
};

// Common base struct for internal and external Collectives extensions.
// Defines traits for CtxDecoding<Extension<Collectives>>.
template <typename CommunicatorContextT>
struct CollectivesExtensionBase {
  using Type = CommunicatorContextT;
  using CExtension = XLA_FFI_Collectives_Extension;

  static constexpr auto kName = "CollectivesExtension";
  static constexpr int32_t kExtensionType = XLA_FFI_Extension_Collectives;
  static constexpr int32_t kMajorVersion =
      XLA_FFI_Extension_Collectives_MajorVersion;
  static constexpr int32_t kMinorVersion =
      XLA_FFI_Extension_Collectives_MinorVersion;

  // Builds a context from the extension.
  static CommunicatorContextT Create(const XLA_FFI_Api* api,
                                     const CExtension* ext) {
    return CommunicatorContextT(api, ext);
  }
};

}  // namespace internal

}  // namespace xla::ffi

#endif  // XLA_FFI_API_COLLECTIVES_API_H_
