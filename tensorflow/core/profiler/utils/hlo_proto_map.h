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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HLO_PROTO_MAP_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HLO_PROTO_MAP_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

absl::flat_hash_map<uint64_t /*program_id*/, std::unique_ptr<xla::HloProto>>
ParseHloProtosFromXSpace(const XSpace& space);

class HloProtoMap {
 public:
  void AddHloProtosFromXSpace(const XSpace& space);

  void AddHloProto(uint64_t program_id,
                   std::unique_ptr<const xla::HloProto> hlo_proto);

  size_t size() const { return hlo_protos_by_program_id_.size(); }

  auto begin() const { return hlo_protos_by_program_id_.begin(); }
  auto end() const { return hlo_protos_by_program_id_.end(); }

  bool contains(absl::string_view name) const {
    return hlo_protos_by_name_.contains(name);
  }

  bool contains(uint64_t program_id) const {
    return hlo_protos_by_program_id_.contains(program_id);
  }

  // Returns a list of module names (not sorted).
  std::vector<absl::string_view> GetModuleList() const;

  // Returns a list of module names sorted alphabetically.
  std::vector<absl::string_view> GetSortedModuleList() const;

  // Returns a list of hlo module names sorted first by heap trace size and then
  // by hlo module name alphabetically.
  std::vector<absl::string_view> GetSortedModuleListByHeapTraceSize() const;

  absl::StatusOr<const xla::HloProto*> GetHloProtoByModuleName(
      absl::string_view module_name) const;

  absl::StatusOr<const xla::HloProto*> GetHloProtoByProgramId(
      uint64_t program_id) const;

 private:
  absl::flat_hash_map<uint64_t, const xla::HloProto*> hlo_protos_by_program_id_;
  absl::flat_hash_map<std::string, const xla::HloProto*> hlo_protos_by_name_;
  std::vector<std::unique_ptr<const xla::HloProto>> owned_hlo_protos_;

  // Try to add proto to the map and returns true if the addition is successful
  // (i.e., the proto is new to the map).
  bool AddHloProto(uint64_t program_id, const xla::HloProto* hlo_proto);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HLO_PROTO_MAP_H_
