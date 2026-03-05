/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_HLO_MODULE_METADATA_H_
#define XLA_HLO_IR_HLO_MODULE_METADATA_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/protobuf.h"

namespace xla {

// This is a temporary define to solve a versioning problem with the XLA and
// xprof repositories. It may be removed after xprof syncs XLA past Feb 16,
// 2026.
#define XLA_HAVE_STACK_FRAME_ID 1

// Strong typedef for stack frame IDs.
struct StackFrameId {
  int value = 0;  // 0 is reserved for "not present".
  bool operator==(StackFrameId other) const { return value == other.value; }
  bool operator!=(StackFrameId other) const { return value != other.value; }
  bool valid() const { return value != 0; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, StackFrameId id) {
    absl::Format(&sink, "%d", id.value);
  }

  template <typename H>
  friend H AbslHashValue(H h, StackFrameId id) {
    return H::combine(std::move(h), id.value);
  }
};

// Describes a stack frame.
struct HloStackFrame {
  absl::string_view file_name;
  absl::string_view function_name;
  int line = 0;
  int column = 0;
  int end_line = 0;
  int end_column = 0;

  // Index of the parent frame. An invalid (0) value indicates the root.
  StackFrameId parent_frame_id;

  bool empty() const {
    return line == 0 && column == 0 && end_line == 0 && end_column == 0 &&
           file_name.empty() && function_name.empty();
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HloStackFrame& frame) {
    absl::Format(&sink, "%s:%d", frame.file_name, frame.line);
    if (frame.column != 0) {
      absl::Format(&sink, ":%d", frame.column);
    }
    if (!frame.function_name.empty()) {
      absl::Format(&sink, " [%s]", frame.function_name);
    }
  }
};

// Wrapper class for HloModuleMetadataProto to avoid allowing callers to mutate
// arbitrary fields. Specifically, callers cannot set timestamps or ids or
// set the fields of any pass not currently running.
class HloModuleMetadata {
 public:
  explicit HloModuleMetadata(tsl::Env* env) : env_(env) {}

  HloModuleMetadata(const HloModuleMetadata& other) { CopyFrom(other); }

  HloModuleMetadata& operator=(const HloModuleMetadata& other) {
    if (this != &other) {
      CopyFrom(other);
    }
    return *this;
  }

  const HloModuleMetadataProto& proto() const { return module_metadata_; }

  // Creates a new HloPassMetadata. All calls to RecordPassStart should be
  // matched by a later call to RecordPassEnd.
  void RecordPassStart();

  // Marks the currently running pass as finished. Returns NotFound if metadata
  // for the currently running pass cannot be found.
  absl::Status RecordPassEnd();

  const std::optional<HloModuleMetadataProto>& prepartitioning_metadata()
      const {
    return prepartitioning_metadata_;
  }
  void set_prepartitioning_metadata(
      const HloModuleMetadata& prepartitioning_metadata);

  // Setters for HloModuleMetadataProto.
  void set_module_group_name(const std::string& name) {
    module_metadata_.set_module_group_name(name);
  }
  void set_canonical_module_id(int64_t id) {
    module_metadata_.set_canonical_module_id(id);
  }
  void add_partitioned_module_id(int64_t id) {
    module_metadata_.add_partitioned_module_ids(id);
  }
  absl::Status set_custom_metadata(const ::tsl::protobuf::Message& message);
  // Adds a (key, value) pair metric if none was already set. Otherwise, it
  // updates the existing value.
  absl::Status set_key_value_metric(const std::string& key, int64_t value);

  absl::StatusOr<int64_t> current_pass_id() {
    TF_ASSIGN_OR_RETURN(HloPassMetadata * pass_metadata,
                        GetCurrentHloPassMetadata());
    return pass_metadata->pass_id();
  }

  // Setters for the current HloPassMetadata.
  absl::Status set_current_pass_name(const std::string& pass_name) {
    return MutateCurrentHloPassMetadata(
        [&pass_name](HloPassMetadata* pass_metadata) {
          pass_metadata->set_pass_name(pass_name);
        });
  }
  absl::Status set_current_pass_pipeline_name(
      const std::string& pipeline_name) {
    return MutateCurrentHloPassMetadata(
        [&pipeline_name](HloPassMetadata* pass_metadata) {
          pass_metadata->set_pipeline_name(pipeline_name);
        });
  }
  absl::Status add_current_pass_dump_filename(
      const std::string& dump_filename) {
    return MutateCurrentHloPassMetadata(
        [&dump_filename](HloPassMetadata* pass_metadata) {
          pass_metadata->add_dump_filenames(dump_filename);
        });
  }
  absl::Status set_current_pass_module_changed(bool module_changed) {
    return MutateCurrentHloPassMetadata(
        [&module_changed](HloPassMetadata* pass_metadata) {
          pass_metadata->set_module_changed(module_changed);
        });
  }
  absl::Status set_current_pass_module_id(int64_t module_id) {
    return MutateCurrentHloPassMetadata(
        [&module_id](HloPassMetadata* pass_metadata) {
          pass_metadata->set_module_id(module_id);
        });
  }
  absl::Status add_current_pass_module_group_module_id(int64_t module_id) {
    return MutateCurrentHloPassMetadata(
        [&module_id](HloPassMetadata* pass_metadata) {
          pass_metadata->add_module_group_module_ids(module_id);
        });
  }

  // Clears all pass metadata.
  void ClearPassMetadata() { module_metadata_.clear_pass_metadata(); }

 private:
  // Gets mutable metadata for the currently running pass. If passes are nested,
  // finds the deepest one still running. Returns NotFound if metadata for the
  // currently running pass cannot be found.
  absl::StatusOr<HloPassMetadata*> GetCurrentHloPassMetadata();

  void CopyFrom(const HloModuleMetadata& other) {
    module_metadata_ = other.module_metadata_;
    env_ = other.env_;
    next_pass_id_ = other.next_pass_id_;
    absl::flat_hash_map<const HloPassMetadata*, int64_t> ptr_to_index;
    for (int64_t i = 0; i < other.module_metadata_.pass_metadata().size();
         ++i) {
      ptr_to_index[&other.module_metadata_.pass_metadata(i)] = i;
    }
    running_passes_.reserve(other.running_passes_.size());
    for (HloPassMetadata* pass_metadata : other.running_passes_) {
      running_passes_.push_back(
          module_metadata_.mutable_pass_metadata(ptr_to_index[pass_metadata]));
    }
    prepartitioning_metadata_ = other.prepartitioning_metadata_;
  }

  absl::Status MutateCurrentHloPassMetadata(
      absl::FunctionRef<void(HloPassMetadata*)> mutator);

  HloModuleMetadataProto module_metadata_;
  tsl::Env* env_;
  int64_t next_pass_id_ = 1;

  // Stack of metadata for passes that are currently running. Size > 1 iff
  // passes are nested.
  std::vector<HloPassMetadata*> running_passes_;

  // Metadata from before the module was partitioned, if applicable.
  std::optional<HloModuleMetadataProto> prepartitioning_metadata_ =
      std::nullopt;
};

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_MODULE_METADATA_H_
