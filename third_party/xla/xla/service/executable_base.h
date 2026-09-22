/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_EXECUTABLE_BASE_H_
#define XLA_SERVICE_EXECUTABLE_BASE_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A given platform's compiler will produce an Executable -- this is a uniform
// interface that is used for launching compiled programs across platforms.
class ExecutableBase {
 public:
  // The hlo_module parameter may be nullptr, if the given executable type
  // doesn't need it for execution.
  explicit ExecutableBase(std::shared_ptr<HloModule> hlo_module)
      : hlo_module_(std::move(hlo_module)) {}
  virtual ~ExecutableBase() = default;

  HloModule& module() const {
    CHECK(hlo_module_ != nullptr);
    return *hlo_module_;
  }
  std::shared_ptr<HloModule> shared_module() const { return hlo_module_; }

  bool has_module() const { return hlo_module_ != nullptr; }

  const HloModuleConfig& module_config() const {
    CHECK(hlo_module_ != nullptr);
    return hlo_module_->config();
  }

  // The shape (including layout) that results from this execution. This is the
  // shape of the DeviceAddressBase result value in ExecuteOnStream above.
  virtual Shape result_shape() const {
    CHECK(hlo_module_ != nullptr);
    return hlo_module_->config().entry_computation_layout().result_shape();
  }

  virtual ComputationLayout compute_computation_layout() const {
    CHECK(hlo_module_ != nullptr);
    return hlo_module_->compute_computation_layout();
  }

  virtual absl::string_view name() const {
    if (has_module()) {
      return module().name();
    }
    return "<unknown executable>";
  }

  // Returns the size of the executable in bytes. Returns -1 if this query is
  // not supported by the executable.
  //
  // Does not include the size of used libraries (e.g. cuDNN, Eigen, etc.).
  virtual int64_t SizeOfGeneratedCodeInBytes() const;

  // Dumping helpers.
  void set_hlo_proto(std::unique_ptr<xla::HloProto> hlo_proto) {
    // Despite the mutex lock, this function is NOT thread-safe.
    // The mutex is needed for the lazy HLO module loading in `hlo_proto()`.
    // Since both `hlo_proto()` and `buffer_assignment_proto()` return a
    // pointer to hlo_proto_, having the mutex is not enough to make this
    // function thread-safe.
    absl::MutexLock lock(hlo_proto_mutex_);
    hlo_proto_ = std::move(hlo_proto);
  }
  bool dumping_snapshot() const {
    return has_module()
               ? module_config().debug_options().xla_dump_hlo_snapshots()
               : false;
  }

  HloProto const* hlo_proto() const {
    absl::MutexLock lock(hlo_proto_mutex_);
    if (hlo_proto_ != nullptr && !hlo_proto_->has_hlo_module()) {
      *hlo_proto_->mutable_hlo_module() = module().ToProto();
    }
    return hlo_proto_.get();
  }

  const BufferAssignmentProto* buffer_assignment_proto() const {
    absl::MutexLock lock(hlo_proto_mutex_);
    return hlo_proto_ != nullptr && hlo_proto_->has_buffer_assignment()
               ? &hlo_proto_->buffer_assignment()
               : nullptr;
  }

 private:
  // HloModule this was compiled from. BufferAssignment keeps pointers to
  // HloInstructions owned by the HloModule so we need to keep the HloModule
  // around if we keep the BufferAssignment around.
  //
  // This member may be nullptr, if the given executable type doesn't need it
  // for execution.
  std::shared_ptr<HloModule> hlo_module_;

  // The serialized HLO proto. Non-null only if dumping snapshots is enabled.
  // This field may also be only partially set: if only
  // hlo_proto_->buffer_assignment is set and hlo_proto_->hlo_module isn't, the
  // hlo_module proto will be computed on the fly when requested with
  // hlo_proto(). This avoids wasting CPU and memory if the proto isn't needed.
  std::unique_ptr<HloProto> hlo_proto_ ABSL_GUARDED_BY(hlo_proto_mutex_);
  mutable absl::Mutex hlo_proto_mutex_;
};

}  // namespace xla

#endif  // XLA_SERVICE_EXECUTABLE_BASE_H_
