/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_PJRT_EXECUTABLE_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_PJRT_EXECUTABLE_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/host_offloading/host_offloading_buffer.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// Host offloading executable implemented on top of PjRt CPU executable.
//
// This implementation allows host execution via the PjRT API which allows
// implementing host offloading via dynamic linking of PjRRT plugins.
class HostOffloadingPjRtExecutable : public HostOffloadingExecutable {
 public:
  // Creates a host offloading executable from a proto. Returns an error
  // if PjRt client can't compile the given computation.
  static absl::StatusOr<std::unique_ptr<HostOffloadingPjRtExecutable>>
  LoadFromProto(const HostOffloadingExecutableProto& proto);

  tsl::AsyncValueRef<ExecuteEvent> Execute(
      absl::Span<const ShapeTree<HostOffloadingBuffer>> parameters,
      const xla::ShapeTree<HostOffloadingBuffer>& result,
      const ExecuteOptions& execute_options) final;

  absl::string_view name() const final { return name_; }

  const ProgramShape& program_shape() const final { return program_shape_; }

  bool needs_layout_conversion() const final {
    return needs_layout_conversion_;
  }

 private:
  explicit HostOffloadingPjRtExecutable(
      std::string name, ProgramShape program_shape,
      HloInputOutputAliasConfig alias_config,
      std::unique_ptr<PjRtLoadedExecutable> executable,
      bool needs_layout_conversion);

  std::string name_;
  ProgramShape program_shape_;
  HloInputOutputAliasConfig alias_config_;
  std::unique_ptr<PjRtLoadedExecutable> executable_;
  const bool needs_layout_conversion_;
};

}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_PJRT_EXECUTABLE_H_
