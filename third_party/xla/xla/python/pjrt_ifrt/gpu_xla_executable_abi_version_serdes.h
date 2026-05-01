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

#ifndef XLA_PYTHON_PJRT_IFRT_GPU_XLA_EXECUTABLE_ABI_VERSION_SERDES_H_
#define XLA_PYTHON_PJRT_IFRT_GPU_XLA_EXECUTABLE_ABI_VERSION_SERDES_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/python/ifrt/serdes.h"

namespace xla {

// IFRT SerDes implementation for XlaExecutableAbiVersion on GPU.
class GpuXlaExecutableAbiVersionSerDes
    : public llvm::RTTIExtends<GpuXlaExecutableAbiVersionSerDes,
                               xla::ifrt::SerDes> {
 public:
  using FactoryFunction = std::function<
      absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>(
          const xla::PjRtExecutableAbiVersionProto&)>;

  explicit GpuXlaExecutableAbiVersionSerDes(FactoryFunction factory_function)
      : factory_function_(std::move(factory_function)) {}

  absl::string_view type_name() const override {
    return "xla::GpuXlaExecutableAbiVersion";
  }

  absl::StatusOr<std::string> Serialize(
      const xla::ifrt::Serializable& serializable,
      std::unique_ptr<xla::ifrt::SerializeOptions> options) override;

  absl::StatusOr<std::unique_ptr<xla::ifrt::Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<xla::ifrt::DeserializeOptions> options) override;

  static char ID;  // NOLINT

 private:
  FactoryFunction factory_function_;
};

}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_GPU_XLA_EXECUTABLE_ABI_VERSION_SERDES_H_
