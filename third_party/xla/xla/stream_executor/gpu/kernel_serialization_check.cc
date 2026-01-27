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

#include "xla/stream_executor/gpu/kernel_serialization_check.h"

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/kernel_symbol_registry.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
using ::testing::IsEmpty;
using ::testing::NotNull;

void VerifyKernelIsSerializable(const KernelLoaderSpec& kernel_spec,
                                Platform::Id platform_id) {
  auto resolve_kernel_symbol =
      [&](absl::string_view persistent_kernel_name) -> absl::StatusOr<void*> {
    return KernelSymbolRegistry::GetGlobalInstance().FindSymbol(
        persistent_kernel_name, platform_id);
  };

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpecProto proto, kernel_spec.ToProto());

  TF_ASSERT_OK_AND_ASSIGN(
      KernelLoaderSpec deserialized_spec,
      KernelLoaderSpec::FromProto(proto, resolve_kernel_symbol));

  if (deserialized_spec.has_in_process_symbol()) {
    EXPECT_THAT(deserialized_spec.in_process_symbol()->symbol, NotNull());
  }
  if (deserialized_spec.has_cuda_cubin_in_memory()) {
    EXPECT_THAT(deserialized_spec.cuda_cubin_in_memory()->cubin_bytes,
                Not(IsEmpty()));
  }
  if (deserialized_spec.has_cuda_ptx_in_memory()) {
    EXPECT_THAT(deserialized_spec.cuda_ptx_in_memory()->ptx, Not(IsEmpty()));
  }
}

}  // namespace stream_executor::gpu
