/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/cub_sort.h"

#include <optional>

#include "absl/status/status.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"  // IWYU pragma: keep
#include "xla/runtime/memref_view.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"

#ifdef GOOGLE_CUDA
#include "xla/service/gpu/cub_sort_thunk.h"
#endif

namespace xla {
namespace gpu {
namespace {

using ::stream_executor::DeviceMemoryBase;
using ::xla::runtime::CustomCall;
using ::xla::runtime::FlatMemrefView;

absl::Status CubDeviceRadixSortKeysImpl(
    const ServiceExecutableRunOptions* run_options, FlatMemrefView input_view,
    FlatMemrefView output_view, FlatMemrefView scratch_view, bool descending) {
#ifdef GOOGLE_CUDA
  return RunCubSort(input_view.dtype, std::nullopt,
                    GetDeviceAddress(input_view), DeviceMemoryBase(),
                    GetDeviceAddress(output_view), DeviceMemoryBase(),
                    GetDeviceAddress(scratch_view), descending);
#else
  return absl::UnimplementedError("CUB is not available");
#endif
}

absl::Status CubDeviceRadixSortPairsImpl(
    const ServiceExecutableRunOptions* run_options,
    FlatMemrefView input_keys_view, FlatMemrefView input_values_view,
    FlatMemrefView output_keys_view, FlatMemrefView output_values_view,
    FlatMemrefView scratch_view, bool descending) {
#ifdef GOOGLE_CUDA
  return RunCubSort(
      input_keys_view.dtype, input_values_view.dtype,
      GetDeviceAddress(input_keys_view), GetDeviceAddress(input_values_view),
      GetDeviceAddress(output_keys_view), GetDeviceAddress(output_values_view),
      GetDeviceAddress(scratch_view), descending);
#else
  return absl::UnimplementedError("CUB is not available");
#endif
}

}  // namespace

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CubDeviceRadixSortKeys, FunctionWrapper<CubDeviceRadixSortKeysImpl>(),
    checks,
    CustomCall::Bind("xla.gpu.radix_sort_keys")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<FlatMemrefView>()  // input
        .Arg<FlatMemrefView>()  // output
        .Arg<FlatMemrefView>()  // scratch
        .Attr<bool>("descending"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CubDeviceRadixSortPairs, FunctionWrapper<CubDeviceRadixSortPairsImpl>(),
    checks,
    CustomCall::Bind("xla.gpu.radix_sort_pairs")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<FlatMemrefView>()  // input_keys
        .Arg<FlatMemrefView>()  // input_values
        .Arg<FlatMemrefView>()  // output_keys
        .Arg<FlatMemrefView>()  // output_values
        .Arg<FlatMemrefView>()  // scratch
        .Attr<bool>("descending"));

void RegisterCubSortCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.radix_sort_keys", CubDeviceRadixSortKeys);
  registry.Register("xla.gpu.radix_sort_pairs", CubDeviceRadixSortPairs);
}

}  // namespace gpu
}  // namespace xla
