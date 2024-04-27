/* Copyright 2022 The OpenXLA Authors.

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

#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/numbers.h"
#include "tsl/platform/errors.h"
#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#else
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/host_callback.h"
#include "xla/primitive_util.h"
#include "xla/python/callback.h"
#include "xla/python/nb_numpy.h"

#if TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
#define gpuStreamHandle hipStream_t
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#else
#define gpuSuccess cudaSuccess
#define gpuStreamHandle CUstream
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#endif

namespace nb = nanobind;

namespace xla {

void XlaPythonGpuCallback(gpuStreamHandle stream, void** buffers,
                          const char* opaque, size_t opaque_len,
                          XlaCustomCallStatus* status) {
  // Ignore `descriptor` arg to callback
  buffers += 1;
  uint64_t descriptor;
  if (!absl::SimpleAtoi(opaque, &descriptor)) {
    throw xla::XlaRuntimeError("Invalid callback descriptor");
    return;
  }
  CpuCallback* callback =
      absl::bit_cast<CpuCallback*>(static_cast<uintptr_t>(descriptor));
  size_t arity = callback->num_args();
  std::vector<void*> host_input_buffers(arity);
  // Copy input GPU buffers to host
  for (size_t i = 0; i < arity; ++i) {
    const CpuCallback::Arg& arg = callback->args()[i];
    if (arg.type == TOKEN) {
      host_input_buffers[i] = nullptr;
      continue;
    }
    void* buf = new char[arg.size_in_bytes];
    host_input_buffers[i] = buf;
    // TODO(b/238441608): Use pinned memory here to speed up the transfer.
    auto gpu_res = gpuMemcpyAsync(buf, buffers[i], arg.size_in_bytes,
                                  gpuMemcpyDeviceToHost, stream);
    CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
  }
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  nb::gil_scoped_acquire gil;
  nb::tuple host_input_arrays = nb::steal<nb::tuple>(PyTuple_New(arity));
  for (size_t i = 0; i < arity; ++i) {
    CpuCallback::Arg arg = callback->args()[i];
    if (arg.type == TOKEN) {
      PyTuple_SET_ITEM(host_input_arrays.ptr(), i, nb::none().inc_ref().ptr());
      continue;
    }
    nb::capsule base(host_input_buffers[i], [](void* ptr) noexcept {
      delete[] static_cast<char*>(ptr);
    });
    auto array = nb_numpy_ndarray(arg.dtype, arg.dims, arg.strides,
                                  const_cast<void*>(host_input_buffers[i]),
                                  /*base=*/base);
    array.attr("flags").attr("writeable") = nb::bool_(false);
    PyTuple_SET_ITEM(host_input_arrays.ptr(), i, array.inc_ref().ptr());
  }
  EnterHostCallback();
  std::optional<nb::tuple> maybe_result_tuple =
      callback->Call(host_input_arrays, status);
  LeaveHostCallback();
  if (!maybe_result_tuple) {
    return;
  }
  nb::tuple result_tuple = maybe_result_tuple.value();
  std::vector<void*> temp_buffers;
  for (size_t i = 0; i < callback->results().size(); ++i) {
    CpuCallback::Result result = callback->results()[i];
    if (result.type == TOKEN) {
      continue;
    }
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    nb_numpy_ndarray array = nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    if (strides == result.expected_strides) {
      auto gpu_res =
          gpuMemcpyAsync(buffers[arity + i], array.data(), result.size_in_bytes,
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
    } else {
      void* temp = new char[result.size_in_bytes];
      temp_buffers.push_back(temp);
      xla::TransposePlan::Options options;
      options.elem_size_in_bytes = xla::primitive_util::ByteWidth(result.type);
      options.dims = dims;
      options.permutation = result.reversed_layout;
      options.input_layout = xla::TransposePlan::Striding{strides};
      absl::StatusOr<std::shared_ptr<xla::TransposePlan>> plan =
          callback->transpose_cache().GetOrCreate(options);
      if (!plan.ok()) {
        throw xla::XlaRuntimeError(plan.status().ToString());
      }
      plan.value()->Execute(array.data(), temp);
      auto gpu_res =
          gpuMemcpyAsync(buffers[arity + i], temp, result.size_in_bytes,
                         gpuMemcpyHostToDevice, stream);
      CHECK_EQ(gpu_res, gpuSuccess) << "Failed to gpuMemcpyAsync";
    }
  }
  nb::gil_scoped_release release;
  CHECK_EQ(gpuStreamSynchronize(stream), gpuSuccess)
      << "Failed to gpuStreamSynchronize";
  for (int i = 0; i < temp_buffers.size(); ++i) {
    delete[] static_cast<char*>(temp_buffers[i]);
  }
}

}  // namespace xla
