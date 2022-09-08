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

#include "absl/base/casts.h"
#include "absl/strings/numbers.h"
#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#else
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/callback.h"
#include "tensorflow/compiler/xla/python/exceptions.h"

#if TENSORFLOW_USE_ROCM
#define gpuStreamHandle hipStream_t
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#else
#define gpuStreamHandle CUstream
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#endif

namespace py = pybind11;

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
    CpuCallback::Arg arg = callback->args()[i];
    if (arg.type == TOKEN) {
      host_input_buffers[i] = nullptr;
      continue;
    }
    void* buf = new char[arg.size_in_bytes];
    host_input_buffers[i] = buf;
    // TODO(b/238441608): Use pinned memory here to speed up the transfer.
    gpuMemcpyAsync(buf, buffers[i], arg.size_in_bytes, gpuMemcpyDeviceToHost,
                   stream);
  }
  gpuStreamSynchronize(stream);
  py::gil_scoped_acquire gil;
  py::tuple host_input_arrays(arity);
  for (size_t i = 0; i < arity; ++i) {
    CpuCallback::Arg arg = callback->args()[i];
    if (arg.type == TOKEN) {
      host_input_arrays[i] = py::none();
      continue;
    }
    py::capsule base(host_input_buffers[i],
                     [](void* ptr) { delete[] static_cast<char*>(ptr); });
    host_input_arrays[i] =
        py::array(arg.dtype, arg.dims, arg.strides,
                  const_cast<void*>(host_input_buffers[i]), /*base=*/base);
    host_input_arrays[i].attr("flags").attr("writeable") = Py_False;
  }
  std::optional<py::tuple> maybe_result_tuple =
      callback->Call(host_input_arrays, status);
  if (!maybe_result_tuple) {
    return;
  }
  py::tuple result_tuple = maybe_result_tuple.value();
  std::vector<void*> temp_buffers;
  for (size_t i = 0; i < callback->results().size(); ++i) {
    CpuCallback::Result result = callback->results()[i];
    if (result.type == TOKEN) {
      continue;
    }
    py::object output = py::reinterpret_borrow<py::object>(
        PyTuple_GetItem(result_tuple.ptr(), i));
    py::array array = py::cast<py::array>(std::move(output));
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    if (strides == result.expected_strides) {
      gpuMemcpyAsync(buffers[arity + i], array.data(), result.size_in_bytes,
                     gpuMemcpyHostToDevice, stream);
    } else {
      void* temp = new char[result.size_in_bytes];
      temp_buffers.push_back(temp);
      xla::StatusOr<std::shared_ptr<xla::TransposePlan>> plan =
          callback->transpose_cache().GetOrCreate(
              xla::primitive_util::ByteWidth(result.type), dims,
              result.reversed_layout,
              /*input_layout=*/xla::TransposePlan::Striding{strides});
      if (!plan.ok()) {
        throw xla::XlaRuntimeError(plan.status().ToString());
      }
      plan.value()->Execute(array.data(), temp);
      gpuMemcpyAsync(buffers[arity + i], temp, result.size_in_bytes,
                     gpuMemcpyHostToDevice, stream);
    }
  }
  py::gil_scoped_release release;
  gpuStreamSynchronize(stream);
  for (int i = 0; i < temp_buffers.size(); ++i) {
    delete[] static_cast<char*>(temp_buffers[i]);
  }
}

}  // namespace xla
