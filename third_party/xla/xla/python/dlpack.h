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

#ifndef XLA_PYTHON_DLPACK_H_
#define XLA_PYTHON_DLPACK_H_

#include <memory>

#include "pybind11/pybind11.h"  // from @pybind11
#include "xla/python/py_buffer.h"
#include "xla/python/py_client.h"

namespace xla {

// If take_ownership is true, ownership of the buffer is handed to DLPack, and
// the receiver may mutate the buffer as they see fit. Otherwise PjRt retains
// ownership of the buffer and it should be immutable.
//
// stream, if set, is a GPU stream, e.g. cudaStream_t for CUDA GPUs, that should
// be synchronized to the buffer as per
// https://dmlc.github.io/dlpack/latest/python_spec.html#python-specification-for-dlpack.
absl::StatusOr<pybind11::capsule> BufferToDLPackManagedTensor(
    pybind11::handle buffer, std::optional<std::intptr_t> stream);

absl::StatusOr<pybind11::object> DLPackManagedTensorToBuffer(
    const pybind11::capsule& tensor, std::shared_ptr<PyClient> cpu_client,
    std::shared_ptr<PyClient> gpu_client);

absl::StatusOr<pybind11::object> DLPackManagedTensorToBuffer(
    const pybind11::capsule& tensor, PjRtDevice* device,
    std::shared_ptr<PyClient> client, std::optional<std::intptr_t> stream);

}  // namespace xla

#endif  // XLA_PYTHON_DLPACK_H_
