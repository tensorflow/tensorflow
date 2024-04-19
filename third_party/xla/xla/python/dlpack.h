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

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/py_client.h"

namespace xla {

// If take_ownership is true, ownership of the buffer is handed to DLPack, and
// the receiver may mutate the buffer as they see fit. Otherwise PjRt retains
// ownership of the buffer and it should be immutable.
//
// stream, if set, is a GPU stream, e.g. cudaStream_t for CUDA GPUs, that should
// be synchronized to the buffer as per
// https://dmlc.github.io/dlpack/latest/python_spec.html#python-specification-for-dlpack.
absl::StatusOr<nanobind::capsule> BufferToDLPackManagedTensor(
    nanobind::handle buffer, std::optional<std::intptr_t> stream);

absl::StatusOr<nanobind::object> DLPackManagedTensorToBuffer(
    const nanobind::capsule& tensor,
    std::optional<nb_class_ptr<PyClient>> cpu_client,
    std::optional<nb_class_ptr<PyClient>> gpu_client);

absl::StatusOr<nanobind::object> DLPackManagedTensorToBuffer(
    const nanobind::capsule& tensor, ifrt::Device* device,
    nb_class_ptr<PyClient> client, std::optional<std::intptr_t> stream);

}  // namespace xla

#endif  // XLA_PYTHON_DLPACK_H_
