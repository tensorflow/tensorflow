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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_CALLBACK_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_CALLBACK_H_

#include <optional>
#include <utility>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class CpuCallback {
 public:
  struct Arg {
    xla::PrimitiveType type;               // XLA type
    pybind11::dtype dtype;                 // NumPy type, for array types.
    absl::InlinedVector<int64_t, 4> dims;  // Dimensions, for array types.
    std::vector<ssize_t> strides;          // Byte strides, for array types.
    size_t size_in_bytes;                  // Size of the array in bytes.
  };
  struct Result {
    xla::PrimitiveType type;  // XLA type
    // Expected output shape, for array types
    absl::InlinedVector<int64_t, 4> expected_dims;
    // Expected output byte strides, for array types. If the strides do not
    // match the output will be transposed into the expected layout.
    std::vector<int64_t> expected_strides;
    // The desired order of output dimensions in major-to-minor order.
    absl::InlinedVector<int64_t, 4> reversed_layout;
    // Size of the array in bytes.
    size_t size_in_bytes;
  };

  explicit CpuCallback(pybind11::function callable, std::vector<Arg> args,
                       std::vector<Result> results)
      : callable_(std::move(callable)),
        args_(std::move(args)),
        results_(std::move(results)),
        transpose_cache_(/*capacity=*/16) {}

  const std::vector<Arg>& args() const { return args_; }
  size_t num_args() const { return args_.size(); }

  const std::vector<Result>& results() const { return results_; }
  size_t num_results() const { return results_.size(); }

  xla::TransposePlanCache& transpose_cache() { return transpose_cache_; }

  void PrepareAndCall(void* result, void** arg_ptrs,
                      XlaCustomCallStatus* status);
  Status PrepareAndCall(void* result, void** arg_ptrs);

  std::optional<pybind11::tuple> Call(pybind11::tuple args,
                                      XlaCustomCallStatus* status);
  StatusOr<pybind11::tuple> Call(pybind11::tuple args);

 private:
  Status PrepareAndCallInternal(void* result, void** arg_ptrs);
  StatusOr<pybind11::tuple> CallInternal(pybind11::tuple args);

  pybind11::function callable_;
  std::vector<Arg> const args_;
  std::vector<Result> const results_;
  xla::TransposePlanCache transpose_cache_;
};

void XlaPythonCpuCallback(void* output, void** inputs,
                          XlaCustomCallStatus* status);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_CALLBACK_H_
