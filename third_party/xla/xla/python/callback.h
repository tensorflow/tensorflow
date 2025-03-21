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

#ifndef XLA_PYTHON_CALLBACK_H_
#define XLA_PYTHON_CALLBACK_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "xla/pjrt/transpose.h"
#include "xla/python/nb_numpy.h"
#include "xla/service/custom_call_status.h"
#include "xla/xla_data.pb.h"

namespace xla {

class CpuCallback {
 public:
  struct Arg {
    xla::PrimitiveType type;               // XLA type
    nb_dtype dtype;                        // NumPy type, for array types.
    absl::InlinedVector<int64_t, 4> dims;  // Dimensions, for array types.
    std::vector<int64_t> strides;          // Byte strides, for array types.
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

  explicit CpuCallback(nanobind::callable callable, std::vector<Arg> args,
                       std::vector<Result> results)
      : callable_(std::move(callable)),
        args_(std::move(args)),
        results_(std::move(results)),
        transpose_cache_(/*capacity=*/16) {}

  ~CpuCallback();

  const std::vector<Arg>& args() const { return args_; }
  size_t num_args() const { return args_.size(); }

  const std::vector<Result>& results() const { return results_; }
  size_t num_results() const { return results_.size(); }
  void* callback() const { return callable_.ptr(); }

  xla::TransposePlanCache& transpose_cache() { return transpose_cache_; }

  absl::Status PrepareAndCall(void* result, void** arg_ptrs);

  absl::StatusOr<nanobind::tuple> Call(nanobind::tuple args);

 private:
  nanobind::callable callable_;
  std::vector<Arg> args_;
  std::vector<Result> results_;
  xla::TransposePlanCache transpose_cache_;
};

void XlaPythonCpuCallback(void* output, void** inputs,
                          XlaCustomCallStatus* status);

}  // namespace xla

#endif  // XLA_PYTHON_CALLBACK_H_
