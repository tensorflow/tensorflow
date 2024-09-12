/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/log.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/lite/experimental/shlo/legacy/bench/util.h"
#include "tensorflow/lite/experimental/shlo/legacy/src/f16.h"

namespace stablehlo {
namespace benchmark {
namespace {

template <xnn_datatype datatype>
struct Storage;

template <>
struct Storage<xnn_datatype_fp32> {
  using Type = float;
};

template <>
struct Storage<xnn_datatype_fp16> {
  using Type = F16;
};

size_t size_in_bytes(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_qint32:
    case xnn_datatype_qcint32:
    case xnn_datatype_fp32:
      return 4;
    case xnn_datatype_fp16:
      return 2;
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qcint8:
      return 1;
    default:
      // Crash OK
      LOG(FATAL) << "Unexpected datatype: " << static_cast<int>(datatype);
  }
}

size_t tensor_size_in_bytes(xnn_datatype datatype,
                            const std::vector<size_t>& shape) {
  size_t size = size_in_bytes(datatype);
  for (auto s : shape) {
    size *= s;
  }
  return size;
}

using UnaryOp = xnn_status (*)(xnn_subgraph_t subgraph, uint32_t input_id,
                               uint32_t output_id, uint32_t flags);

using BinaryOp = xnn_status (*)(xnn_subgraph_t subgraph, uint32_t input1_id,
                                uint32_t input2_id, uint32_t output_id,
                                uint32_t flags);

using ClampedBinaryOp = xnn_status (*)(xnn_subgraph_t subgraph,
                                       float output_min, float output_max,
                                       uint32_t input1_id, uint32_t input2_id,
                                       uint32_t output_id, uint32_t flags);

template <xnn_datatype datatype, typename Op>
void BM_XNN_HELPER(::benchmark::State& state, Op op, size_t size) {
  size_t num_operands;
  if constexpr (std::is_same_v<Op, UnaryOp>) {
    num_operands = 1;
  } else {
    static_assert(std::is_same_v<Op, BinaryOp> or
                  std::is_same_v<Op, ClampedBinaryOp>);
    num_operands = 2;
  }

  std::vector<size_t> shape = {size};

  using ST = typename Storage<datatype>::Type;

  std::vector<std::vector<ST>> operand_values;
  operand_values.reserve(num_operands);
  for (auto i = 0; i < num_operands; ++i) {
    operand_values.emplace_back(GenerateRandomVector<ST>(size));
  }

  auto size_in_bytes = tensor_size_in_bytes(datatype, shape);
  std::vector<uint8_t> result_values(size_in_bytes);

  std::vector<xnn_external_value> external_values;
  external_values.reserve(num_operands + 1);
  for (auto i = 0; i < num_operands; ++i) {
    external_values.push_back(
        {.id = static_cast<uint32_t>(i), .data = operand_values[i].data()});
  }
  external_values.push_back({.id = static_cast<uint32_t>(num_operands),
                             .data = result_values.data()});

  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    // Crash OK
    LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
  }

  xnn_workspace_t workspace;
  status = xnn_create_workspace(&workspace);
  if (status != xnn_status_success) {
    // Crash OK
    LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
  }

  xnn_subgraph_t subgraph = nullptr;
  auto max_external_value_id =
      std::max_element(external_values.begin(), external_values.end(),
                       [](auto x, auto y) { return x.id < y.id; })
          ->id;
  status =
      xnn_create_subgraph(1 + max_external_value_id, /*flags=*/0, &subgraph);
  if (status != xnn_status_success) {
    // Crash OK
    LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
  }

  for (auto i = 0; i < num_operands; ++i) {
    uint32_t operand_id;
    status = xnn_define_tensor_value(
        subgraph, datatype, shape.size(), shape.data(),
        /* data */ nullptr, /* xnn_external_id */ i,
        /* flags */ XNN_VALUE_FLAG_EXTERNAL_INPUT, &operand_id);
    if (status != xnn_status_success) {
      // Crash OK
      LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
    }
  }

  uint32_t result_id;
  status = xnn_define_tensor_value(
      subgraph, datatype, shape.size(), shape.data(),
      /* data */ nullptr, /* xnn_external_id */ num_operands,
      /* flags */ XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &result_id);
  if (status != xnn_status_success) {
    // Crash OK
    LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
  }

  if constexpr (std::is_same_v<Op, UnaryOp>) {
    status = op(subgraph, 0, 1, /* flags */ 0);
  } else if constexpr (std::is_same_v<Op, BinaryOp>) {
    status = op(subgraph, 0, 1, 2, /* flags */ 0);
  } else {
    static_assert(std::is_same_v<Op, ClampedBinaryOp>);
    status = op(subgraph, /* output_min */ std::numeric_limits<float>::min(),
                /* output_max */ std::numeric_limits<float>::max(), 0, 1, 2,
                /* flags */ 0);
  }
  if (status != xnn_status_success) {
    // Crash OK
    LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
  }

  xnn_runtime_t runtime = nullptr;
  status = xnn_create_runtime_v4(subgraph,
                                 /*weights_buffer_cache*/ nullptr, workspace,
                                 /*threadpool*/ nullptr,
                                 /*flags*/ 0, &runtime);
  if (status != xnn_status_success) {
    // Crash OK
    LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
  }

  for (auto _ : state) {
    status = xnn_setup_runtime(runtime, external_values.size(),
                               external_values.data());
    if (status != xnn_status_success) {
      // Crash OK
      LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
    }
    status = xnn_invoke_runtime(runtime);
    if (status != xnn_status_success) {
      // Crash OK
      LOG(FATAL) << "Failed to invoke XNNPACK runtime: " << status;
    }
  }

  xnn_delete_runtime(runtime);
  xnn_delete_subgraph(subgraph);
  xnn_release_workspace(workspace);
  xnn_deinitialize();
}

template <UnaryOp op, xnn_datatype datatype, size_t size>
void BM_XNN(::benchmark::State& state) {
  BM_XNN_HELPER<datatype>(state, op, size);
}

template <BinaryOp op, xnn_datatype datatype, size_t size>
void BM_XNN(::benchmark::State& state) {
  BM_XNN_HELPER<datatype>(state, op, size);
}

template <ClampedBinaryOp op, xnn_datatype datatype, size_t size>
void BM_XNN(::benchmark::State& state) {
  BM_XNN_HELPER<datatype>(state, op, size);
}

#define BENCHMARK_OP_HELPER(Op, DataType)   \
  BENCHMARK(BM_XNN<Op, DataType, 8 * KB>);  \
  BENCHMARK(BM_XNN<Op, DataType, 16 * KB>); \
  BENCHMARK(BM_XNN<Op, DataType, 32 * KB>); \
  BENCHMARK(BM_XNN<Op, DataType, 64 * KB>);

#define BENCHMARK_OP(Op) BENCHMARK_OP_HELPER(Op, xnn_datatype_fp32);
//  BENCHMARK_OP_HELPER(Op, xnn_datatype_fp16);

BENCHMARK_OP(xnn_define_abs);
BENCHMARK_OP(xnn_define_add2);

}  // namespace
}  // namespace benchmark
}  // namespace stablehlo

// Run the benchmark
BENCHMARK_MAIN();
