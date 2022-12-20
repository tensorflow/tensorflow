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
#include "tensorflow/core/tfrt/graph_executor/synchronous_graph_executor.h"

#include <memory>
#include <utility>
#include <vector>

#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel/kernel.h"
#include "learning/brain/experimental/tfrt/native_lowering/kernels/math_kernels.h"
#include "learning/brain/experimental/tfrt/native_lowering/kernels/sync_fallback_kernels.h"
#include "learning/infra/mira/mlrt/interpreter/value.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tfrt/cpp_tests/test_util.h""  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

absl::Status GetSimpleGraphDef(GraphDef& graph_def) {
  auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

  auto input = ops::Placeholder(scope.WithOpName("input"), DT_INT32);
  auto rank = ops::Rank(scope.WithOpName("rank"), input);

  return tfrt::AbslStatusFromTfStatus(scope.ToGraphDef(&graph_def));
}

TEST(TfrtSynchronousSessionTest, Sanity) {
  tensorflow::GraphDef graph_def;
  ASSERT_OK(GetSimpleGraphDef(graph_def));

  auto kernel_registry = std::make_unique<mlrt::KernelRegistry>();
  tensorflow::tf_mlrt::RegisterTfMlrtKernels(*kernel_registry);
  tfrt::cpu::RegisterMlrtMathKernels(kernel_registry.get());
  tfrt::cpu::RegisterMlrtFallbackCompatKernels(kernel_registry.get());

  ASSERT_OK_AND_ASSIGN(
      auto session,
      SynchronousGraphExecutor::Create(graph_def, std::move(kernel_registry)));

  std::vector<mlrt::Value> inputs;
  tfrt::DenseHostTensor dht =
      tfrt::CreateTensorFromValues<int32_t>({1, 3}, {1, 1, 1});
  inputs.emplace_back(std::move(dht));
  std::vector<mlrt::Value> results;
  results.resize(1);

  ASSERT_OK(session->Run("test_graph", absl::Span<mlrt::Value>(inputs),
                         /*input_names=*/{"input"}, /*input_dtypes=*/{DT_INT32},
                         /*output_tensor_names=*/{"rank"},
                         /*target_tensor_names=*/{},
                         absl::Span<mlrt::Value>(results)));
  tfrt::DenseHostTensor expected =
      tfrt::CreateTensorFromValues<int32_t>({}, {2});

  EXPECT_EQ(expected, results[0].Get<tfrt::DenseHostTensor>());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
