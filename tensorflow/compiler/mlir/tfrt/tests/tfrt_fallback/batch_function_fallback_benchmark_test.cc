/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "testing/base/public/benchmark.h"
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/util/fallback_test_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/rc_array.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace {

// Creates a BEF file with a program that runs
// tfrt_fallback_async.batch_function with a empty function forwarding inputs or
// outputs.
std::pair<tfrt::BefBuffer, tfrt::RCReference<tfrt::BEFFile>> CreateBefFile(
    tfrt::HostContext* host) {
  std::string file_path = GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/tfrt_fallback/"
      "batch_function_fallback.mlir.bef");

  std::string data;
  CHECK_OK(ReadFileToString(Env::Default(), file_path, &data));

  tfrt::BefBuffer bef_buffer(data.begin(), data.end());

  auto bef_file = tfrt::BEFFile::Open(bef_buffer, host->GetKernelRegistry(),
                                      host->diag_handler(), host->allocator());
  CHECK(bef_file);
  return std::make_pair(std::move(bef_buffer), std::move(bef_file));
}

std::unique_ptr<tfrt::CoreRuntime> CreateTestCoreRuntime() {
  auto corert = tfrt::CoreRuntime::Create(
      /*diag_handler=*/[](const tfrt::DecodedDiagnostic&
                              diag) { LOG(ERROR) << diag.message(); },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(16, 16));
  CHECK(corert);
  auto fallback_op_handler = tensorflow::tfd::CreateKernelFallbackOpHandler(
      corert->get(), corert->get()->GetHostContext()->GetHostDeviceRef());
  CHECK(fallback_op_handler);
  corert.get()->RegisterOpHandler("tfkernel", fallback_op_handler.get());
  return std::move(corert.get());
}

tfrt::RCArray<tfrt::AsyncValue> CreateTestArguments(const tfrt::Function* func,
                                                    tfrt::HostContext* host) {
  Tensor tensor(DataType::DT_INT32, TensorShape({1}));
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> arguments;
  arguments.reserve(func->argument_types().size());
  arguments.push_back(tfrt::GetReadyChain());
  for (int i = 1, e = func->argument_types().size(); i < e; ++i) {
    arguments.push_back(
        tfrt::MakeAvailableAsyncValueRef<tfrt_stub::FallbackTensor>(tensor));
  }

  return tfrt::RCArray<tfrt::AsyncValue>(arguments);
}

TEST(BatchFunctionTest, Basic) {
  auto corert = CreateTestCoreRuntime();
  auto* host = corert->GetHostContext();
  auto [bef_buffer, bef_file] = CreateBefFile(host);
  auto* func = bef_file->GetFunction("main");
  CHECK(func);
  CHECK_EQ(func->result_types().size(), 113);
  CHECK_EQ(func->argument_types().size(), 113);

  auto arguments = CreateTestArguments(func, host);

  tfrt::ResourceContext resource_ctx;
  auto exec_ctx = tfd::CreateFallbackTestExecutionContext(host, &resource_ctx);

  std::vector<tfrt::RCReference<tfrt::AsyncValue>> results;
  results.resize(func->result_types().size());
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> result_tensors;
  result_tensors.resize(func->result_types().size() - 1);

  func->Execute(exec_ctx, arguments.values(), results);
  host->Await(results);

  for (auto& result : results) {
    EXPECT_FALSE(result->IsError());
  }
}

// Runs a BEF function that batches a function that does nothing just to measure
// the runtime overhead. The BEF function signature is adapted from a real model
// and is useful for benchmarking ops with large attributes and many
// input/output.
void BM_BatchFunctionFallbackWithLargeAttributesAndManyInputsOutputs(
    benchmark::State& state) {
  auto corert = CreateTestCoreRuntime();
  auto* host = corert->GetHostContext();
  auto [bef_buffer, bef_file] = CreateBefFile(host);
  auto* func = bef_file->GetFunction("main");
  CHECK(func);
  CHECK_EQ(func->result_types().size(), 113);
  CHECK_EQ(func->argument_types().size(), 113);

  auto arguments = CreateTestArguments(func, host);

  tfrt::ResourceContext resource_ctx;
  auto exec_ctx = tfd::CreateFallbackTestExecutionContext(host, &resource_ctx);

  std::vector<tfrt::RCReference<tfrt::AsyncValue>> results;
  results.resize(func->result_types().size());

  for (auto _ : state) {
    func->Execute(exec_ctx, arguments.values(), results);
    host->Await(results);
    results.clear();
    results.resize(func->result_types().size());
  }
}

BENCHMARK(BM_BatchFunctionFallbackWithLargeAttributesAndManyInputsOutputs);

}  // namespace
}  // namespace tensorflow
