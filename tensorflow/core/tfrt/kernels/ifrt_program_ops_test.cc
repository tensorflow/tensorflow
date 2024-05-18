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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/framework/serving_device_selector.h"
#include "tsl/framework/test_util/mock_serving_device_selector.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

using tensorflow::ifrt_serving::IfrtLoadedVariableRegistry;
using tensorflow::ifrt_serving::IfrtRestoreTensorRegistry;
using tensorflow::ifrt_serving::IfrtServingCoreSelector;
using tensorflow::ifrt_serving::IfrtServingExecutable;
using tensorflow::ifrt_serving::ServingExecutableRegistry;
using tensorflow::test::AsTensor;
using tensorflow::test::TensorEq;
using ::testing::Return;

const bool kUnused =
    (xla::ifrt::test_util::RegisterClientFactory(
         []() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
           xla::CpuClientOptions options;
           options.cpu_device_count = 4;
           TF_ASSIGN_OR_RETURN(auto pjrt_client,
                               xla::GetTfrtCpuClient(options));
           return std::shared_ptr<xla::ifrt::Client>(
               xla::ifrt::PjRtClient::Create(std::move(pjrt_client)));
         }),
     true);

class IfrtCallOpTest : public OpsTestBase {
 protected:
  Status Init(int64_t program_id, int num_inputs, DataType input_type,
              const std::vector<int>& variable_arg_indices,
              const std::vector<DataType>& output_type_list) {
    TF_CHECK_OK(NodeDefBuilder("op", "IfrtCall")
                    .Input(FakeInput(num_inputs, input_type))
                    .Attr("program_id", program_id)
                    .Attr("variable_arg_indices", variable_arg_indices)
                    .Attr("Tout", output_type_list)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(IfrtCallOpTest, Basic) {
  int64_t program_id = 123;
  TF_ASSERT_OK(Init(
      /*program_id=*/program_id,
      /*num_inputs=*/2,
      /*input_type=*/DT_INT32,
      /*variable_arg_indices=*/{},
      /*output_type_list=*/{DT_INT32}));

  // TODO(b/324451111): Extract the executable creation to a common function
  // to be shared by all tests (e.g.
  // tensorflow/core/tfrt/ifrt/ifrt_serving_executable_test.cc)
  tsl::test_util::MockServingDeviceSelector selector;
  EXPECT_CALL(selector, ReserveDevice(absl::StrCat(program_id)))
      .Times(1)
      .WillOnce(Return(tsl::DeviceReservation(0, /*selector=*/nullptr)));

  auto core_selector = std::make_unique<IfrtServingCoreSelector>(&selector);

  constexpr absl::string_view kDataDirectory =
      "tensorflow/core/tfrt/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/executable.mlir"));
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  auto mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());

  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtSharding",
      kMaxParallelism);

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(auto device_mgr,
                          ifrt_serving::CreateTfStaticDeviceMgr());

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  IfrtRestoreTensorRegistry ifrt_restore_tensor_registry;

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      IfrtServingExecutable::Create(
          program_id, "test", "main", std::move(mlir_module), client,
          thread_pool.get(), &ifrt_loaded_variable_registry,
          &ifrt_restore_tensor_registry, work_queue.get(), device_mgr_.get(),
          tensorflow::IdentityShapeRepresentationFn(), core_selector.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      ServingExecutableRegistry::Handle handle,
      ServingExecutableRegistry::Register(program_id, std::move(executable)));
  auto handle_cleaner = gtl::MakeCleanup([&handle] { handle.Release(); });

  AddInputFromArray<int32_t>(TensorShape({1, 3}), {1, 2, 3});
  AddInputFromArray<int32_t>(TensorShape({3, 1}), {1, 2, 3});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected_out = AsTensor<int32_t>({14}, TensorShape({1, 1}));
  EXPECT_THAT(*GetOutput(0), TensorEq(expected_out));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
