/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_TESTS_EXECUTABLE_IMPL_TEST_BASE_H_
#define XLA_PYTHON_IFRT_IR_TESTS_EXECUTABLE_IMPL_TEST_BASE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Base class to help create tests that compile and execute IFRT IR.
class IfrtIrExecutableImplTestBase : public testing::Test {
 public:
  IfrtIrExecutableImplTestBase();
  void SetUp() override;

 protected:
  // Loads mlir from source string.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadFromSource(
      absl::string_view source);

  // Loads mlir from file.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadFromFile(
      absl::string_view file_path);

  // Creates an Array from per shard data.
  // TODO(hyeontaek): Remove this when MakeArrayFromHostBuffer supports it
  // directly.
  absl::StatusOr<tsl::RCReference<Array>> CreateArray(
      absl::Span<void* const> per_shard_data, Shape shape, DType dtype,
      ShardingParam sharding_param, DeviceList device_list);

  // Picks a given number of devices.
  // Error when `count` is larger than the total number of devices.
  absl::StatusOr<DeviceList> PickDevices(int count);

  mlir::MLIRContext mlir_context_;
  std::shared_ptr<Client> client_;
};

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TESTS_EXECUTABLE_IMPL_TEST_BASE_H_
