/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class DeallocationTest : public ClientLibraryTestBase {
 protected:
  // Build and execute the given computation then verify the results can be
  // transferred from the device successfully.
  std::unique_ptr<GlobalData> ExecuteAndCheckTransfer(
      ComputationBuilder* builder,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
    Computation computation = builder->Build().ConsumeValueOrDie();
    auto global_data =
        client_->Execute(computation, arguments).ConsumeValueOrDie();
    TF_CHECK_OK(client_->Transfer(*global_data).status());
    return global_data;
  }
};

TEST_F(DeallocationTest, DeallocateScalar) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR0<float>(42.0);
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  // A result can be transfered an arbitrary number of times.  Add an extra
  // transfer here so we're not just testing that a second call to Transfer
  // fails.
  ASSERT_IS_OK(client_->Transfer(*global_data).status());

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_MATCH(transfer_status.status().error_message(),
               testing::HasSubstr("was previously deallocated"));
}

TEST_F(DeallocationTest, DeallocateVector) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>({1.0, 2.0, 3.0, 4.0});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_MATCH(transfer_status.status().error_message(),
               testing::HasSubstr("was previously deallocated"));
}

TEST_F(DeallocationTest, DeallocateEmptyVector) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>({});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_MATCH(transfer_status.status().error_message(),
               testing::HasSubstr("was previously deallocated"));
}

XLA_TEST_F(DeallocationTest, DeallocateTuple) {
  ComputationBuilder builder(client_, TestName());
  builder.Tuple({builder.ConstantR0<float>(42.0),
                 builder.ConstantR1<float>({1.0, 2.0, 3.0})});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_MATCH(transfer_status.status().error_message(),
               testing::HasSubstr("was previously deallocated"));
}

XLA_TEST_F(DeallocationTest, DeallocateTupleWithRepeatedElements) {
  ComputationBuilder builder(client_, TestName());
  auto element = builder.ConstantR0<float>(42.0);
  auto inner_tuple = builder.Tuple({builder.ConstantR0<float>(42.0), element});
  builder.Tuple({element, inner_tuple, element});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_MATCH(transfer_status.status().error_message(),
               testing::HasSubstr("was previously deallocated"));
}

XLA_TEST_F(DeallocationTest, DeallocateNestedTuple) {
  ComputationBuilder builder(client_, TestName());
  auto inner_tuple =
      builder.Tuple({builder.ConstantR0<float>(42.0),
                     builder.ConstantR1<float>({1.0, 2.0, 3.0})});
  builder.Tuple({inner_tuple, builder.ConstantR1<float>({0.123, 0.456})});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_MATCH(transfer_status.status().error_message(),
               testing::HasSubstr("was previously deallocated"));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
