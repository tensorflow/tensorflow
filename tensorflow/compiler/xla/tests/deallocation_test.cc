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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class DeallocationTest : public ClientLibraryTestBase {
 protected:
  // Build and execute the given computation then verify the results can be
  // transferred from the device successfully.
  std::unique_ptr<GlobalData> ExecuteAndCheckTransfer(
      XlaBuilder* builder, absl::Span<GlobalData* const> arguments) {
    XlaComputation computation = builder->Build().ConsumeValueOrDie();
    auto global_data =
        client_->Execute(computation, arguments, &execution_options_)
            .ConsumeValueOrDie();
    TF_CHECK_OK(client_->Transfer(*global_data).status());
    return global_data;
  }
};

TEST_F(DeallocationTest, DeallocateScalar) {
  XlaBuilder builder(TestName());
  ConstantR0<float>(&builder, 42.0);
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  // A result can be transferred an arbitrary number of times.  Add an extra
  // transfer here so we're not just testing that a second call to Transfer
  // fails.
  ASSERT_IS_OK(client_->Transfer(*global_data).status());

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_THAT(transfer_status.status().error_message(),
              HasSubstr("was previously deallocated"));
}

TEST_F(DeallocationTest, DeallocateVector) {
  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, {1.0, 2.0, 3.0, 4.0});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_THAT(transfer_status.status().error_message(),
              HasSubstr("was previously deallocated"));
}

TEST_F(DeallocationTest, DeallocateEmptyVector) {
  XlaBuilder builder(TestName());
  ConstantR1<float>(&builder, {});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_THAT(transfer_status.status().error_message(),
              HasSubstr("was previously deallocated"));
}

XLA_TEST_F(DeallocationTest, DeallocateTuple) {
  XlaBuilder builder(TestName());
  Tuple(&builder, {ConstantR0<float>(&builder, 42.0),
                   ConstantR1<float>(&builder, {1.0, 2.0, 3.0})});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_THAT(transfer_status.status().error_message(),
              HasSubstr("was previously deallocated"));
}

XLA_TEST_F(DeallocationTest, DeallocateTupleWithRepeatedElements) {
  XlaBuilder builder(TestName());
  auto element = ConstantR0<float>(&builder, 42.0);
  auto inner_tuple =
      Tuple(&builder, {ConstantR0<float>(&builder, 42.0), element});
  Tuple(&builder, {element, inner_tuple, element});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_THAT(transfer_status.status().error_message(),
              HasSubstr("was previously deallocated"));
}

XLA_TEST_F(DeallocationTest, DeallocateNestedTuple) {
  XlaBuilder builder(TestName());
  auto inner_tuple =
      Tuple(&builder, {ConstantR0<float>(&builder, 42.0),
                       ConstantR1<float>(&builder, {1.0, 2.0, 3.0})});
  Tuple(&builder, {inner_tuple, ConstantR1<float>(&builder, {0.123, 0.456})});
  auto global_data = ExecuteAndCheckTransfer(&builder, {});

  ASSERT_IS_OK(client_->Unregister(*global_data));

  auto transfer_status = client_->Transfer(*global_data);
  ASSERT_FALSE(transfer_status.ok());
  ASSERT_THAT(transfer_status.status().error_message(),
              HasSubstr("was previously deallocated"));
}

}  // namespace
}  // namespace xla
