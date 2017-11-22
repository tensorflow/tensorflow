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

#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class Bfloat16Test : public ClientLibraryTestBase {
 protected:
  const ErrorSpec error_spec_{0.001, 0.001};
};

XLA_TEST_F(Bfloat16Test, DISABLED_ON_GPU(DISABLED_ON_CPU_PARALLEL(
                             DISABLED_ON_CPU(ScalarOperation)))) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR0<bfloat16>(static_cast<bfloat16>(2.0f));
  auto y = builder.ConstantR0<bfloat16>(static_cast<bfloat16>(1.0f));
  builder.Add(x, y);

  ComputeAndCompareR0<bfloat16>(&builder, static_cast<bfloat16>(3.0f), {},
                                error_spec_);
}

XLA_TEST_F(Bfloat16Test, DISABLED_ON_GPU(DISABLED_ON_CPU_PARALLEL(
                             DISABLED_ON_CPU(NegateScalarF16)))) {
  ComputationBuilder builder(client_, TestName());
  builder.Neg(builder.ConstantR0<bfloat16>(static_cast<bfloat16>(2.1f)));

  ComputeAndCompareR0<bfloat16>(&builder, static_cast<bfloat16>(-2.1f), {},
                                error_spec_);
}

}  // namespace
}  // namespace xla
