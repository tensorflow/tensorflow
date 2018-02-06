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

#include "tensorflow/compiler/xla/client/lib/testing.h"

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

std::unique_ptr<GlobalData> MakeFakeDataViaDeviceOrDie(const Shape& shape,
                                                       Client* client) {
  ComputationBuilder b(
      client,
      tensorflow::strings::StrCat("make_fake_", ShapeUtil::HumanString(shape)));
  // TODO(b/26811613): Replace this when RNG is supported on all backends.
  b.Broadcast(b.ConstantLiteral(Literal::One(shape.element_type())),
              AsInt64Slice(shape.dimensions()));
  Computation computation = b.Build().ConsumeValueOrDie();

  auto execution_options = CreateDefaultExecutionOptions();
  *execution_options.mutable_shape_with_output_layout() = shape;
  return client->Execute(computation, /*arguments=*/{}, &execution_options)
      .ConsumeValueOrDie();
}

}  // namespace

std::unique_ptr<GlobalData> MakeFakeDataOrDie(const Shape& shape,
                                              Client* client) {
  if (ShapeUtil::ByteSizeOf(shape) < (1LL << 20)) {
    StatusOr<std::unique_ptr<Literal>> literal_status = MakeFakeLiteral(shape);
    if (!literal_status.ok()) {
      // If we got an Unimplemented error, fall back to making the fake data via
      // an on-device computation.
      CHECK_EQ(literal_status.status().code(),
               tensorflow::error::UNIMPLEMENTED);
      return MakeFakeDataViaDeviceOrDie(shape, client);
    }
    return client->TransferToServer(*literal_status.ValueOrDie()).ValueOrDie();
  }

  // If the data is large, generate it on-device.
  return MakeFakeDataViaDeviceOrDie(shape, client);
}

std::vector<std::unique_ptr<GlobalData>> MakeFakeArgumentsOrDie(
    const Computation& computation, Client* client) {
  auto program_shape =
      client->GetComputationShape(computation).ConsumeValueOrDie();

  // For every (unbound) parameter that the computation wants, we manufacture
  // some arbitrary data so that we can invoke the computation.
  std::vector<std::unique_ptr<GlobalData>> fake_arguments;
  for (const Shape& parameter : program_shape->parameters()) {
    fake_arguments.push_back(MakeFakeDataOrDie(parameter, client));
  }

  return fake_arguments;
}

}  // namespace xla
