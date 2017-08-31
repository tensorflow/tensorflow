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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_TESTING_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_TESTING_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Generates fake data in a literal of the given shape, or returns an error
// status if the element type is currently unhandled for fake data generation.
StatusOr<std::unique_ptr<Literal>> MakeFakeLiteral(const Shape& shape);

// Generates fake data of the given shape on the device or dies. The fake data
// is created by performing a computation on the device rather than transferring
// data from the host to the device.
std::unique_ptr<GlobalData> MakeFakeDataOrDie(const Shape& shape,
                                              Client* client);

// Returns vector of GlobalData handles of fake data (created using
// MakeFakeDataOrDie) that are correctly shaped arguments for the given
// computation.
std::vector<std::unique_ptr<GlobalData>> MakeFakeArgumentsOrDie(
    const Computation& computation, Client* client);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_TESTING_H_
