/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_CLIENT_LIB_TESTING_H_
#define XLA_CLIENT_LIB_TESTING_H_

#include <memory>
#include <vector>

#include "xla/client/client.h"
#include "xla/client/global_data.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Generates fake data of the given shape on the device or dies. The fake data
// is created by performing a computation on the device rather than transferring
// data from the host to the device.
//
// The optional DebugOptions are used when generating fake data on the device.
std::unique_ptr<GlobalData> MakeFakeDataOrDie(
    const Shape& shape, Client* client, DebugOptions* debug_opts = nullptr);

// Returns vector of GlobalData handles of fake data (created using
// MakeFakeDataOrDie) that are correctly shaped arguments for the given
// xla computation.
//
// The optional DebugOptions are used when generating fake data on the device.
std::vector<std::unique_ptr<GlobalData>> MakeFakeArgumentsOrDie(
    const XlaComputation& computation, Client* client,
    DebugOptions* debug_opts = nullptr);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_TESTING_H_
