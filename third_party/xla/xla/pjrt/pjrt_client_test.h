/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_CLIENT_TEST_H_
#define XLA_PJRT_PJRT_CLIENT_TEST_H_

#include <functional>
#include <memory>

#include "xla/pjrt/pjrt_client.h"

namespace xla {

void RegisterTestClientFactory(
    std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory);
}

#endif  // XLA_PJRT_PJRT_CLIENT_TEST_H_
