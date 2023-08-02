/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_TEST_BASE_H_

namespace pjrt {

class PjrtCApiTestBase : public ::testing::Test {
 public:
  explicit PjrtCApiTestBase(const PJRT_Api* api);
  ~PjrtCApiTestBase() override;

 protected:
  const PJRT_Api* api_;
  PJRT_Client* client_;
  void destroy_client(PJRT_Client* client);

 private:
  PjrtCApiTestBase(const PjrtCApiTestBase&) = delete;
  void operator=(const PjrtCApiTestBase&) = delete;
};

}  // namespace pjrt

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_TEST_BASE_H_
