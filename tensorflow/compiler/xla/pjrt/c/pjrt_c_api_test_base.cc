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

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_test_base.h"

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace pjrt {
namespace {

PJRT_Client* CreateClient(const PJRT_Api* api) {
  PJRT_Client_Create_Args create_args;
  create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  create_args.priv = nullptr;
  create_args.create_options = nullptr;
  create_args.num_options = 0;
  create_args.kv_get_callback = nullptr;
  create_args.kv_put_callback = nullptr;
  create_args.kv_put_user_arg = nullptr;
  create_args.kv_get_user_arg = nullptr;
  PJRT_Error* error = api->PJRT_Client_Create(&create_args);
  CHECK_EQ(error, nullptr);
  CHECK_NE(create_args.client, nullptr);
  return create_args.client;
}

}  // namespace

PjrtCApiTestBase::PjrtCApiTestBase(const PJRT_Api* api) {
  api_ = api;
  client_ = CreateClient(api_);
}

PjrtCApiTestBase::~PjrtCApiTestBase() { destroy_client(client_); }

void PjrtCApiTestBase::destroy_client(PJRT_Client* client) {
  PJRT_Client_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.priv = nullptr;
  destroy_args.client = client;
  PJRT_Error* error = api_->PJRT_Client_Destroy(&destroy_args);
  CHECK_EQ(error, nullptr);
}

}  // namespace pjrt
