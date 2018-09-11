/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <stdlib.h>
#include "ignite_dataset.h"
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace {

class IgniteDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    std::string cache_name = "";
    std::string host = "";
    int32 port = -1;
    bool local = false;
    int32 part = -1;
    int32 page_size = -1;
    std::string username = "";
    std::string password = "";
    std::string certfile = "";
    std::string keyfile = "";
    std::string cert_password = "";

    const char* env_cache_name = std::getenv("IGNITE_DATASET_CACHE_NAME");
    const char* env_host = std::getenv("IGNITE_DATASET_HOST");
    const char* env_port = std::getenv("IGNITE_DATASET_PORT");
    const char* env_local = std::getenv("IGNITE_DATASET_LOCAL");
    const char* env_part = std::getenv("IGNITE_DATASET_PART");
    const char* env_page_size = std::getenv("IGNITE_DATASET_PAGE_SIZE");
    const char* env_username = std::getenv("IGNITE_DATASET_USERNAME");
    const char* env_password = std::getenv("IGNITE_DATASET_PASSWORD");
    const char* env_certfile = std::getenv("IGNITE_DATASET_CERTFILE");
    const char* env_keyfile = std::getenv("IGNITE_DATASET_KEYFILE");
    const char* env_cert_password = std::getenv("IGNITE_DATASET_CERT_PASSWORD");

    if (env_cache_name)
      cache_name = std::string(env_cache_name);
    else
      OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "cache_name",
                                                           &cache_name));

    if (env_host)
      host = std::string(env_host);
    else
      OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "host", &host));

    if (env_port)
      port = atoi(env_port);
    else
      OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "port", &port));

    if (env_local)
      local = true;
    else
      OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "local", &local));

    if (env_part)
      part = atoi(env_part);
    else
      OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "part", &part));

    if (env_page_size)
      page_size = atoi(env_page_size);
    else
      OP_REQUIRES_OK(ctx,
                     ParseScalarArgument<int32>(ctx, "page_size", &page_size));

    if (env_username)
      username = std::string(env_username);
    else
      OP_REQUIRES_OK(
          ctx, ParseScalarArgument<std::string>(ctx, "username", &username));

    if (env_password)
      password = std::string(env_password);
    else
      OP_REQUIRES_OK(
          ctx, ParseScalarArgument<std::string>(ctx, "password", &password));

    if (env_certfile)
      certfile = std::string(env_certfile);
    else
      OP_REQUIRES_OK(
          ctx, ParseScalarArgument<std::string>(ctx, "certfile", &certfile));

    if (env_keyfile)
      keyfile = std::string(env_keyfile);
    else
      OP_REQUIRES_OK(
          ctx, ParseScalarArgument<std::string>(ctx, "keyfile", &keyfile));

    if (env_cert_password)
      cert_password = std::string(env_cert_password);
    else
      OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "cert_password",
                                                           &cert_password));

    const Tensor* schema_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("schema", &schema_tensor));
    OP_REQUIRES(ctx, schema_tensor->dims() == 1,
                errors::InvalidArgument("`schema` must be a vector."));

    std::vector<int32> schema;
    schema.reserve(schema_tensor->NumElements());
    for (int i = 0; i < schema_tensor->NumElements(); i++) {
      schema.push_back(schema_tensor->flat<int32>()(i));
    }

    const Tensor* permutation_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("permutation", &permutation_tensor));
    OP_REQUIRES(ctx, schema_tensor->dims() == 1,
                errors::InvalidArgument("`permutation` must be a vector."));

    std::vector<int32> permutation;
    permutation.reserve(permutation_tensor->NumElements());
    for (int i = 0; i < permutation_tensor->NumElements(); i++) {
      permutation.push_back(permutation_tensor->flat<int32>()(i));
    }

    *output =
        new IgniteDataset(ctx, cache_name, host, port, local, part, page_size,
                          username, password, certfile, keyfile, cert_password,
                          std::move(schema), std::move(permutation));
  }
};

REGISTER_KERNEL_BUILDER(Name("IgniteDataset").Device(DEVICE_CPU),
                        IgniteDatasetOp);

}  // namespace
}  // namespace tensorflow
