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

#include "tensorflow/contrib/ignite/kernels/dataset/ignite_binary_object_parser.h"
#include "tensorflow/contrib/ignite/kernels/dataset/ignite_dataset.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {
namespace {

Status SchemaToTypes(const std::vector<int32>& schema, DataTypeVector* dtypes) {
  for (auto e : schema) {
    if (e == BYTE || e == BYTE_ARR) {
      dtypes->push_back(DT_UINT8);
    } else if (e == SHORT || e == SHORT_ARR) {
      dtypes->push_back(DT_INT16);
    } else if (e == INT || e == INT_ARR) {
      dtypes->push_back(DT_INT32);
    } else if (e == LONG || e == LONG_ARR) {
      dtypes->push_back(DT_INT64);
    } else if (e == FLOAT || e == FLOAT_ARR) {
      dtypes->push_back(DT_FLOAT);
    } else if (e == DOUBLE || e == DOUBLE_ARR) {
      dtypes->push_back(DT_DOUBLE);
    } else if (e == USHORT || e == USHORT_ARR) {
      dtypes->push_back(DT_UINT8);
    } else if (e == BOOL || e == BOOL_ARR) {
      dtypes->push_back(DT_BOOL);
    } else if (e == STRING || e == STRING_ARR) {
      dtypes->push_back(DT_STRING);
    } else {
      return errors::Unknown("Unexpected type in schema [type_id=", e, "]");
    }
  }

  return Status::OK();
}

Status SchemaToShapes(const std::vector<int32>& schema,
                      std::vector<PartialTensorShape>* shapes) {
  for (auto e : schema) {
    if (e >= 1 && e < 10) {
      shapes->push_back(PartialTensorShape({}));
    } else if (e >= 12 && e < 21) {
      shapes->push_back(PartialTensorShape({-1}));
    } else {
      return errors::Unknown("Unexpected type in schema [type_id=", e, "]");
    }
  }

  return Status::OK();
}

class IgniteDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    string cache_name = "";
    string host = "";
    int32 port = -1;
    bool local = false;
    int32 part = -1;
    int32 page_size = -1;
    string username = "";
    string password = "";
    string certfile = "";
    string keyfile = "";
    string cert_password = "";

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

    if (env_cache_name) {
      cache_name = string(env_cache_name);
    } else {
      OP_REQUIRES_OK(
          ctx, ParseScalarArgument<string>(ctx, "cache_name", &cache_name));
    }

    if (env_host) {
      host = string(env_host);
    } else {
      OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "host", &host));
    }

    if (env_port) {
      OP_REQUIRES(ctx, strings::safe_strto32(env_port, &port),
                  errors::InvalidArgument("IGNITE_DATASET_PORT environment "
                                          "variable is not a valid integer: ",
                                          env_port));
    } else {
      OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "port", &port));
    }

    if (env_local) {
      local = true;
    } else {
      OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "local", &local));
    }

    if (env_part) {
      OP_REQUIRES(ctx, strings::safe_strto32(env_part, &part),
                  errors::InvalidArgument("IGNITE_DATASET_PART environment "
                                          "variable is not a valid integer: ",
                                          env_part));
    } else {
      OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "part", &part));
    }

    if (env_page_size) {
      OP_REQUIRES(ctx, strings::safe_strto32(env_page_size, &page_size),
                  errors::InvalidArgument("IGNITE_DATASET_PAGE_SIZE "
                                          "environment variable is not a valid "
                                          "integer: ",
                                          env_page_size));
    } else {
      OP_REQUIRES_OK(ctx,
                     ParseScalarArgument<int32>(ctx, "page_size", &page_size));
    }

    if (env_username) username = string(env_username);

    if (env_password) password = string(env_password);

    if (env_certfile) certfile = string(env_certfile);

    if (env_keyfile) keyfile = string(env_keyfile);

    if (env_cert_password) cert_password = string(env_cert_password);

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
    OP_REQUIRES(ctx, permutation_tensor->dims() == 1,
                errors::InvalidArgument("`permutation` must be a vector."));

    std::vector<int32> permutation;
    permutation.resize(permutation_tensor->NumElements());
    for (int i = 0; i < permutation_tensor->NumElements(); i++) {
      // Inversed permutation.
      permutation[permutation_tensor->flat<int32>()(i)] = i;
    }

    DataTypeVector dtypes;
    std::vector<PartialTensorShape> shapes;

    OP_REQUIRES_OK(ctx, SchemaToTypes(schema, &dtypes));
    OP_REQUIRES_OK(ctx, SchemaToShapes(schema, &shapes));

    *output = new IgniteDataset(
        ctx, std::move(cache_name), std::move(host), port, local, part,
        page_size, std::move(username), std::move(password),
        std::move(certfile), std::move(keyfile), std::move(cert_password),
        std::move(schema), std::move(permutation), std::move(dtypes),
        std::move(shapes));
  }
};

REGISTER_KERNEL_BUILDER(Name("IgniteDataset").Device(DEVICE_CPU),
                        IgniteDatasetOp);

}  // namespace
}  // namespace tensorflow
