/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/experimental/data_service_ops.h"

#include <string>
#include <utility>

#include "absl/strings/numbers.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

namespace {
const int64_t kRetryTimeoutMicros = 1000LL * 1000 * 60 * 60;  // 60 minutes.
}

RegisterDatasetOp::RegisterDatasetOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  int64_t external_state_policy_int;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kExternalStatePolicy, &external_state_policy_int));
  external_state_policy_ =
      SerializationContext::ExternalStatePolicy(external_state_policy_int);

  if (ctx->HasAttr(kElementSpec)) {
    tstring element_spec;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kElementSpec, &element_spec));
    element_spec_ = element_spec;
  }

  if (ctx->HasAttr(kMetadata)) {
    tstring serialized_metadata;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kMetadata, &serialized_metadata));
    serialized_metadata_ = serialized_metadata;
  }
}

void RegisterDatasetOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be non-empty."));

  tstring protocol;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kProtocol, &protocol));
  OP_REQUIRES(ctx, !protocol.empty(),
              errors::InvalidArgument(kProtocol, " must be non-empty."));

  SerializationContext::Params params(ctx);
  params.external_state_policy = external_state_policy_;
  SerializationContext serialization_ctx(params);
  DatasetDef dataset_def;
  Status s = AsGraphDef(dataset, std::move(serialization_ctx),
                        dataset_def.mutable_graph());
  if (!s.ok()) {
    OP_REQUIRES_OK(
        ctx,
        errors::FailedPrecondition(
            "Serialization error while trying to register a dataset with "
            "tf.data service. "
            "The dataset may depend on a resource located on a different "
            "device. "
            "To address this, call `register_dataset` from the device with the "
            "resource, then use `from_dataset_id` to create per-device "
            "datasets. "
            "Original error: ",
            s));
  }

  DataServiceMetadata metadata;
  if (!element_spec_.empty()) {
    metadata.set_element_spec(element_spec_);
  }
  if (!serialized_metadata_.empty()) {
    OP_REQUIRES(ctx, metadata.ParseFromString(serialized_metadata_),
                errors::InvalidArgument(
                    "Failed to parse DataServiceMetadata from string: ",
                    std::string(serialized_metadata_)));
  }
  metadata.set_cardinality(dataset->Cardinality());

  DataServiceDispatcherClient client(address, protocol);
  std::string dataset_id;
  int64_t deadline_micros = EnvTime::NowMicros() + kRetryTimeoutMicros;
  OP_REQUIRES_OK(
      ctx, grpc_util::Retry(
               [&]() {
                 return client.RegisterDataset(dataset_def, metadata,
                                               dataset_id);
               },
               /*description=*/
               strings::StrCat("register dataset with dispatcher at ", address),
               deadline_micros));

  int64_t dataset_id_int;
  OP_REQUIRES(ctx, absl::SimpleAtoi(dataset_id, &dataset_id_int),
              errors::InvalidArgument("Failed to parse dataset ID: ",
                                      dataset_id, ". Expect integers."));

  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));
  auto output_dataset_id = output->tensor<int64_t, 0>();
  output_dataset_id() = dataset_id_int;
}

REGISTER_KERNEL_BUILDER(Name("RegisterDataset").Device(DEVICE_CPU),
                        RegisterDatasetOp);

}  // namespace data
}  // namespace tensorflow
