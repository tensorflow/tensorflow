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

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/master.grpc.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"

namespace tensorflow {
namespace data {

RegisterDatasetOp::RegisterDatasetOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  int64 external_state_policy_int;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr(kExternalStatePolicy, &external_state_policy_int));
  external_state_policy_ =
      SerializationContext::ExternalStatePolicy(external_state_policy_int);
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

  SerializationContext::Params params;
  params.external_state_policy = external_state_policy_;
  SerializationContext serialization_ctx(params);
  GraphDef graph_def;
  OP_REQUIRES_OK(
      ctx, AsGraphDef(ctx, dataset, std::move(serialization_ctx), &graph_def));

  // ::grpc::ChannelArguments args;
  std::shared_ptr<::grpc::ChannelCredentials> credentials;
  OP_REQUIRES_OK(
      ctx, CredentialsFactory::CreateClientCredentials(protocol, &credentials));
  auto channel = ::grpc::CreateChannel(address, credentials);
  auto master_stub = MasterService::NewStub(channel);
  GetOrRegisterDatasetRequest req;
  *req.mutable_dataset()->mutable_graph() = graph_def;
  GetOrRegisterDatasetResponse resp;
  grpc::ClientContext client_ctx;
  auto status = master_stub->GetOrRegisterDataset(&client_ctx, req, &resp);
  if (!status.ok()) {
    ctx->CtxFailure(grpc_util::WrapError("Failed to register dataset", status));
    return;
  }
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));
  auto output_dataset_id = output->tensor<int64, 0>();
  output_dataset_id() = resp.dataset_id();
}

BeginEpochOp::BeginEpochOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void BeginEpochOp::Compute(OpKernelContext* ctx) {
  int64 dataset_id;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kDatasetId, &dataset_id));

  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be non-empty."));

  tstring protocol;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kProtocol, &protocol));
  OP_REQUIRES(ctx, !protocol.empty(),
              errors::InvalidArgument(kProtocol, " must be non-empty."));

  std::shared_ptr<::grpc::ChannelCredentials> credentials;
  OP_REQUIRES_OK(
      ctx, CredentialsFactory::CreateClientCredentials(protocol, &credentials));
  auto channel = ::grpc::CreateChannel(address, credentials);
  auto master_stub = MasterService::NewStub(channel);
  BeginEpochRequest req;
  req.set_dataset_id(dataset_id);
  BeginEpochResponse resp;
  grpc::ClientContext client_ctx;
  auto status = master_stub->BeginEpoch(&client_ctx, req, &resp);
  if (!status.ok()) {
    ctx->CtxFailure(grpc_util::WrapError(
        absl::StrCat("Failed to begin epoch for dataset id ", dataset_id),
        status));
    return;
  }
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &output));
  auto output_epoch_id = output->tensor<int64, 0>();
  output_epoch_id() = resp.epoch_id();
}

Status MakeDataServiceIteratorOp::DoCompute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  const Tensor* epoch_id_tensor;
  TF_RETURN_IF_ERROR(ctx->input(kEpochId, &epoch_id_tensor));
  int64 epoch_id = epoch_id_tensor->scalar<int64>()();

  IteratorResource* iterator_resource;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, 2), &iterator_resource));

  core::ScopedUnref unref_iterator(iterator_resource);
  return iterator_resource->SetIteratorFromDataset(ctx, dataset, epoch_id);
}

REGISTER_KERNEL_BUILDER(Name("RegisterDataset").Device(DEVICE_CPU),
                        RegisterDatasetOp);
REGISTER_KERNEL_BUILDER(Name("BeginEpoch").Device(DEVICE_CPU), BeginEpochOp);
REGISTER_KERNEL_BUILDER(Name("MakeDataServiceIterator").Device(DEVICE_CPU),
                        MakeDataServiceIteratorOp);

}  // namespace data
}  // namespace tensorflow
