/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/distributed_save_op.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/io/compression.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/py_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {

namespace {

const absl::Duration kRetryTimeout = absl::Hours(1);

constexpr char kDistributedSave[] = "DistributedSave";

}  // namespace

DistributedSaveOp::DistributedSaveOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  if (ctx->HasAttr(kMetadata)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kMetadata, &serialized_metadata_));
  }
}

void DistributedSaveOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  OP_REQUIRES(
      ctx, dataset->Cardinality() != kInfiniteCardinality,
      errors::InvalidArgument("Saving an infinite dataset is not allowed: ",
                              dataset->DebugString()));

  tstring directory;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kDirectory, &directory));
  OP_REQUIRES(ctx, !directory.empty(),
              errors::InvalidArgument(kDirectory, " must be nonempty"));

  tstring address;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kAddress, &address));
  OP_REQUIRES(ctx, !address.empty(),
              errors::InvalidArgument(kAddress, " must be nonempty"));

  bool has_atomic_move = false;
  OP_REQUIRES_OK(ctx, ctx->env()->HasAtomicMove(directory, &has_atomic_move));
  OP_REQUIRES(ctx, has_atomic_move,
              absl::FailedPreconditionError(absl::StrCat(
                  "The file system for ", std::string(directory),
                  " does not support atomic move (rename), which is required "
                  "to write tf.data snapshots.")));

  SerializationContext::Params params(ctx);
  SerializationContext serialization_ctx(params);
  DatasetDef dataset_def;
  absl::Status s = AsGraphDef(dataset, std::move(serialization_ctx),
                              dataset_def.mutable_graph());
  if (!s.ok()) {
    OP_REQUIRES_OK(
        ctx,
        errors::FailedPrecondition(
            "Serialization error while trying to save dataset with tf.data "
            "service. The dataset may depend on a resource located on a "
            "different device. To address this, call `distributed_save` from "
            "the device with the resource. Original error: ",
            s));
  }

  experimental::DistributedSnapshotMetadata metadata;
  if (!serialized_metadata_.empty()) {
    OP_REQUIRES(ctx, metadata.ParseFromString(serialized_metadata_),
                errors::InvalidArgument(
                    "Failed to parse DistributedSnapshotMetadata from string: ",
                    std::string(serialized_metadata_)));
  }
  if (metadata.compression() == "AUTO") {
    metadata.set_compression(tsl::io::compression::kSnappy);
  }

  DataServiceDispatcherClient client(address, DefaultProtocol());
  int64_t deadline_micros =
      EnvTime::NowMicros() + absl::ToInt64Microseconds(kRetryTimeout);
  OP_REQUIRES_OK(
      ctx,
      grpc_util::Retry(
          [&]() { return client.Snapshot(dataset_def, directory, metadata); },
          /*description=*/
          strings::StrCat("save with tf.data service dispatcher at ", address),
          deadline_micros));
  metrics::RecordTFDataServiceSnapshotOp(directory, kDistributedSave);
}

REGISTER_KERNEL_BUILDER(Name(kDistributedSave).Device(DEVICE_CPU),
                        DistributedSaveOp);

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
