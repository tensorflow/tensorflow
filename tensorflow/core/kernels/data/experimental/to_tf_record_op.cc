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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class ToTFRecordOp : public AsyncOpKernel {
 public:
  explicit ToTFRecordOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_to_tf_record") {}

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece& argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an inter-op
    // thread pool thread, so we issue the call using a background thread.
    background_worker_.Schedule(std::bind(
        [this, ctx](std::function<void()>& done) {
          tstring filename;
          OP_REQUIRES_OK_ASYNC(
              ctx, ParseScalarArgument<tstring>(ctx, "filename", &filename),
              done);
          tstring compression_type;
          OP_REQUIRES_OK_ASYNC(ctx,
                               ParseScalarArgument<tstring>(
                                   ctx, "compression_type", &compression_type),
                               done);
          std::unique_ptr<WritableFile> file;
          OP_REQUIRES_OK_ASYNC(
              ctx, ctx->env()->NewWritableFile(filename, &file), done);
          auto writer = absl::make_unique<io::RecordWriter>(
              file.get(), io::RecordWriterOptions::CreateRecordWriterOptions(
                              compression_type));

          DatasetBase* dataset;
          OP_REQUIRES_OK_ASYNC(
              ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);

          IteratorContext::Params params(ctx);
          FunctionHandleCache function_handle_cache(params.flr);
          params.function_handle_cache = &function_handle_cache;
          ResourceMgr resource_mgr;
          params.resource_mgr = &resource_mgr;
          CancellationManager cancellation_manager;
          params.cancellation_manager = &cancellation_manager;
          std::function<void()> deregister_fn;
          OP_REQUIRES_OK_ASYNC(
              ctx,
              RegisterCancellationCallback(
                  ctx->cancellation_manager(),
                  [cm = params.cancellation_manager]() { cm->StartCancel(); },
                  &deregister_fn),
              done);

          // Update the `done` callback to deregister the cancellation callback.
          done = std::bind(
              [](const std::function<void()>& done,
                 const std::function<void()>& deregister_fn) {
                deregister_fn();
                done();
              },
              std::move(done), std::move(deregister_fn));

          IteratorContext iter_ctx(std::move(params));
          std::unique_ptr<IteratorBase> iterator;
          OP_REQUIRES_OK_ASYNC(
              ctx,
              dataset->MakeIterator(&iter_ctx, "ToTFRecordOpIterator",
                                    &iterator),
              done);

          // Update the `done` callback to destroy the iterator before calling
          // the actual callback to avoid destruction races.
          IteratorBase* raw_iterator = iterator.release();
          done = std::bind(
              [raw_iterator](const std::function<void()>& done) {
                delete raw_iterator;
                done();
              },
              std::move(done));

          std::vector<Tensor> components;
          components.reserve(dataset->output_dtypes().size());
          bool end_of_sequence;
          do {
            OP_REQUIRES_OK_ASYNC(
                ctx,
                raw_iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
                done);

            if (!end_of_sequence) {
              OP_REQUIRES_OK_ASYNC(
                  ctx, writer->WriteRecord(components[0].scalar<tstring>()()),
                  done);
            }
            components.clear();
          } while (!end_of_sequence);
          done();
        },
        std::move(done)));
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(Name("DatasetToTFRecord").Device(DEVICE_CPU),
                        ToTFRecordOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDatasetToTFRecord").Device(DEVICE_CPU), ToTFRecordOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
