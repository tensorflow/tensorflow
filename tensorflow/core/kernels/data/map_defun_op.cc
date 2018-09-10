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

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {
namespace data {
namespace {

void SetRunOptions(OpKernelContext* ctx, FunctionLibraryRuntime::Options* opts,
                   bool always_collect_stats) {
  opts->step_id = ctx->step_id();
  opts->rendezvous = ctx->rendezvous();
  if (always_collect_stats) {
    opts->stats_collector = ctx->stats_collector();
  }
  opts->runner = ctx->runner();
}

class MapDefunOp : public AsyncOpKernel {
 public:
  explicit MapDefunOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    auto func_lib = ctx->function_library();
    OP_REQUIRES(ctx, func_lib != nullptr,
                errors::Internal("No function library."));
    const NameAttrList* func;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func));
    OP_REQUIRES_OK(ctx,
                   func_lib->Instantiate(func->name(), AttrSlice(&func->attr()),
                                         &func_handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));

    OP_REQUIRES(ctx, ctx->num_inputs() >= 0,
                errors::InvalidArgument("Must have at least one input."));
    OP_REQUIRES(ctx, ctx->num_outputs() >= 0,
                errors::InvalidArgument("Must have at least one output."));
    OP_REQUIRES(ctx, ctx->num_outputs() == output_shapes_.size(),
                errors::InvalidArgument(
                    "Length of output_shapes and output_types must match."));
  }

  ~MapDefunOp() override {}

  Status GetInputBatchSize(OpKernelContext* ctx, int64* batch_size) {
    // Validates inputs and gets the size of their leading dimension.
    *batch_size = ctx->input(0).dims() > 0 ? ctx->input(0).dim_size(0) : -1;
    for (size_t i = 0; i < ctx->num_inputs(); ++i) {
      if (ctx->input(i).dims() == 0) {
        return errors::InvalidArgument(
            "All inputs must have rank at least 1. Input ", i,
            " has a rank of 0.");
      } else if (ctx->input(i).dim_size(0) != *batch_size) {
        return errors::InvalidArgument(
            "All inputs must have the same dimension 0. Input ", i,
            " has leading dimension ", ctx->input(i).dim_size(0),
            ", while all previous inputs have leading dimension ", batch_size);
      }
    }
    return Status::OK();
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    int64 batch_size;
    OP_REQUIRES_OK_ASYNC(ctx, GetInputBatchSize(ctx, &batch_size), done);

    // Inputs
    auto* args = new std::vector<Tensor>;
    auto* arg_shapes = new std::vector<TensorShape>;

    // Create a copy because every `Compute` may have different output shapes.
    auto* output_shapes = new std::vector<PartialTensorShape>(output_shapes_);
    arg_shapes->reserve(ctx->num_inputs());
    args->reserve(ctx->num_inputs());

    auto* mu = new mutex;

    for (size_t i = 0; i < ctx->num_inputs(); ++i) {
      args->push_back(ctx->input(i));
      arg_shapes->push_back(ctx->input(i).shape());
      arg_shapes->at(i).RemoveDim(0);  // Remove the first batch dimension
    }

    // Outputs
    auto* output = new OpOutputList;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("output", output), done);

    for (size_t i = 0; i < output_types().size(); ++i) {
      if (output_shapes_.at(i).IsFullyDefined()) {
        Tensor* out = nullptr;
        TensorShape output_shape;
        output_shapes_.at(i).AsTensorShape(&output_shape);
        output_shape.InsertDim(0, batch_size);
        OP_REQUIRES_OK_ASYNC(ctx, output->allocate(i, output_shape, &out),
                             done);
      }
    }

    SetRunOptions(ctx, &opts_, false);

    // Run loop
    StatusCallback callback = std::bind(
        [](OpKernelContext* ctx, std::vector<Tensor>* args,
           std::vector<TensorShape>* arg_shapes,
           std::vector<PartialTensorShape>* output_shapes, OpOutputList* output,
           mutex* mu, DoneCallback& done, const Status& status) {
          delete args;
          delete arg_shapes;
          delete output;
          delete output_shapes;
          delete mu;
          ctx->SetStatus(status);
          done();
        },
        ctx, args, arg_shapes, output_shapes, output, mu, std::move(done),
        std::placeholders::_1);

    auto* refcounted = new ReffedStatusCallback(std::move(callback));

    for (size_t i = 1; i < static_cast<size_t>(batch_size); ++i) {
      // Start from i = 1 because refcounted is initialized with refcount = 1
      refcounted->Ref();
    }

    for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
      auto* call_frame = new MapFunctionCallFrame(
          *args, *arg_shapes, output_shapes, mu, output, this, i,
          static_cast<size_t>(batch_size));
      CancellationManager* c_mgr = new CancellationManager;
      opts_.cancellation_manager = c_mgr;
      ctx->function_library()->Run(
          opts_, func_handle_, call_frame,
          [call_frame, refcounted, c_mgr](const Status& func_status) {
            delete call_frame;
            delete c_mgr;
            refcounted->UpdateStatus(func_status);
            refcounted->Unref();
          });
    }
  }

 private:
  FunctionLibraryRuntime::Handle func_handle_;
  FunctionLibraryRuntime::Options opts_;
  std::vector<PartialTensorShape> output_shapes_;

  class MapFunctionCallFrame : public CallFrameInterface {
   public:
    MapFunctionCallFrame(const std::vector<Tensor>& args,
                         const std::vector<TensorShape>& arg_shapes,
                         std::vector<PartialTensorShape>* output_shapes,
                         mutex* output_shapes_mutex, OpOutputList* output,
                         OpKernel* kernel, size_t iter, size_t batch_size)
        : args_(args),
          arg_shapes_(arg_shapes),
          output_shapes_(output_shapes),
          output_shapes_mutex_(output_shapes_mutex),
          output_(output),
          kernel_(kernel),
          iter_(iter),
          batch_size_(batch_size) {}

    ~MapFunctionCallFrame() override {}

    size_t num_args() const override { return args_.size(); }
    size_t num_retvals() const override {
      return static_cast<size_t>(kernel_->num_outputs());
    }

    Status GetArg(int index, Tensor* val) const override {
      if (index < 0 || index >= args_.size()) {
        return errors::InvalidArgument(
            "Mismatch in number of function inputs.");
      }
      bool result = val->CopyFrom(args_.at(index).Slice(iter_, iter_ + 1),
                                  arg_shapes_.at(index));
      if (!result) {
        return errors::Internal("GetArg failed.");
      } else if (!val->IsAligned()) {
        // Ensure alignment
        *val = tensor::DeepCopy(*val);
      }

      return Status::OK();
    }

    Status SetRetval(int index, const Tensor& val) override {
      if (index < 0 || index >= kernel_->num_outputs()) {
        return errors::InvalidArgument(
            "Mismatch in number of function outputs.");
      }

      if (val.dtype() != kernel_->output_type(index)) {
        return errors::InvalidArgument(
            "Mismatch in function return type and expected output type for "
            "output: ",
            index);
      }
      {  // Locking scope
        mutex_lock l(*output_shapes_mutex_);
        if (!output_shapes_->at(index).IsCompatibleWith(val.shape())) {
          return errors::InvalidArgument(
              "Mismatch in function retval shape, ", val.shape(),
              ", and expected output shape,",
              output_shapes_->at(index).DebugString(), ".");
        }
        if (!output_shapes_->at(index).IsFullyDefined()) {
          // Given val, we have new information about the output shape at
          // this index. Store the shape and allocate the output accordingly.
          output_shapes_->at(index) = val.shape();

          Tensor* out = nullptr;
          TensorShape actual_shape = val.shape();
          actual_shape.InsertDim(0, batch_size_);
          TF_RETURN_IF_ERROR(output_->allocate(index, actual_shape, &out));
        }
      }
      return batch_util::CopyElementToSlice(val, (*output_)[index], iter_);
    }

   private:
    const std::vector<Tensor>& args_;
    const std::vector<TensorShape>& arg_shapes_;
    std::vector<PartialTensorShape>* output_shapes_;
    mutex* output_shapes_mutex_;
    OpOutputList* output_;
    const OpKernel* kernel_;
    const size_t iter_;
    const size_t batch_size_;
  };
};

REGISTER_KERNEL_BUILDER(Name("MapDefun").Device(DEVICE_CPU), MapDefunOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
