/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"
#include "tensorflow/lite/delegates/flex/subgraph_resource.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tensorflow {

namespace {
constexpr int kTfLiteSubgraphResource = 0;
}

REGISTER_OP("TfLiteSubgraphExecute")
    .Input("subgraph_key: string")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

// The `TfLiteSubgraphExecute` executes a tflite subgraph with the designated
// inputs. This op will first look up the tflite subgraph from TF resource
// manager based on the resource name stored on the first input, and then it
// will call that specific subgraph with the remaining arguments. The first
// input of this op is always a scalar string, which denotes the name of the
// subgraph resource. The remaining inputs will be fed to the subgraph as
// inputs, so the caller needs to ensure that the remaining inputs match with
// the subgraph's expected inputs. This is currently WIP/experimental and
// subject to change.
class TfLiteSubgraphExecute : public OpKernel {
 public:
  explicit TfLiteSubgraphExecute(OpKernelConstruction* ctx)
      : OpKernel(ctx), tfl_tensors_need_allocation_(true) {}

  void Compute(OpKernelContext* ctx) override {
    // Fetch the TF Lite subgraph to execute.
    tflite::flex::TFLiteSubgraphResource* resource = nullptr;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->Lookup<tflite::flex::TFLiteSubgraphResource>(
            "flex", ctx->input(kTfLiteSubgraphResource).flat<tstring>()(0),
            &resource));
    tensorflow::core::ScopedUnref unref_resource(resource);

    // Try to acquire a mutex lock from this resource. This is because tflite
    // subgraph is not thread-safe and we need to guarantee exclusive access to
    // it.
    mutex_lock lock(resource->GetExclusiveLock());
    tflite::Subgraph& subgraph_selected = resource->GetSubgraphResource();

    OP_REQUIRES(ctx, ctx->num_inputs() == subgraph_selected.inputs().size() + 1,
                errors::InvalidArgument("TF Lite subgraph expects ",
                                        subgraph_selected.inputs().size(),
                                        " inputs, but received ",
                                        ctx->num_inputs() - 1, "."));

    // Resize input tensors if necessary.
    ResizeInputTensor(ctx, subgraph_selected);

    if (tfl_tensors_need_allocation_) {
      subgraph_selected.AllocateTensors();
      tfl_tensors_need_allocation_ = false;
    }

    // Copy input tensors to subgraph.
    SetSubgraphInput(ctx, subgraph_selected);

    OP_REQUIRES(ctx, subgraph_selected.Invoke() == kTfLiteOk,
                errors::Internal("Failed to invoke tflite subgraph"));

    // Copy tflite results.
    CopyTFLiteSubgraphResult(ctx, subgraph_selected);
  }

 private:
  void ResizeInputTensor(OpKernelContext* ctx,
                         tflite::Subgraph& subgraph_selected) {
    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      // Shift index by 1 since the first input is always the resource name.
      const Tensor& tf_tensor = ctx->input(i + 1);
      TfLiteTensor* subgraph_input =
          subgraph_selected.tensor(subgraph_selected.inputs()[i]);

      bool need_resize = false;
      for (int dim = 0; dim < tf_tensor.shape().dims(); dim++) {
        if (tf_tensor.shape().dim_size(dim) !=
            subgraph_input->dims->data[dim]) {
          need_resize = true;
          break;
        }
      }
      if (need_resize) {
        std::vector<int> new_shape;
        for (auto dim : tf_tensor.shape().dim_sizes()) {
          new_shape.push_back(dim);
        }
        tfl_tensors_need_allocation_ = true;
        OP_REQUIRES(ctx,
                    subgraph_selected.ResizeInputTensor(
                        subgraph_selected.inputs()[i], new_shape) == kTfLiteOk,
                    errors::Internal("Failed to resize tflite tensor"));
      }
    }
  }

  void SetSubgraphInput(OpKernelContext* ctx,
                        tflite::Subgraph& subgraph_selected) const {
    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      const Tensor& tf_tensor = ctx->input(i + 1);
      TfLiteTensor* subgraph_input =
          subgraph_selected.tensor(subgraph_selected.inputs()[i]);

      if (subgraph_input->type == kTfLiteString) {
        OP_REQUIRES(ctx, tf_tensor.dtype() == tensorflow::DT_STRING,
                    errors::InvalidArgument("Tensor doesn't have string type"));
        tflite::DynamicBuffer dynamic_buffer;
        auto tf_data = tf_tensor.flat<tensorflow::tstring>();
        for (int i = 0; i < tf_tensor.NumElements(); ++i) {
          dynamic_buffer.AddString(tf_data(i).data(), tf_data(i).size());
        }

        dynamic_buffer.WriteToTensor(subgraph_input, /*new_shape=*/nullptr);
      } else if (subgraph_input->type == kTfLiteResource ||
                 subgraph_input->type == kTfLiteVariant) {
        // TODO(b/179094265): This is an experimental implementation, subject to
        // change. This can be re-implemented with life cycle management
        // mechanism like reference counting.
        const size_t required_bytes = sizeof(tensorflow::Tensor**);
        const tensorflow::Tensor** tf_tensor_ptr =
            reinterpret_cast<const tensorflow::Tensor**>(
                malloc(required_bytes));
        *tf_tensor_ptr = &tf_tensor;

        TfLiteTensorDataFree(subgraph_input);
        subgraph_input->data.raw = reinterpret_cast<char*>(tf_tensor_ptr);
        subgraph_input->bytes = required_bytes;
        subgraph_input->data_is_stale = true;
      } else {
        tensorflow::StringPiece tensor_data = tf_tensor.tensor_data();
        OP_REQUIRES(ctx, subgraph_input->bytes == tensor_data.size(),
                    errors::Internal("tensor size doesn't match"));
        // TODO(b/181352924): This could incur some overhead in memory copy.
        // Optimize this away in the future.
        memcpy(subgraph_input->data.raw, tensor_data.data(),
               tensor_data.size());
      }
    }
  }

  void CopyTFLiteSubgraphResult(OpKernelContext* ctx,
                                tflite::Subgraph& subgraph_selected) const {
    for (int i = 0; i < subgraph_selected.outputs().size(); ++i) {
      OP_REQUIRES(ctx,
                  subgraph_selected.EnsureTensorDataIsReadable(
                      subgraph_selected.outputs()[i]) == kTfLiteOk,
                  errors::Internal("TF lite subgraph output is not readable"));
      // Create an output tensor.
      TfLiteTensor* subgraph_output =
          subgraph_selected.tensor(subgraph_selected.outputs()[i]);

      Tensor tensor;
      OP_REQUIRES_OK(
          ctx, tflite::flex::SetTfTensorFromTfLite(subgraph_output, &tensor));
      ctx->set_output(i, std::move(tensor));
    }
  }

  // Tells if the target subgraph needs to invoko AllocateTensors().
  bool tfl_tensors_need_allocation_;
};

REGISTER_KERNEL_BUILDER(Name("TfLiteSubgraphExecute").Device(DEVICE_CPU),
                        TfLiteSubgraphExecute);

}  // namespace tensorflow
