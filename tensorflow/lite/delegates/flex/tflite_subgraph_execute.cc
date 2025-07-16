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

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"
#include "tensorflow/lite/delegates/flex/subgraph_resource.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
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
      : OpKernel(ctx),
        tfl_tensors_need_allocation_(true),
        fill_output_to_input_map_(true),
        first_run_(true),
        output_tensors_can_be_shared_(true) {}

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

    OutputToInputMap(ctx, subgraph_selected);
    OP_REQUIRES(ctx, ctx->num_inputs() == subgraph_selected.inputs().size() + 1,
                errors::InvalidArgument("TF Lite subgraph expects ",
                                        subgraph_selected.inputs().size(),
                                        " inputs, but received ",
                                        ctx->num_inputs() - 1, "."));

    // Resize input tensors if necessary.
    ResizeInputTensor(ctx, subgraph_selected);
    SetCustomAllocatorsForInputTensors(ctx, subgraph_selected);

    if (tfl_tensors_need_allocation_) {
      OP_REQUIRES(ctx, subgraph_selected.AllocateTensors() == kTfLiteOk,
                  errors::Internal("Failed to call allocate tensors"));
      tfl_tensors_need_allocation_ = false;
      // TODO(b/274934361): re-enable once allocation issue is solved.
      // output_tensors_can_be_shared_ =!subgraph_selected.HasDynamicTensors();
      output_tensors_can_be_shared_ = false;
    }

    if (output_tensors_can_be_shared_) {
      SetCustomAllocatorsForOutputTensors(ctx, subgraph_selected);
      // The output tensors have been allocated by
      // `SetCustomAllocatorsForOutputTensors`. They have also been arena
      // allocated by `AllocateTensors`. Calling `ReleaseMemory` and
      // `AllocateTensors` allows us to reduce peak memory usage by no longer
      // allocating the output tensors in the arena. This only needs to be done
      // once as subsequent calls to `AllocateTensors` will not allocate the
      // output tensors as they have been set to
      // `kTfLiteCustom`.
      if (first_run_) {
        subgraph_selected.ReleaseMemory();
        TfLiteStatus status = subgraph_selected.AllocateTensors();
        if (status != kTfLiteOk) {
          CleanUpCustomOutputs();
          ctx->CtxFailure(errors::Internal("Failed to call allocate tensors",
                                           subgraph_selected.GetName()));
          return;
        }
        first_run_ = false;
      }
    }
    // Copy input tensors to subgraph.
    SetSubgraphInput(ctx, subgraph_selected, resource->GetFlexDelegate());

    TfLiteStatus status = subgraph_selected.Invoke();
    if (status != kTfLiteOk) {
      CleanUpCustomOutputs();
      ctx->CtxFailure(errors::Internal("Failed to invoke tflite subgraph",
                                       subgraph_selected.GetName()));
      return;
    }

    // Copy tflite results.
    CopyTFLiteSubgraphResult(ctx, subgraph_selected);
  }

 private:
  void CleanUpCustomOutputs() {
    for (void* ptr : custom_output_ptrs_) {
      tensorflow::cpu_allocator()->DeallocateRaw(ptr);
    }
  }
  // Identifies subgraph inputs which are also outputs and maps the output
  // tensor id to the input number. This means that tensors which are both
  // subgraph inputs and outputs can be handled with zero copies.
  void OutputToInputMap(OpKernelContext* ctx,
                        tflite::Subgraph& subgraph_selected) {
    if (!fill_output_to_input_map_) {
      return;
    }
    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      int input_idx = subgraph_selected.inputs()[i];
      for (int j = 0; j < subgraph_selected.outputs().size(); ++j) {
        int output_idx = subgraph_selected.outputs()[j];
        if (input_idx == output_idx) {
          output_to_input_map_[output_idx] = i;
          break;
        }
      }
    }
    fill_output_to_input_map_ = false;
  }

  bool TensorCanBeShared(const TfLiteTensor* tensor) const {
    if ((tensor->type == kTfLiteResource || tensor->type == kTfLiteVariant ||
         tensor->type == kTfLiteString)) {
      return false;
    }
    if (tensor->allocation_type != kTfLiteArenaRw &&
        tensor->allocation_type != kTfLiteArenaRwPersistent &&
        tensor->allocation_type != kTfLiteCustom) {
      return false;
    }
    return true;
  }

  // Sets custom allocators for all output tensors which are not Resources,
  // Variants or Strings. This means that these tensors can share the same
  // memory as the TF tensors, reducing memcpys and memory usage.
  void SetCustomAllocatorsForOutputTensors(
      OpKernelContext* ctx, tflite::Subgraph& subgraph_selected) {
    custom_output_ptrs_.clear();
    for (int i = 0; i < subgraph_selected.outputs().size(); ++i) {
      int tensor_idx = subgraph_selected.outputs()[i];
      TfLiteTensor* subgraph_output = subgraph_selected.tensor(tensor_idx);
      if (!TensorCanBeShared(subgraph_output)) {
        continue;
      }
      if (output_to_input_map_.find(tensor_idx) != output_to_input_map_.end()) {
        continue;
      }
      void* ptr = tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, subgraph_output->bytes);
      custom_output_ptrs_.push_back(ptr);
      TfLiteCustomAllocation allocation{ptr, subgraph_output->bytes};
      OP_REQUIRES(
          ctx,
          subgraph_selected.SetCustomAllocationForTensor(
              tensor_idx, allocation,
              // Using kTfLiteCustomAllocationFlagsSkipAlignCheck is
              // safe as the pointer comes from TensorFlow.
              // TODO(b/257964109): Remove this flag when fixed.
              kTfLiteCustomAllocationFlagsSkipAlignCheck) == kTfLiteOk,
          errors::Internal(
              "Failed to set custom allocation for output tensor %d, name: %s",
              tensor_idx, subgraph_output->name));
    }
  }

  // Sets custom allocators for all input tensors which are not Resources,
  // Variants or Strings. This means that these tensors can share the same
  // memory as the TF tensors, reducing memcpys and memory usage.
  void SetCustomAllocatorsForInputTensors(OpKernelContext* ctx,
                                          tflite::Subgraph& subgraph_selected) {
    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      int tensor_idx = subgraph_selected.inputs()[i];
      TfLiteTensor* subgraph_input = subgraph_selected.tensor(tensor_idx);
      if (!TensorCanBeShared(subgraph_input)) {
        continue;
      }
      const Tensor& tf_tensor = ctx->input(i + 1);
      TfLiteCustomAllocation allocation{tf_tensor.data(),
                                        tf_tensor.AllocatedBytes()};
      OP_REQUIRES(ctx,
                  subgraph_selected.SetCustomAllocationForTensor(
                      tensor_idx, allocation,
                      // Using kTfLiteCustomAllocationFlagsSkipAlignCheck is
                      // safe as the pointer comes from TensorFlow.
                      // TODO(b/257964109): Remove this flag when fixed.
                      kTfLiteCustomAllocationFlagsSkipAlignCheck) == kTfLiteOk,
                  errors::Internal(
                      "Failed to set custom allocation for input tensor %d",
                      tensor_idx));
    }
  }

  void ResizeInputTensor(OpKernelContext* ctx,
                         tflite::Subgraph& subgraph_selected) {
    for (int i = 0; i < subgraph_selected.inputs().size(); ++i) {
      // Shift index by 1 since the first input is always the resource name.
      const Tensor& tf_tensor = ctx->input(i + 1);
      TfLiteTensor* subgraph_input =
          subgraph_selected.tensor(subgraph_selected.inputs()[i]);

      // Always resize for unranked tensors.
      bool need_resize = (subgraph_input->dims->size == 0);
      if (!need_resize) {
        for (int dim = 0; dim < tf_tensor.shape().dims(); dim++) {
          if (tf_tensor.shape().dim_size(dim) !=
              subgraph_input->dims->data[dim]) {
            need_resize = true;
            break;
          }
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
                        tflite::Subgraph& subgraph_selected,
                        TfLiteDelegate* flex_delegate) const {
    auto InitializeVariantOrResource = [flex_delegate](
                                           const Tensor& tf_tensor,
                                           TfLiteTensor* subgraph_input) {
      // The code here initializes the TfLiteTensor which points the data field
      // to the original TF resource or variant tensor. This requires the TF
      // tensor's lifetime must extend beyond the execution of callee subgraph.
      // TODO(b/179094265): This is an experimental implementation, subject to
      // change. This can be re-implemented with life cycle management
      // mechanism like reference counting.
      const size_t required_bytes =
          tflite::flex::kTensorflowResourceTensorBytes;
      const tensorflow::Tensor** tf_tensor_ptr =
          reinterpret_cast<const tensorflow::Tensor**>(malloc(required_bytes));
      *tf_tensor_ptr = &tf_tensor;

      TfLiteTensorDataFree(subgraph_input);
      subgraph_input->data.raw = reinterpret_cast<char*>(tf_tensor_ptr);
      subgraph_input->bytes = required_bytes;
      subgraph_input->data_is_stale = true;
      subgraph_input->delegate = flex_delegate;
    };

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
      } else if (subgraph_input->type == kTfLiteResource) {
        // Here we will try to parse the input tensor handle to see if it
        // contains a valid TF lite resource ID. If not, then we know that the
        // input is a TF resource tensor.
        tensorflow::ResourceHandle handle =
            tf_tensor.flat<tensorflow::ResourceHandle>()(0);
        if (!tflite::flex::GetTfLiteResourceTensorFromResourceHandle(
                handle, subgraph_input)) {
          InitializeVariantOrResource(tf_tensor, subgraph_input);
        }
      } else if (subgraph_input->type == kTfLiteVariant) {
        InitializeVariantOrResource(tf_tensor, subgraph_input);
      } else if (!TensorCanBeShared(subgraph_input)) {
        absl::string_view tensor_data = tf_tensor.tensor_data();
        OP_REQUIRES(ctx, subgraph_input->bytes == tensor_data.size(),
                    errors::Internal("tensor size doesn't match"));
        // TODO(b/181352924): This could incur some overhead in memory copy.
        // Optimize this away in the future.
        std::memcpy(subgraph_input->data.raw, tensor_data.data(),
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

      // If the output is also an input, create the output tensor using the same
      // buffer as the input tensor so that no copy is needed and to also save
      // memory.
      int output_idx = subgraph_selected.outputs()[i];
      auto it = output_to_input_map_.find(output_idx);
      if (it != output_to_input_map_.end()) {
        const Tensor& tf_tensor = ctx->input(it->second + 1);
        Tensor tensor(tf_tensor);
        ctx->set_output(i, std::move(tensor));
        continue;
      }
      // Take ownership of the TFLite output ptr where possible, otherwise copy
      // the output. This is because the callee subgraph might be invoked
      // repeatedly for each item in the dataset, and the result TfLiteTensor's
      // data should be immediately copied into tensorflow::Tensor.
      Tensor tensor;
      if (output_tensors_can_be_shared_ && TensorCanBeShared(subgraph_output)) {
        tflite::flex::TfLiteTensorBuffer* buf =
            new tflite::flex::TfLiteTensorBuffer(subgraph_output, true);
        buf->TakeOwnershipOfBuffer();
        tensorflow::TensorShape shape;
        int num_dims = subgraph_output->dims->size;
        for (int i = 0; i < num_dims; ++i) {
          OP_REQUIRES_OK(
              ctx, shape.AddDimWithStatus(subgraph_output->dims->data[i]));
        }
        tensor = tensorflow::TensorCApi::MakeTensor(
            tflite::flex::GetTensorFlowDataType(subgraph_output->type), shape,
            buf);
        buf->Unref();
      } else {
        OP_REQUIRES_OK(
            ctx, tflite::flex::SetTfTensorFromTfLite(subgraph_output, &tensor,
                                                     /*allow_reusing=*/false));
      }
      ctx->set_output(i, std::move(tensor));
    }
  }

  // Tells if the target subgraph needs to invoko AllocateTensors().
  bool tfl_tensors_need_allocation_;
  bool fill_output_to_input_map_;
  // NOLINTNEXTLINE - absl::flat_hash_map increases binary size by 106kB.
  std::unordered_map<int, int> output_to_input_map_;
  bool first_run_;
  bool output_tensors_can_be_shared_;
  std::vector<void*> custom_output_ptrs_;
  std::vector<TfLiteAllocationType> original_allocation_type_;
};

REGISTER_KERNEL_BUILDER(Name("TfLiteSubgraphExecute").Device(DEVICE_CPU),
                        TfLiteSubgraphExecute);

}  // namespace tensorflow
