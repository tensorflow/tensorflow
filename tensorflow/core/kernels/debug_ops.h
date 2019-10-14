/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DEBUG_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DEBUG_OPS_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif
#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/debug_events_writer.h"

namespace tensorflow {

// Copy op for debugging.
// Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
// device on which the tensor is allocated.
class CopyOp : public OpKernel {
 public:
  explicit CopyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));

    std::vector<string> debug_ops_spec;
    OP_REQUIRES_OK(context,
                   context->GetAttr("debug_ops_spec", &debug_ops_spec));
    for (const string& debug_op_spec : debug_ops_spec) {
      // Assume debug_op_spec has the format
      // <debug_op>;<debug_url>;<gated_grpc>, e.g.,
      // DebugIdentity;grpc://localhost:3333;1
      const std::vector<string> items = str_util::Split(debug_op_spec, ";");
      OP_REQUIRES(
          context, items.size() == 3,
          errors::Internal(
              "Unexpected number of semicolons in debug_ops_spec element: ",
              debug_op_spec));
      debug_op_and_url_specs_.push_back(
          DebugWatchAndURLSpec(strings::StrCat(tensor_name_, ":", items[0]),
                               items[1], items[2] == "1"));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& src_tensor = context->input(0);

    if (src_tensor.IsInitialized() &&
        DataTypeCanUseMemcpy(src_tensor.dtype()) &&
        DebugIO::IsCopyNodeGateOpen(debug_op_and_url_specs_)) {
      // Source tensor is initialized and is mem-copyable. Make a copy.
      Tensor* copied_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(),
                                                       &copied_tensor));

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      Device* device = static_cast<Device*>(context->device());
      // Determine if the input tensor is not on CPU (e.g., on GPU).
      bool off_host_input = device->device_type() == DEVICE_GPU &&
                            !context->input_alloc_attr(0).on_host();

      if (off_host_input) {
        DeviceContext* device_ctxt = context->op_device_context();
        // Input is not on host: deep-copy it from GPU to the same GPU.
        Notification done_copy;
        GPUUtil::CopyGPUTensorToSameGPU(
            device, device_ctxt, &src_tensor, copied_tensor,
            [&done_copy](const Status& s) { done_copy.Notify(); });
        done_copy.WaitForNotification();
      } else {
        // The input tensor is on the host (CPU): deep-copy from CPU to CPU.
        *copied_tensor = tensor::DeepCopy(src_tensor);
      }
#elif defined(TENSORFLOW_USE_SYCL)
      Device* device = static_cast<Device*>(context->device());
      // Determine if the input tensor is not on CPU (e.g., on GPU).
      const bool off_host_input = device->device_type() == DEVICE_SYCL &&
                                  !context->input_alloc_attr(0).on_host();

      if (off_host_input) {
        SYCLmemcpy(context->eigen_sycl_device(), src_tensor, copied_tensor);
      } else {
        *copied_tensor = tensor::DeepCopy(src_tensor);
      }
#else
      *copied_tensor = tensor::DeepCopy(src_tensor);
#endif
    } else {
      // Source tensor is NOT initialized and/or is not mem-copyable: Forward
      // the Tensor object.
      context->set_output(0, src_tensor);
    }
  }

  bool IsExpensive() override { return false; }

 private:
  string tensor_name_;
  std::vector<DebugWatchAndURLSpec> debug_op_and_url_specs_;
};

// Base class of all debug ops.
class BaseDebugOp : public OpKernel {
 public:
  explicit BaseDebugOp(const string& debug_op_name,
                       OpKernelConstruction* context)
      : OpKernel(context), debug_op_name_(debug_op_name) {
    OP_REQUIRES_OK(context, context->GetAttr("debug_urls", &debug_urls_));
    OP_REQUIRES_OK(context, context->GetAttr("gated_grpc", &gated_grpc_));

    string device_name;
    string tensor_name;
    OP_REQUIRES_OK(context, context->GetAttr("device_name", &device_name));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name));

    std::vector<string> name_items = str_util::Split(tensor_name, ':');
    string node_name;
    int32 output_slot = 0;
    OP_REQUIRES(context, name_items.size() == 1 || name_items.size() == 2,
                errors::InvalidArgument("Failed to parse tensor name: \"",
                                        tensor_name, "\""));
    if (name_items.size() == 2) {
      node_name = name_items[0];
      OP_REQUIRES(
          context, strings::safe_strto32(name_items[1], &output_slot),
          errors::InvalidArgument("Invalid string value for output_slot: \"",
                                  name_items[1], "\""));
    } else if (name_items.size() == 1) {
      node_name = name_items[0];
    }

    debug_watch_key_.reset(
        new DebugNodeKey(device_name, node_name, output_slot, debug_op_name_));
  }

  bool IsExpensive() override { return false; }

 protected:
  // Apply gRPC gating (if gated_grpc_ attribute is true).
  //
  // Returns false if and only if all grpc:// debug URLs of the debug op are
  // disabled currently (i.e., gated off), in which case the debug op will emit
  // an empty (size {0}) tensor of undefined data type.
  bool ApplyGrpcGating(OpKernelContext* context) {
    if (gated_grpc_ && !DebugIO::IsDebugNodeGateOpen(
                           debug_watch_key_->debug_node_name, debug_urls_)) {
      // The entire node is gated off: Output an empty tensor and avoid
      // expensive computation.
      Tensor* output_tensor;
      TensorShape shape({0});
      if (!context->allocate_output(0, shape, &output_tensor).ok()) {
        LOG(ERROR) << "Debug node of watch key "
                   << debug_watch_key_->debug_node_name
                   << " failed to allocate empty tensor under gated-off state.";
      }
      return false;
    } else {
      return true;
    }
  }

  // Publish a tensor to all debug URLs of the debug op.
  // Log an error if the publishing failed.
  Status PublishTensor(const Tensor& tensor) {
    if (debug_urls_.empty()) {
      return Status::OK();
    } else {
      Status status = DebugIO::PublishDebugTensor(*debug_watch_key_, tensor,
                                                  Env::Default()->NowMicros(),
                                                  debug_urls_, gated_grpc_);
      if (!status.ok()) {
        LOG(ERROR) << "Debug node of watch key "
                   << debug_watch_key_->debug_node_name
                   << " failed to publish debug tensor data to all URLs "
                   << str_util::Join(debug_urls_, ", ")
                   << ", due to: " << status.error_message();
      }
      return status;
    }
  }

 private:
  const string debug_op_name_;
  std::unique_ptr<DebugNodeKey> debug_watch_key_;
  std::vector<string> debug_urls_;
  bool gated_grpc_;
};

// Identity op for debugging.
//   Output slot 0 carries the debug signal and is always allocated on the
//   host (CPU) as a non-Ref tensor. In the case of DebugIdentityOp,
//   the debug signal is equal to the input tensor.
class DebugIdentityOp : public BaseDebugOp {
 public:
  explicit DebugIdentityOp(OpKernelConstruction* context)
      : BaseDebugOp("DebugIdentity", context) {}

  void Compute(OpKernelContext* context) override {
    if (!ApplyGrpcGating(context)) {
      return;
    }

    OP_REQUIRES_OK(context, PublishTensor(context->input(0)));
    context->set_output(0, context->input(0));
  }
};

// NaN-counter op for debugging.
template <typename T>
class DebugNanCountOp : public BaseDebugOp {
 public:
  explicit DebugNanCountOp(OpKernelConstruction* context)
      : BaseDebugOp("DebugNanCount", context) {}

  void Compute(OpKernelContext* context) override {
    if (!ApplyGrpcGating(context)) {
      return;
    }

    Tensor* output_tensor;
    const Tensor& input = context->input(0);

    // Use DT_INT64/int64 to be consistent with TensorShape::num_elements().
    int64 nan_count = 0;

    // If the input is an uninitialized tensor, let nan_count be 0.
    if (input.IsInitialized()) {
      // Count NaNs.
      const TensorShape& input_shape = input.shape();
      const T* input_flat = input.template flat<T>().data();

      for (int64 i = 0; i < input_shape.num_elements(); ++i) {
        if (Eigen::numext::isnan(static_cast<double>(input_flat[i]))) {
          nan_count++;
        }
      }
    }

    TensorShape shape({1});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<int64>()(0) = nan_count;
    OP_REQUIRES_OK(context, PublishTensor(*output_tensor));
  }
};

// Numeric summary op for debugging.
template <typename T>
class DebugNumericSummaryOp : public BaseDebugOp {
 public:
  explicit DebugNumericSummaryOp(OpKernelConstruction* context)
      : BaseDebugOp("DebugNumericSummary", context) {
    OP_REQUIRES_OK(context, context->GetAttr("lower_bound", &lower_bound_));
    OP_REQUIRES_OK(context, context->GetAttr("upper_bound", &upper_bound_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("mute_if_healthy", &mute_if_healthy_));
  }

  void Compute(OpKernelContext* context) override {
    if (!ApplyGrpcGating(context)) {
      return;
    }

    Tensor* output_tensor;
    const Tensor& input = context->input(0);

    int64 is_initialized = 0;
    int64 element_count = 0;
    int64 negative_inf_count = 0;
    int64 negative_count = 0;
    int64 zero_count = 0;
    int64 positive_count = 0;
    int64 positive_inf_count = 0;
    int64 nan_count = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double variance = std::numeric_limits<double>::quiet_NaN();

    // Equal to negative_count + zero_count + positive_count.
    int64 non_inf_nan_count = 0;

    const TensorShape& input_shape = input.shape();
    if (input.IsInitialized()) {
      is_initialized = 1;
      const T* input_flat = input.template flat<T>().data();

      element_count = input_shape.num_elements();
      const bool is_lower_bound_custom = !Eigen::numext::isinf(lower_bound_);
      const bool is_upper_bound_custom = !Eigen::numext::isinf(upper_bound_);

      for (int64 i = 0; i < element_count; ++i) {
        const double x = static_cast<double>(input_flat[i]);
        if (Eigen::numext::isnan(x)) {
          nan_count++;
        } else if (Eigen::numext::isinf(x)) {
          if (x < 0.0) {
            negative_inf_count++;
          } else {
            positive_inf_count++;
          }
        } else {
          if (is_lower_bound_custom && x <= lower_bound_) {
            negative_inf_count++;
          } else if (is_upper_bound_custom && x >= upper_bound_) {
            positive_inf_count++;
          } else if (x < 0.0) {
            negative_count++;
          } else if (x > 0.0) {
            positive_count++;
          } else {
            zero_count++;
          }

          if (x < min) {
            min = x;
          }
          if (x > max) {
            max = x;
          }

          non_inf_nan_count++;
          sum += x;
        }
      }

      if (non_inf_nan_count > 0) {
        mean = sum / non_inf_nan_count;

        // Do a second pass to compute variance.
        variance = 0.0;
        for (int64 i = 0; i < element_count; ++i) {
          const double x = static_cast<double>(input_flat[i]);
          if (!Eigen::numext::isnan(x) && !Eigen::numext::isinf(x)) {
            variance += (x - mean) * (x - mean);
          }
        }
        variance /= non_inf_nan_count;
      }
    }

    TensorShape shape({14 + input_shape.dims()});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<double>()(0) = static_cast<double>(is_initialized);
    output_tensor->vec<double>()(1) = static_cast<double>(element_count);
    output_tensor->vec<double>()(2) = static_cast<double>(nan_count);
    output_tensor->vec<double>()(3) = static_cast<double>(negative_inf_count);
    output_tensor->vec<double>()(4) = static_cast<double>(negative_count);
    output_tensor->vec<double>()(5) = static_cast<double>(zero_count);
    output_tensor->vec<double>()(6) = static_cast<double>(positive_count);
    output_tensor->vec<double>()(7) = static_cast<double>(positive_inf_count);
    output_tensor->vec<double>()(8) = min;
    output_tensor->vec<double>()(9) = max;
    output_tensor->vec<double>()(10) = mean;
    output_tensor->vec<double>()(11) = variance;

    output_tensor->vec<double>()(12) = static_cast<double>(input.dtype());
    output_tensor->vec<double>()(13) = static_cast<double>(input_shape.dims());
    for (size_t d = 0; d < input_shape.dims(); ++d) {
      output_tensor->vec<double>()(14 + d) =
          static_cast<double>(input_shape.dim_sizes()[d]);
    }

    bool mute = mute_if_healthy_ && nan_count == 0 && negative_inf_count == 0 &&
                positive_inf_count == 0;
    if (!mute) {
      OP_REQUIRES_OK(context, PublishTensor(*output_tensor));
    }
  }

 private:
  float lower_bound_;
  float upper_bound_;
  bool mute_if_healthy_;
};

// Identity op for tfdbg v2: Writes debug data using DebugEventsWriter.
class DebugIdentityV2Op : public OpKernel {
 public:
  explicit DebugIdentityV2Op(OpKernelConstruction* context)
      : OpKernel(context), output_slot_(-1), tensor_debug_mode_(0) {
    std::vector<string> debug_urls;
    OP_REQUIRES_OK(context, context->GetAttr("debug_urls", &debug_urls));
    for (const string& debug_url : debug_urls) {
      if (absl::StartsWith(debug_url, DebugIO::kFileURLScheme)) {
        dump_roots_.emplace_back(
            debug_url.substr(strlen(DebugIO::kFileURLScheme)));
      } else {
        context->SetStatus(
            errors::Internal("Unsupported debug URL schema in: ", debug_url));
      }
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("tfdbg_context_id", &tfdbg_context_id_));
    OP_REQUIRES_OK(context, context->GetAttr("op_name", &op_name_));
    OP_REQUIRES_OK(context, context->GetAttr("output_slot", &output_slot_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_debug_mode", &tensor_debug_mode_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    for (const string& dump_root : dump_roots_) {
      tfdbg::DebugEventsWriter* debug_events_writer =
          tfdbg::DebugEventsWriter::GetDebugEventsWriter(dump_root);
      debug_events_writer->WriteGraphExecutionTrace(tfdbg_context_id_, op_name_,
                                                    output_slot_,
                                                    tensor_debug_mode_, tensor);
    }
    context->set_output(0, tensor);
  }

 private:
  std::vector<string> dump_roots_;
  string tfdbg_context_id_;
  string op_name_;
  int32 output_slot_;
  int32 tensor_debug_mode_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DEBUG_OPS_H_
