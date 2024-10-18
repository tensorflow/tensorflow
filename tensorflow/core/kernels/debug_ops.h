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

#include <cstdint>
#include <memory>
#include <numeric>

#include "tensorflow/core/platform/bfloat16.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/util/determinism.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/core/platform/cuda.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif

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
    int32_t output_slot = 0;
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
  absl::Status PublishTensor(const Tensor& tensor, int64_t step_id = -1) {
    if (debug_urls_.empty()) {
      return absl::OkStatus();
    } else {
      absl::Status status = DebugIO::PublishDebugTensor(
          *debug_watch_key_, tensor, Env::Default()->NowMicros(), debug_urls_,
          gated_grpc_, step_id);
      if (!status.ok()) {
        LOG(ERROR) << "Debug node of watch key "
                   << debug_watch_key_->debug_node_name
                   << " failed to publish debug tensor data to all URLs "
                   << absl::StrJoin(debug_urls_, ", ")
                   << ", due to: " << status.message();
      }
      return status;
    }
  }

  void CompleteDebugNodeKey(const string& io_of_node, bool is_input,
                            int io_index) {
    debug_watch_key_ = std::make_unique<DebugNodeKey>(
        debug_watch_key_->device_name, debug_watch_key_->node_name,
        debug_watch_key_->output_slot, debug_op_name_, io_of_node, is_input,
        io_index);
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

// Identity op for debugging.
//   Output slot 0 carries the debug signal and is always allocated on the
//   host (CPU) as a non-Ref tensor. In the case of DebugIdentityOp,
//   the debug signal is equal to the input tensor.
class DebugIdentityV3Op : public BaseDebugOp {
 public:
  explicit DebugIdentityV3Op(OpKernelConstruction* context)
      : BaseDebugOp("DebugIdentityV3", context) {
    string io_of_node;
    bool is_input;
    int io_index;
    OP_REQUIRES_OK(context, context->GetAttr("io_of_node", &io_of_node));
    OP_REQUIRES_OK(context, context->GetAttr("is_input", &is_input));
    OP_REQUIRES_OK(context, context->GetAttr("io_index", &io_index));
    if (!io_of_node.empty()) {
      CompleteDebugNodeKey(io_of_node, is_input, io_index);
    }
  }

  void Compute(OpKernelContext* context) override {
    if (!ApplyGrpcGating(context)) {
      return;
    }

    OP_REQUIRES_OK(context,
                   PublishTensor(context->input(0), context->step_id()));
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
    int64_t nan_count = 0;

    // If the input is an uninitialized tensor, let nan_count be 0.
    if (input.IsInitialized()) {
      // Count NaNs.
      const TensorShape& input_shape = input.shape();
      const T* input_flat = input.template flat<T>().data();

      for (int64_t i = 0; i < input_shape.num_elements(); ++i) {
        if (Eigen::numext::isnan(static_cast<double>(input_flat[i]))) {
          nan_count++;
        }
      }
    }

    TensorShape shape({1});
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    output_tensor->vec<int64_t>()(0) = nan_count;
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

    int64_t is_initialized = 0;
    int64_t element_count = 0;
    int64_t negative_inf_count = 0;
    int64_t negative_count = 0;
    int64_t zero_count = 0;
    int64_t positive_count = 0;
    int64_t positive_inf_count = 0;
    int64_t nan_count = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double mean = std::numeric_limits<double>::quiet_NaN();
    double variance = std::numeric_limits<double>::quiet_NaN();

    // Equal to negative_count + zero_count + positive_count.
    int64_t non_inf_nan_count = 0;

    const TensorShape& input_shape = input.shape();
    if (input.IsInitialized()) {
      is_initialized = 1;
      const T* input_flat = input.template flat<T>().data();

      element_count = input_shape.num_elements();
      const bool is_lower_bound_custom = !Eigen::numext::isinf(lower_bound_);
      const bool is_upper_bound_custom = !Eigen::numext::isinf(upper_bound_);

      for (int64_t i = 0; i < element_count; ++i) {
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
        for (int64_t i = 0; i < element_count; ++i) {
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
      : OpKernel(context),
        device_name_(context->device()->name()),
        output_slot_(-1),
        tensor_debug_mode_(0),
        tfdbg_run_id_() {
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
    if (context->HasAttr("circular_buffer_size")) {
      OP_REQUIRES_OK(context, context->GetAttr("circular_buffer_size",
                                               &circular_buffer_size_));
    } else {
      circular_buffer_size_ =
          tfdbg::DebugEventsWriter::kDefaultCyclicBufferSize;
    }
    if (context->HasAttr("tfdbg_run_id")) {
      OP_REQUIRES_OK(context, context->GetAttr("tfdbg_run_id", &tfdbg_run_id_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    for (const string& dump_root : dump_roots_) {
      tfdbg::DebugEventsWriter* debug_events_writer =
          tfdbg::DebugEventsWriter::GetDebugEventsWriter(
              dump_root, tfdbg_run_id_, circular_buffer_size_);
      OP_REQUIRES_OK(context, debug_events_writer->WriteGraphExecutionTrace(
                                  tfdbg_context_id_, device_name_, op_name_,
                                  output_slot_, tensor_debug_mode_, tensor));
    }
    context->set_output(0, tensor);
  }

 private:
  std::vector<string> dump_roots_;
  string tfdbg_context_id_;
  string device_name_;
  string op_name_;
  int32 output_slot_;
  int32 tensor_debug_mode_;
  int64_t circular_buffer_size_;
  string tfdbg_run_id_;
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename Tin, typename Tout>
struct CurtHealthLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[1]);
};

extern template struct CurtHealthLaunch<Eigen::half, float>;
extern template struct CurtHealthLaunch<float, float>;
extern template struct CurtHealthLaunch<double, float>;
extern template struct CurtHealthLaunch<Eigen::half, double>;
extern template struct CurtHealthLaunch<float, double>;
extern template struct CurtHealthLaunch<double, double>;

template <typename Tin, typename Tout>
struct ConciseHealthLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[3]);
};

extern template struct ConciseHealthLaunch<Eigen::half, float>;
extern template struct ConciseHealthLaunch<float, float>;
extern template struct ConciseHealthLaunch<double, float>;
extern template struct ConciseHealthLaunch<Eigen::half, double>;
extern template struct ConciseHealthLaunch<float, double>;
extern template struct ConciseHealthLaunch<double, double>;

template <typename Tin, typename Tout>
struct FullHealthLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[6]);
};

extern template struct FullHealthLaunch<Eigen::half, float>;
extern template struct FullHealthLaunch<float, float>;
extern template struct FullHealthLaunch<double, float>;
extern template struct FullHealthLaunch<Eigen::half, double>;
extern template struct FullHealthLaunch<float, double>;
extern template struct FullHealthLaunch<double, double>;

template <typename Tin, typename Tout>
struct ReduceInfNanThreeSlotsLaunch {
  void Run(const GPUDevice& d, const Tin* data, int size, Tout output[3]);
};

extern template struct ReduceInfNanThreeSlotsLaunch<Eigen::half, float>;
extern template struct ReduceInfNanThreeSlotsLaunch<float, float>;
extern template struct ReduceInfNanThreeSlotsLaunch<double, float>;
extern template struct ReduceInfNanThreeSlotsLaunch<Eigen::half, double>;
extern template struct ReduceInfNanThreeSlotsLaunch<float, double>;
extern template struct ReduceInfNanThreeSlotsLaunch<double, double>;

#endif

template <typename Device, typename Tin, typename Tout>
class DebugNumericSummaryV2Op;

// Numeric summary op for tfdbg v2: CPU Kernel.
template <typename Tin, typename Tout>
class DebugNumericSummaryV2Op<CPUDevice, Tin, Tout> : public OpKernel {
 public:
  explicit DebugNumericSummaryV2Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_debug_mode", &tensor_debug_mode_));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_id", &tensor_id_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    auto in = tensor.flat<Tin>();
    const Tin* data = in.data();
    const int64_t size = in.size();
    Tensor* output_tensor;
    Tout tensor_id = static_cast<Tout>(tensor_id_);
    const Tout num_elem = static_cast<Tout>(context->input(0).NumElements());
    // Disregard lossy cast if mode is REDUCE_INF_NAN_THREE_SLOTS because
    // that mode does not make use of tensor_id.
    if (tensor_debug_mode_ != 8) {
      OP_REQUIRES(
          context, tensor_id_ <= kMaxTensorId,
          errors::InvalidArgument("DebugNumericSummaryV2Op requires "
                                  "tensor_id to be less than or equal to "
                                  "(2^",
                                  std::numeric_limits<Tout>::digits,
                                  "). Given tensor_id:", tensor_id_));
    }

    if (tensor_debug_mode_ == 2) {  // CURT_HEALTH
      TensorShape shape({2});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));
      output_tensor->flat<Tout>()(0) = tensor_id;  // Slot tensor id
      output_tensor->flat<Tout>()(1) = 0.0;        // Has inf or nan
      int fp_props =
          std::accumulate(data, data + size, 0, [](const int x, const Tin& y) {
            return Eigen::numext::isfinite(y) ? x : 1;
          });
      if (fp_props) {
        output_tensor->flat<Tout>()(1) = 1.0;
      }
    } else if (tensor_debug_mode_ == 3) {  // CONCISE_HEALTH
      TensorShape shape({5});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));
      output_tensor->flat<Tout>()(0) = tensor_id;
      output_tensor->flat<Tout>()(1) = num_elem;

      // Accumulator value [neg_inf_count, pos_inf_count, nan_count]
      Tout fp_props[3] = {0.0, 0.0, 0.0};
      std::for_each(data, data + size, [&fp_props](const Tin& y) {
        if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
          // Do nothing: common case.
        } else if (Eigen::numext::isinf(y)) {
          if (y < static_cast<Tin>(0.f)) {
            ++fp_props[0];
          } else {
            ++fp_props[1];
          }
        } else if (Eigen::numext::isnan(y)) {
          ++fp_props[2];
        }
      });
      output_tensor->flat<Tout>()(2) = fp_props[0];  // Slot for -inf count
      output_tensor->flat<Tout>()(3) = fp_props[1];  // Slot for inf count
      output_tensor->flat<Tout>()(4) = fp_props[2];  // Slot for nan count
    } else if (tensor_debug_mode_ == 4) {            // FULL HEALTH
      TensorShape shape({11});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));
      int num_dims = tensor.dims();
      output_tensor->flat<Tout>()(0) = tensor_id;
      output_tensor->flat<Tout>()(1) = -1.0;  // TODO(144919262): Device ID
      output_tensor->flat<Tout>()(2) = static_cast<Tout>(tensor.dtype());
      output_tensor->flat<Tout>()(3) = static_cast<Tout>(num_dims);
      output_tensor->flat<Tout>()(4) = num_elem;

      // Accumulator value [neg_inf_count, pos_inf_count, nan_count, neg_count,
      //                   zero_count, pos_count]
      Tout fp_props[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      std::for_each(data, data + size, [&fp_props](const Tin& y) {
        if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
          if (y < static_cast<Tin>(0.f)) {
            ++fp_props[3];
          } else if (y == static_cast<Tin>(0.f)) {
            ++fp_props[4];
          } else {
            ++fp_props[5];
          }
        } else if (Eigen::numext::isinf(y)) {
          if (y < static_cast<Tin>(0.f)) {
            ++fp_props[0];
          } else {
            ++fp_props[1];
          }
        } else if (Eigen::numext::isnan(y)) {
          ++fp_props[2];
        }
      });
      output_tensor->flat<Tout>()(5) = fp_props[0];   // Slot for -inf count
      output_tensor->flat<Tout>()(6) = fp_props[1];   // Slot for inf count
      output_tensor->flat<Tout>()(7) = fp_props[2];   // Slot for nan count.
      output_tensor->flat<Tout>()(8) = fp_props[3];   // Slot for neg count.
      output_tensor->flat<Tout>()(9) = fp_props[4];   // Slot for zero count.
      output_tensor->flat<Tout>()(10) = fp_props[5];  // Slot for pos count.
    } else if (tensor_debug_mode_ == 5) {             // SHAPE
      TensorShape shape({10});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      int num_dims = tensor.dims();
      output_tensor->flat<Tout>()(0) = tensor_id;
      output_tensor->flat<Tout>()(1) = static_cast<Tout>(tensor.dtype());
      output_tensor->flat<Tout>()(2) = static_cast<Tout>(num_dims);
      output_tensor->flat<Tout>()(3) = num_elem;

      // Tensor shape - stored as (6 columns)
      // if num_dim is less than 6, we right pad the shape with zeros
      // if num_dim is greater than 6, we truncate the head (left most) of the
      // dimensions as they are more predictable than the last few (e.g. batch
      // size as first dimension)
      int dim_idx = 4;
      for (int i = std::max(0, num_dims - kShapeDims);
           i < std::max(6, num_dims); ++i) {
        if (i < num_dims) {
          output_tensor->flat<Tout>()(dim_idx++) =
              static_cast<Tout>(tensor.dim_size(i));
        } else {
          output_tensor->flat<Tout>()(dim_idx++) = 0.0;
        }
      }
    } else if (tensor_debug_mode_ == 8) {  // REDUCE_INF_NAN_THREE_SLOTS.
      TensorShape shape({3});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));
      output_tensor->flat<Tout>()(0) = 0.0;  // Slot for -inf.
      output_tensor->flat<Tout>()(1) = 0.0;  // Slot for inf.
      output_tensor->flat<Tout>()(2) = 0.0;  // Slot for nan.

      int fp_props =
          std::accumulate(data, data + size, 0, [](const int x, const Tin& y) {
            int result = x;
            if (TF_PREDICT_TRUE(Eigen::numext::isfinite(y))) {
              // Do nothing: common case.
            } else if (Eigen::numext::isinf(y)) {
              result |= y < static_cast<Tin>(0.f) ? kNegInfBit : kPosInfBit;
            } else if (Eigen::numext::isnan(y)) {
              result |= kNaNBit;
            }
            return result;
          });

      if (fp_props & kNegInfBit) {
        output_tensor->flat<Tout>()(0) = -std::numeric_limits<Tout>::infinity();
      }
      if (fp_props & kPosInfBit) {
        output_tensor->flat<Tout>()(1) = std::numeric_limits<Tout>::infinity();
      }
      if (fp_props & kNaNBit) {
        output_tensor->flat<Tout>()(2) = std::numeric_limits<Tout>::quiet_NaN();
      }
    } else {
      // TODO(cais): Implement other tensor debug modes in debug_event.proto.
      context->SetStatus(errors::Unimplemented(
          "Unimplemented tensor debug mode: ", tensor_debug_mode_));
    }
  }

 private:
  int tensor_debug_mode_;
  int64_t tensor_id_;
  static constexpr int kShapeDims = 6;
  static constexpr int kNegInfBit = 0x01;
  static constexpr int kPosInfBit = 0x02;
  static constexpr int kNaNBit = 0x04;
  static constexpr int64_t kMaxTensorId = 1LL
                                          << std::numeric_limits<Tout>::digits;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Tin, typename Tout>
class DebugNumericSummaryV2Op<GPUDevice, Tin, Tout> : public AsyncOpKernel {
 public:
  typedef GPUDevice Device;

  explicit DebugNumericSummaryV2Op(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_debug_mode", &tensor_debug_mode_));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_id", &tensor_id_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    Tensor* output_tensor;
    Tout tensor_id = static_cast<Tout>(tensor_id_);
    const Tensor& tensor = context->input(0);
    const Tout num_elem = static_cast<Tout>(tensor.NumElements());
    const Device& d = context->eigen_device<Device>();
    auto input = tensor.flat<Tin>();
    auto check_cb = [this, done]() { done(); };
    // Disregard lossy cast if mode is REDUCE_INF_NAN_THREE_SLOTS because
    // that mode does not make use of tensor_id.
    if (tensor_debug_mode_ != 8) {
      OP_REQUIRES_ASYNC(
          context, tensor_id_ <= kMaxTensorId,
          errors::InvalidArgument("DebugNumericSummaryV2Op requires "
                                  "tensor_id to be less than or equal to "
                                  "(2^",
                                  std::numeric_limits<Tout>::digits,
                                  "). Given tensor_id:", tensor_id_),
          done);
    }

    if (tensor_debug_mode_ == 2) {  // CURT_HEALTH.
      TensorShape shape({2});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      auto* stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);

      se::DeviceMemoryBase output_tensor_ptr(
          output_tensor->flat<Tout>().data(),
          output_tensor->flat<Tout>().size());
      OP_REQUIRES_OK(context,
                     stream->MemZero(&output_tensor_ptr, 2 * sizeof(Tout)));
      // Copy tensor_id to slot zero
      OP_REQUIRES_OK(context, stream->Memcpy(&output_tensor_ptr, &tensor_id,
                                             sizeof(Tout)));
      if (num_elem == 0) {
        done();
        return;
      }

      // Call the GPU kernels for the numerical (inf/nan) checks.
      auto input = context->input(0).flat<Tin>();
      CurtHealthLaunch<Tin, Tout>().Run(d, input.data(), input.size(),
                                        output_tensor->flat<Tout>().data() + 1);

      context->device()
          ->tensorflow_accelerator_device_info()
          ->event_mgr->ThenExecute(stream, std::move(check_cb));
    } else if (tensor_debug_mode_ == 3) {  // CONCISE_HEALTH.
      TensorShape shape({5});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));
      OP_REQUIRES_ASYNC(context, !tensorflow::OpDeterminismRequired(),
                        errors::Unimplemented(
                            "Determinism is not yet supported for "
                            "DebugNumericSummaryV2 when tensor_debug_mode is "
                            "CONCISE_HEALTH."),
                        done);

      auto* stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);

      se::DeviceMemoryBase output_tensor_ptr(
          output_tensor->flat<Tout>().data(),
          output_tensor->flat<Tout>().size());
      OP_REQUIRES_OK(context,
                     stream->Memset32(&output_tensor_ptr, 0, 5 * sizeof(Tout)));
      const Tout static_output[] = {tensor_id, num_elem};
      OP_REQUIRES_OK(context, stream->Memcpy(&output_tensor_ptr, &static_output,
                                             2 * sizeof(Tout)));
      if (num_elem == 0) {
        done();
        return;
      }

      // Call the GPU kernels for the numerical (inf/nan) checks.
      ConciseHealthLaunch<Tin, Tout>().Run(
          d, input.data(), input.size(),
          output_tensor->flat<Tout>().data() + 2);

      context->device()
          ->tensorflow_accelerator_device_info()
          ->event_mgr->ThenExecute(stream, std::move(check_cb));
    } else if (tensor_debug_mode_ == 4) {  // FULL HEALTH
      TensorShape shape({11});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      auto* stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);
      OP_REQUIRES_ASYNC(context, !tensorflow::OpDeterminismRequired(),
                        errors::Unimplemented(
                            "Determinism is not yet supported for "
                            "DebugNumericSummaryV2 when tensor_debug_mode is "
                            "FULL_HEALTH."),
                        done);

      se::DeviceMemoryBase output_tensor_ptr(
          output_tensor->flat<Tout>().data(),
          output_tensor->flat<Tout>().size());
      OP_REQUIRES_OK(
          context, stream->Memset32(&output_tensor_ptr, 0, 11 * sizeof(Tout)));

      int num_dims = tensor.dims();
      const Tout static_output[] = {tensor_id,
                                    -1.0,  // TODO(144919262): Device ID
                                    static_cast<Tout>(tensor.dtype()),
                                    static_cast<Tout>(num_dims), num_elem};
      OP_REQUIRES_OK(context, stream->Memcpy(&output_tensor_ptr, &static_output,
                                             5 * sizeof(Tout)));
      if (num_elem == 0) {
        done();
        return;
      }

      // Call the GPU kernels for the numerical (inf/nan) checks and
      // pos/neg/zero counts.
      FullHealthLaunch<Tin, Tout>().Run(d, input.data(), input.size(),
                                        output_tensor->flat<Tout>().data() + 5);

      context->device()
          ->tensorflow_accelerator_device_info()
          ->event_mgr->ThenExecute(stream, std::move(check_cb));
    } else if (tensor_debug_mode_ == 5) {  // SHAPE
      TensorShape shape({10});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      auto* stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);

      se::DeviceMemoryBase output_tensor_ptr(
          output_tensor->flat<Tout>().data(),
          output_tensor->flat<Tout>().size());

      int num_dims = tensor.dims();
      Tout static_output[10] = {tensor_id,
                                static_cast<Tout>(tensor.dtype()),
                                static_cast<Tout>(num_dims),
                                num_elem,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0};
      // Tensor shape: right pad zeros, truncate head
      int dim_idx = 4;
      for (int i = std::max(0, num_dims - 6); i < num_dims; ++i) {
        static_output[dim_idx++] = static_cast<Tout>(tensor.dim_size(i));
      }
      // Write to device stream
      OP_REQUIRES_OK(context, stream->Memcpy(&output_tensor_ptr, &static_output,
                                             sizeof(Tout) * 10));
      context->device()
          ->tensorflow_accelerator_device_info()
          ->event_mgr->ThenExecute(stream, std::move(check_cb));
    } else if (tensor_debug_mode_ == 8) {  // REDUCE_INF_NAN_THREE_SLOTS.
      TensorShape shape({3});
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, shape, &output_tensor));

      auto* stream = context->op_device_context()->stream();
      OP_REQUIRES_ASYNC(context, stream != nullptr,
                        errors::Internal("No GPU stream available."), done);

      se::DeviceMemoryBase output_tensor_ptr(
          output_tensor->flat<Tout>().data(),
          output_tensor->flat<Tout>().size());
      OP_REQUIRES_OK(
          context,
          stream->Memset32(&output_tensor_ptr, 0,
                           output_tensor->flat<Tout>().size() * sizeof(Tout)));
      if (num_elem == 0) {
        done();
        return;
      }

      // Call the GPU kernels for the numerical (inf/nan) checks.
      auto input = context->input(0).flat<Tin>();
      ReduceInfNanThreeSlotsLaunch<Tin, Tout>().Run(
          d, input.data(), input.size(), output_tensor->flat<Tout>().data());

      context->device()
          ->tensorflow_accelerator_device_info()
          ->event_mgr->ThenExecute(stream, std::move(check_cb));
    } else {
      // TODO(cais): Implement other tensor debug modes in debug_event.proto.
      context->SetStatus(errors::Unimplemented(
          "Unimplemented tensor debug mode: ", tensor_debug_mode_));
      done();
    }
  }

 private:
  int tensor_debug_mode_;
  int64_t tensor_id_;
  static constexpr int64_t kMaxTensorId = 1L
                                          << std::numeric_limits<Tout>::digits;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DEBUG_OPS_H_
