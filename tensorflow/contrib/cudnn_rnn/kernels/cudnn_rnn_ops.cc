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
#define EIGEN_USE_THREADS

#include <stddef.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif  // GOOGLE_CUDA

/*
 * This module implements ops that fuse a multi-layer multi-step RNN/LSTM model
 * using the underlying Cudnn library.
 *
 * Cudnn RNN library exposes an opaque parameter buffer with unknown layout and
 * format. And it is very likely that if saved, they cannot be used across
 * different GPUs. So users need to first query the size of the opaque
 * parameter buffer, and convert it to and from its canonical forms. But each
 * actual training step is carried out with the parameter buffer.
 *
 * Similar to many other ops, the forward op has two flavors: training and
 * inference. When training is specified, additional data in reserve_space will
 * be produced for the backward pass. So there is a performance penalty.
 *
 * In addition to the actual data and reserve_space, Cudnn also needs more
 * memory as temporary workspace. The memory management to and from
 * stream-executor is done through ScratchAllocator. In general,
 * stream-executor is responsible for creating the memory of proper size. And
 * TensorFlow is responsible for making sure the memory is alive long enough
 * and recycles afterwards.
 *
 */
namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

#if GOOGLE_CUDA

using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T, typename Index>
class CudnnRNNParamsSizeOp;

template <typename Device, typename T>
class CudnnRNNParamsToCanonical;

template <typename Device, typename T>
class CudnnRNNCanonicalToParams;

template <typename Device, typename T>
class CudnnRNNForwardOp;

template <typename Device, typename T>
class CudnnRNNBackwardOp;

enum class TFRNNInputMode {
  kRNNLinearInput = 0,
  kRNNSkipInput = 1,
  kAutoSelect = 9999999
};

namespace {
using perftools::gputools::DeviceMemory;
using perftools::gputools::DeviceMemoryBase;
using perftools::gputools::ScratchAllocator;
using perftools::gputools::dnn::RnnDirectionMode;
using perftools::gputools::dnn::RnnInputMode;
using perftools::gputools::dnn::RnnMode;
using perftools::gputools::dnn::ToDataType;
using perftools::gputools::port::StatusOr;

Status ParseRNNMode(const string& str, RnnMode* rnn_mode) {
  if (str == "rnn_relu") {
    *rnn_mode = RnnMode::kRnnRelu;
    return Status::OK();
  } else if (str == "rnn_tanh") {
    *rnn_mode = RnnMode::kRnnTanh;
    return Status::OK();
  } else if (str == "lstm") {
    *rnn_mode = RnnMode::kRnnLstm;
    return Status::OK();
  } else if (str == "gru") {
    *rnn_mode = RnnMode::kRnnGru;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN mode: ", str);
}

Status ParseTFRNNInputMode(const string& str, TFRNNInputMode* rnn_input_mode) {
  if (str == "linear_input") {
    *rnn_input_mode = TFRNNInputMode::kRNNLinearInput;
    return Status::OK();
  } else if (str == "skip_input") {
    *rnn_input_mode = TFRNNInputMode::kRNNSkipInput;
    return Status::OK();
  } else if (str == "auto_select") {
    *rnn_input_mode = TFRNNInputMode::kAutoSelect;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN input mode: ", str);
}

Status ParseRNNDirectionMode(const string& str,
                             RnnDirectionMode* rnn_dir_mode) {
  if (str == "unidirectional") {
    *rnn_dir_mode = RnnDirectionMode::kRnnUnidirectional;
    return Status::OK();
  } else if (str == "bidirectional") {
    *rnn_dir_mode = RnnDirectionMode::kRnnBidirectional;
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid RNN direction mode: ", str);
}

Status ToRNNInputMode(TFRNNInputMode tf_input_mode, int num_units,
                      int input_size, RnnInputMode* input_mode) {
  switch (tf_input_mode) {
    case TFRNNInputMode::kRNNLinearInput:
      *input_mode = RnnInputMode::kRnnLinearSkip;
      break;
    case TFRNNInputMode::kRNNSkipInput:
      *input_mode = RnnInputMode::kRnnSkipInput;
      break;
    case TFRNNInputMode::kAutoSelect:
      *input_mode = (input_size == num_units) ? RnnInputMode::kRnnSkipInput
                                              : RnnInputMode::kRnnLinearSkip;
      break;
    default:
      return errors::InvalidArgument("Invalid TF input mode: ",
                                     static_cast<int>(tf_input_mode));
  }
  return Status::OK();
}

// TODO(zhengxq): Merge those into stream_executor_util.h.
template <typename T>
const DeviceMemory<T> AsDeviceMemory(const Tensor* tensor) {
  return DeviceMemory<T>::MakeFromByteSize(
      const_cast<T*>(tensor->template flat<T>().data()),
      tensor->template flat<T>().size() * sizeof(T));
}

template <typename T>
DeviceMemory<T> AsDeviceMemory(Tensor* tensor) {
  return DeviceMemory<T>::MakeFromByteSize(
      tensor->template flat<T>().data(),
      tensor->template flat<T>().size() * sizeof(T));
}

template <typename U, typename T>
DeviceMemory<U> CastDeviceMemory(Tensor* tensor) {
  return DeviceMemory<U>::MakeFromByteSize(
      tensor->template flat<T>().data(),
      tensor->template flat<T>().size() * sizeof(T));
}

DeviceMemoryBase SliceDeviceMemory(const DeviceMemoryBase& device_memory,
                                   int64 offset, int64 size) {
  const void* base_ptr = device_memory.opaque();
  void* offset_ptr =
      const_cast<char*>(reinterpret_cast<const char*>(base_ptr) + offset);
  CHECK(offset + size <= device_memory.size())
      << "The slice is not within the region of DeviceMemory.";
  return DeviceMemoryBase(offset_ptr, size);
}

inline Status FromExecutorStatus(const perftools::gputools::port::Status& s) {
  return s.ok() ? Status::OK()
                : Status(static_cast<tensorflow::error::Code>(
                             static_cast<int>(s.code())),
                         s.error_message());
}

template <typename T>
inline Status FromExecutorStatus(
    const perftools::gputools::port::StatusOr<T>& s) {
  return FromExecutorStatus(s.status());
}

inline perftools::gputools::port::Status ToExecutorStatus(const Status& s) {
  return s.ok() ? perftools::gputools::port::Status::OK()
                : perftools::gputools::port::Status(
                      static_cast<perftools::gputools::port::error::Code>(
                          static_cast<int>(s.code())),
                      s.error_message());
}

// A helper to allocate temporary scratch memory for Cudnn RNN models. It takes
// the ownership of the underlying memory. The expectation is that the memory
// should be alive for the span of the Cudnn RNN itself.
class CudnnRNNWorkspaceAllocator : public ScratchAllocator {
 public:
  ~CudnnRNNWorkspaceAllocator() override {}
  explicit CudnnRNNWorkspaceAllocator(OpKernelContext* context)
      : context_(context) {}
  int64 GetMemoryLimitInBytes(perftools::gputools::Stream* stream) override {
    return std::numeric_limits<int64>::max();
  }
  StatusOr<DeviceMemory<uint8>> AllocateBytes(
      perftools::gputools::Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return StatusOr<DeviceMemory<uint8>>(
        AsDeviceMemory<uint8>(&temporary_memory));
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  std::vector<Tensor> allocated_tensors_;
};

// A helper to allocate reserve-space memory for Cudnn RNN models. The tensors
// are allocated as a kernel output, and will be fed into the backward pass.
// The memory is expected to live long enough after the backward pass is
// finished.
template <typename T>
class CudnnRNNReserveSpaceAllocator : public ScratchAllocator {
 public:
  ~CudnnRNNReserveSpaceAllocator() override {}
  CudnnRNNReserveSpaceAllocator(OpKernelContext* context, int output_index)
      : context_(context), output_index_(output_index) {}
  int64 GetMemoryLimitInBytes(perftools::gputools::Stream* stream) override {
    return std::numeric_limits<int64>::max();
  }
  StatusOr<DeviceMemory<uint8>> AllocateBytes(
      perftools::gputools::Stream* stream, int64 byte_size) override {
    CHECK(total_byte_size_ == 0)
        << "Reserve space allocator can only be called once";
    int64 allocate_count =
        Eigen::divup(byte_size, static_cast<int64>(sizeof(T)));

    Tensor* temporary_memory = nullptr;
    Status allocation_status(context_->allocate_output(
        output_index_, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    total_byte_size_ += byte_size;
    auto memory_uint8 = DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory->template flat<T>().data(),
        temporary_memory->template flat<T>().size() * sizeof(T));
    return StatusOr<DeviceMemory<uint8>>(memory_uint8);
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  int output_index_;
};

// A helper to allocate persistent memory for Cudnn RNN models, which is
// expected to live between kernel invocations.
// This class is not thread-safe.
class CudnnRNNPersistentSpaceAllocator : public ScratchAllocator {
 public:
  explicit CudnnRNNPersistentSpaceAllocator(OpKernelContext* context)
      : context_(context) {}

  ~CudnnRNNPersistentSpaceAllocator() override {}

  int64 GetMemoryLimitInBytes(perftools::gputools::Stream* stream) override {
    return std::numeric_limits<int64>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(
      perftools::gputools::Stream* stream, int64 byte_size) override {
    if (total_byte_size_ != 0) {
      return Status(error::FAILED_PRECONDITION,
                    "Persistent space allocator can only be called once");
    }

    Status allocation_status = context_->allocate_persistent(
        DT_UINT8, TensorShape({byte_size}), &handle_, nullptr);
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    total_byte_size_ += byte_size;
    return AsDeviceMemory<uint8>(handle_.AccessTensor(context_));
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 total_byte_size_ = 0;
  PersistentTensor handle_;
  OpKernelContext* context_;  // not owned
};

struct CudnnModelTypes {
  RnnMode rnn_mode;
  TFRNNInputMode rnn_input_mode;
  RnnDirectionMode rnn_direction_mode;
  bool HasInputC() const {
    // For Cudnn 5.0, only LSTM has input-c. All other models use only input-h.
    return rnn_mode == RnnMode::kRnnLstm;
  }
};

// A helper class that collects the shapes to describe a RNN model.
struct CudnnModelShapes {
  int num_layers;
  int input_size;
  int num_units;
  int seq_length;
  int batch_size;
  int dir_count;
  TensorShape input_shape;
  TensorShape output_shape;
  TensorShape hidden_state_shape;
  // At present only fields related to cached RnnDescriptor are concerned.
  bool IsCompatibleWith(const CudnnModelShapes& rhs) const {
    return num_layers == rhs.num_layers && input_size == rhs.input_size &&
           num_units == rhs.num_units && dir_count == rhs.dir_count;
  }
  string RnnDescDebugString() {
    return strings::Printf(
        "[num_layers, input_size, num_units, dir_count]: [%d, %d, %d, %d]",
        num_layers, input_size, num_units, dir_count);
  }
};

// Extract and checks the forward input tensors, parameters, and shapes from the
// OpKernelContext.
Status ExtractForwardInput(OpKernelContext* context,
                           const CudnnModelTypes& model_types,
                           const Tensor** input, const Tensor** input_h,
                           const Tensor** input_c, const Tensor** params,
                           CudnnModelShapes* model_shapes) {
  TF_RETURN_IF_ERROR(context->input("input", input));
  TF_RETURN_IF_ERROR(context->input("input_h", input_h));
  if (model_types.HasInputC()) {
    TF_RETURN_IF_ERROR(context->input("input_c", input_c));
  }
  TF_RETURN_IF_ERROR(context->input("params", params));

  if ((*input)->dims() != 3) {
    return errors::InvalidArgument("RNN input must be a 3-D vector.");
  }
  model_shapes->seq_length = (*input)->dim_size(0);
  model_shapes->batch_size = (*input)->dim_size(1);
  model_shapes->input_size = (*input)->dim_size(2);
  model_shapes->input_shape = (*input)->shape();
  model_shapes->dir_count =
      (model_types.rnn_direction_mode == RnnDirectionMode::kRnnBidirectional)
          ? 2
          : 1;

  if ((*input_h)->dims() != 3) {
    return errors::InvalidArgument("RNN input must be a 3-D vector.");
  }
  model_shapes->num_layers = (*input_h)->dim_size(0) / model_shapes->dir_count;
  model_shapes->num_units = (*input_h)->dim_size(2);

  model_shapes->hidden_state_shape =
      TensorShape({model_shapes->dir_count * model_shapes->num_layers,
                   model_shapes->batch_size, model_shapes->num_units});
  if ((*input_h)->shape() != model_shapes->hidden_state_shape) {
    return errors::InvalidArgument(
        "Invalid input_h shape: ", (*input_h)->shape().DebugString(), " ",
        model_shapes->hidden_state_shape.DebugString());
  }
  if (model_types.HasInputC()) {
    if ((*input_h)->shape() != (*input_c)->shape()) {
      return errors::InvalidArgument(
          "input_h and input_c must have the same shape: ",
          (*input_h)->shape().DebugString(), " ",
          (*input_c)->shape().DebugString());
    }
  }
  model_shapes->output_shape =
      TensorShape({model_shapes->seq_length, model_shapes->batch_size,
                   model_shapes->dir_count * model_shapes->num_units});
  return Status::OK();
}

using perftools::gputools::dnn::RnnDescriptor;

template <typename T>
void RestoreParams(const OpInputList params_input,
                   const std::vector<RnnDescriptor::ParamsRegion>& params,
                   DeviceMemoryBase* data_dst,
                   perftools::gputools::Stream* stream) {
  int num_params = params.size();
  CHECK(params_input.size() == num_params)
      << "Number of params mismatch. Expected " << params_input.size()
      << ", got " << num_params;
  for (int i = 0; i < params.size(); i++) {
    int64 size_in_bytes = params[i].size;
    int64 size = size_in_bytes / sizeof(T);
    CHECK(size == params_input[i].NumElements())
        << "Params size mismatch. Expected " << size << ", got "
        << params_input[i].NumElements();
    auto data_src_ptr = StreamExecutorUtil::AsDeviceMemory<T>(params_input[i]);
    DeviceMemoryBase data_dst_ptr =
        SliceDeviceMemory(*data_dst, params[i].offset, size_in_bytes);
    stream->ThenMemcpy(&data_dst_ptr, data_src_ptr, size_in_bytes);
  }
}

}  // namespace

// Note: all following kernels depend on a RnnDescriptor instance, which
// according to Cudnn official doc should be kept around and reused across all
// Cudnn kernels in the same model.
// In Tensorflow, we don't pass the reference across different OpKernels,
// rather, recreate it separately in each OpKernel, which does no cause issue:
// CudnnDropoutDescriptor keeps a reference to a memory for
// random number generator state. During recreation, this state is lost.
// However, only forward-pass Cudnn APIs make use of the state.

// A common base class for RNN kernels. It extracts common attributes and
// shape validations.
class CudnnRNNKernelCommon : public OpKernel {
 protected:
  explicit CudnnRNNKernelCommon(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout", &dropout_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(context, context->GetAttr("seed2", &seed2_));
    string str;
    OP_REQUIRES_OK(context, context->GetAttr("rnn_mode", &str));
    OP_REQUIRES_OK(context, ParseRNNMode(str, &model_types_.rnn_mode));
    OP_REQUIRES_OK(context, context->GetAttr("input_mode", &str));
    OP_REQUIRES_OK(context,
                   ParseTFRNNInputMode(str, &model_types_.rnn_input_mode));
    OP_REQUIRES_OK(context, context->GetAttr("direction", &str));
    OP_REQUIRES_OK(
        context, ParseRNNDirectionMode(str, &model_types_.rnn_direction_mode));
    // Reset CudnnRnnDescriptor and related random number generate states in
    // every Compute() call.
    OP_REQUIRES_OK(context, ReadBoolFromEnvVar("TF_CUDNN_RESET_RND_GEN_STATE",
                                               false, &reset_rnd_gen_state_));
  }

  bool HasInputC() const { return model_types_.HasInputC(); }
  RnnMode rnn_mode() const { return model_types_.rnn_mode; }
  TFRNNInputMode rnn_input_mode() const { return model_types_.rnn_input_mode; }
  RnnDirectionMode rnn_direction_mode() const {
    return model_types_.rnn_direction_mode;
  }
  CudnnModelTypes model_types() const { return model_types_; }
  float dropout() const { return dropout_; }
  uint64 seed() { return (static_cast<uint64>(seed_) << 32) | seed2_; }
  bool ResetRndGenState() { return reset_rnd_gen_state_; }

  template <typename T>
  Status ExtractCudnnRNNParamsInfo(OpKernelContext* context,
                                   std::unique_ptr<RnnDescriptor>* rnn_desc) {
    const Tensor* num_layers_t = nullptr;
    TF_RETURN_IF_ERROR(context->input("num_layers", &num_layers_t));
    if (!TensorShapeUtils::IsScalar(num_layers_t->shape())) {
      return errors::InvalidArgument("num_layers is not a scalar");
    }
    int num_layers = num_layers_t->scalar<int>()();
    const Tensor* num_units_t = nullptr;
    TF_RETURN_IF_ERROR(context->input("num_units", &num_units_t));
    if (!TensorShapeUtils::IsScalar(num_units_t->shape())) {
      return errors::InvalidArgument("num_units is not a scalar");
    }
    int num_units = num_units_t->scalar<int>()();
    const Tensor* input_size_t = nullptr;
    TF_RETURN_IF_ERROR(context->input("input_size", &input_size_t));
    if (!TensorShapeUtils::IsScalar(input_size_t->shape())) {
      return errors::InvalidArgument("input_size is not a scalar");
    }
    int input_size = input_size_t->scalar<int>()();

    RnnInputMode input_mode;
    TF_RETURN_IF_ERROR(
        ToRNNInputMode(rnn_input_mode(), num_units, input_size, &input_mode));

    auto* stream = context->op_device_context()->stream();
    // ExtracCudnnRNNParamsInfo is only called by op_kernels that do not require
    // random number generator, therefore set state_allocator to nullptr.
    auto rnn_desc_s = stream->parent()->createRnnDescriptor(
        num_layers, num_units, input_size, input_mode, rnn_direction_mode(),
        rnn_mode(), ToDataType<T>::value, dropout(), seed(),
        nullptr /* state_allocator */);
    if (!rnn_desc_s.ok()) {
      return FromExecutorStatus(rnn_desc_s);
    }
    *rnn_desc = rnn_desc_s.ConsumeValueOrDie();
    return Status::OK();
  }

 private:
  int seed_;
  int seed2_;
  float dropout_;
  bool reset_rnd_gen_state_;

  CudnnModelTypes model_types_;
};

// A class that returns the size of the opaque parameter buffer. The user should
// use that to create the actual parameter buffer for training. However, it
// should not be used for saving and restoring.
template <typename T, typename Index>
class CudnnRNNParamsSizeOp<GPUDevice, T, Index> : public CudnnRNNKernelCommon {
 public:
  typedef GPUDevice Device;
  explicit CudnnRNNParamsSizeOp(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    std::unique_ptr<RnnDescriptor> rnn_desc;
    OP_REQUIRES_OK(context, ExtractCudnnRNNParamsInfo<T>(context, &rnn_desc));
    int64 params_size_in_bytes = rnn_desc->ParamsSizeInBytes();
    CHECK(params_size_in_bytes % sizeof(T) == 0)
        << "params_size_in_bytes must be multiple of element size";
    int64 params_size = params_size_in_bytes / sizeof(T);

    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_t));
    *output_t->template flat<Index>().data() = params_size;
  }
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNParamsSize")       \
                              .Device(DEVICE_GPU)          \
                              .HostMemory("num_layers")    \
                              .HostMemory("num_units")     \
                              .HostMemory("input_size")    \
                              .HostMemory("params_size")   \
                              .TypeConstraint<T>("T")      \
                              .TypeConstraint<int32>("S"), \
                          CudnnRNNParamsSizeOp<GPUDevice, T, int32>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

// Convert weight and bias params from a platform-specific layout to the
// canonical form.
template <typename T>
class CudnnRNNParamsToCanonical<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  typedef GPUDevice Device;
  explicit CudnnRNNParamsToCanonical(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_params", &num_params_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(3);
    auto input_ptr = StreamExecutorUtil::AsDeviceMemory<T>(input);
    auto* stream = context->op_device_context()->stream();

    std::unique_ptr<RnnDescriptor> rnn_desc;
    OP_REQUIRES_OK(context, ExtractCudnnRNNParamsInfo<T>(context, &rnn_desc));
    int64 params_size_in_bytes = rnn_desc->ParamsSizeInBytes();
    CHECK(params_size_in_bytes % sizeof(T) == 0)
        << "params_size_in_bytes must be multiple of element size";

    const Tensor* num_units_t = nullptr;
    OP_REQUIRES_OK(context, context->input("num_units", &num_units_t));
    CHECK(TensorShapeUtils::IsScalar(num_units_t->shape()))
        << "num_units is not a scalar";
    int num_units = num_units_t->scalar<int>()();

    const Tensor* input_size_t = nullptr;
    OP_REQUIRES_OK(context, context->input("input_size", &input_size_t));
    CHECK(TensorShapeUtils::IsScalar(input_size_t->shape()))
        << "input_size is not a scalar";
    int input_size = input_size_t->scalar<int>()();

    const Tensor* num_layers_t = nullptr;
    OP_REQUIRES_OK(context, context->input("num_layers", &num_layers_t));
    CHECK(TensorShapeUtils::IsScalar(num_layers_t->shape()))
        << "num_layers is not a scalar";
    int num_layers = num_layers_t->scalar<int>()();
    int num_dirs = 1;
    if (rnn_direction_mode() == RnnDirectionMode::kRnnBidirectional) {
      num_dirs = 2;
    }
    const int num_params_per_layer = num_params_ / num_layers / num_dirs;
    // Number of params applied on inputs. The rest are applied on recurrent
    // hiddden states.
    const int num_params_input_state = num_params_per_layer / 2;
    CHECK(num_params_ % (num_layers * num_dirs) == 0)
        << "Number of params is not a multiple of num_layers * num_dirs.";
    CHECK(num_params_per_layer % 2 == 0)
        << "Number of params per layer is not a even number.";

    CHECK(num_params_ == rnn_desc->ParamsWeightRegions().size())
        << "Number of params mismatch. Expected " << num_params_ << ", got "
        << rnn_desc->ParamsWeightRegions().size();
    for (int i = 0; i < rnn_desc->ParamsWeightRegions().size(); i++) {
      int64 size_in_bytes = rnn_desc->ParamsWeightRegions()[i].size;
      int64 size = size_in_bytes / sizeof(T);
      const int layer_idx = i / num_params_per_layer;
      const int index_within_layer = i % num_params_per_layer;
      int width = 0, height = num_units;
      // In CuDNN layout, each layer has num_params_per_layer params, with the
      // first half a.k.a num_params_input_state params applied on the inputs,
      // and the second half on the recurrent hidden states.
      bool apply_on_input_state = index_within_layer < num_params_input_state;
      if (rnn_direction_mode() == RnnDirectionMode::kRnnUnidirectional) {
        if (layer_idx == 0 && apply_on_input_state) {
          width = input_size;
        } else {
          width = num_units;
        }
      } else {
        if (apply_on_input_state) {
          if (layer_idx <= 1) {
            // First fwd or bak layer.
            width = input_size;
          } else {
            // Following layers, cell inputs are concatenated outputs of
            // its prior layer.
            width = 2 * num_units;
          }
        } else {
          width = num_units;
        }
      }
      CHECK(size == width * height) << "Params size mismatch. Expected "
                                    << width * height << ", got " << size;
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  i, TensorShape({height, width}), &output));
      DeviceMemoryBase data_src_ptr = SliceDeviceMemory(
          input_ptr, rnn_desc->ParamsWeightRegions()[i].offset, size_in_bytes);
      auto data_dst_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*output);
      stream->ThenMemcpy(&data_dst_ptr, data_src_ptr, size_in_bytes);
    }

    OP_REQUIRES(context, num_params_ == rnn_desc->ParamsBiasRegions().size(),
                errors::InvalidArgument("Number of params mismatch. Expected ",
                                        num_params_, ", got ",
                                        rnn_desc->ParamsBiasRegions().size()));
    for (int i = 0; i < rnn_desc->ParamsBiasRegions().size(); i++) {
      int64 size_in_bytes = rnn_desc->ParamsBiasRegions()[i].size;
      int64 size = size_in_bytes / sizeof(T);
      OP_REQUIRES(context, size == num_units,
                  errors::InvalidArgument("Params size mismatch. Expected ",
                                          num_units, ", got ", size));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(num_params_ + i,
                                              TensorShape({size}), &output));
      DeviceMemoryBase data_src_ptr = SliceDeviceMemory(
          input_ptr, rnn_desc->ParamsBiasRegions()[i].offset, size_in_bytes);
      auto data_dst_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*output);
      stream->ThenMemcpy(&data_dst_ptr, data_src_ptr, size_in_bytes);
    }
  }

 private:
  int num_params_;
};

#define REGISTER_GPU(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNParamsToCanonical") \
                              .Device(DEVICE_GPU)           \
                              .HostMemory("num_layers")     \
                              .HostMemory("num_units")      \
                              .HostMemory("input_size")     \
                              .TypeConstraint<T>("T"),      \
                          CudnnRNNParamsToCanonical<GPUDevice, T>);
TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

// Convert weight and bias params from the canonical form to a
// platform-specific layout.
template <typename T>
class CudnnRNNCanonicalToParams<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  typedef GPUDevice Device;
  explicit CudnnRNNCanonicalToParams(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    std::unique_ptr<RnnDescriptor> rnn_desc;
    OP_REQUIRES_OK(context, ExtractCudnnRNNParamsInfo<T>(context, &rnn_desc));
    int64 params_size_in_bytes = rnn_desc->ParamsSizeInBytes();
    CHECK(params_size_in_bytes % sizeof(T) == 0)
        << "params_size_in_bytes must be multiple of element size";
    Tensor* output = nullptr;
    int params_size = params_size_in_bytes / sizeof(T);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {params_size}, &output));
    auto output_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*output);
    auto* stream = context->op_device_context()->stream();

    OpInputList weights;
    OP_REQUIRES_OK(context, context->input_list("weights", &weights));
    RestoreParams<T>(weights, rnn_desc->ParamsWeightRegions(), &output_ptr,
                     stream);

    OpInputList biases;
    OP_REQUIRES_OK(context, context->input_list("biases", &biases));
    RestoreParams<T>(biases, rnn_desc->ParamsBiasRegions(), &output_ptr,
                     stream);
  }
};

#define REGISTER_GPU(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNCanonicalToParams") \
                              .Device(DEVICE_GPU)           \
                              .HostMemory("num_layers")     \
                              .HostMemory("num_units")      \
                              .HostMemory("input_size")     \
                              .TypeConstraint<T>("T"),      \
                          CudnnRNNCanonicalToParams<GPUDevice, T>);
TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

// Run the forward operation of the RNN model.
template <typename T>
class CudnnRNNForwardOp<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  typedef GPUDevice Device;
  explicit CudnnRNNForwardOp(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    CudnnModelShapes model_shapes;
    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, model_types(), &input, &input_h,
                                       &input_c, &params, &model_shapes));
    const auto& input_shape = model_shapes.input_shape;
    const auto& hidden_state_shape = model_shapes.hidden_state_shape;
    const auto& output_shape = model_shapes.output_shape;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    Tensor* output_h = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, hidden_state_shape, &output_h));
    Tensor* output_c = nullptr;
    if (HasInputC()) {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      OP_REQUIRES_OK(
          context, context->allocate_output(2, hidden_state_shape, &output_c));
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_c));
    }

    auto* stream = context->op_device_context()->stream();
    auto* executor = stream->parent();
    RnnInputMode input_mode;
    OP_REQUIRES_OK(context,
                   ToRNNInputMode(rnn_input_mode(), model_shapes.num_units,
                                  model_shapes.input_size, &input_mode));
    // TODO(zhengxq): cache the descriptor so we don't have to create them all
    // the time.
    auto data_type = ToDataType<T>::value;
    {
      mutex_lock l(mu_);
      if (model_shapes_ == nullptr) {
        model_shapes_.reset(new CudnnModelShapes(model_shapes));
      } else {
        OP_REQUIRES(context, model_shapes_->IsCompatibleWith(model_shapes),
                    errors::InvalidArgument(
                        "Incompatible rnn model shapes inferred: expecting ",
                        model_shapes_->RnnDescDebugString(), ", getting ",
                        model_shapes.RnnDescDebugString(), "."));
      }
      if (rnn_desc_ == nullptr || ResetRndGenState()) {
        dropout_state_allocator_.reset(
            new CudnnRNNPersistentSpaceAllocator(context));
        auto rnn_desc_s = executor->createRnnDescriptor(
            model_shapes_->num_layers, model_shapes_->num_units,
            model_shapes_->input_size, input_mode, rnn_direction_mode(),
            rnn_mode(), data_type, dropout(), seed(),
            dropout_state_allocator_.get());
        OP_REQUIRES_OK(context, FromExecutorStatus(rnn_desc_s));
        rnn_desc_ = std::move(rnn_desc_s.ConsumeValueOrDie());
      }
    }

    auto input_desc_s = executor->createRnnSequenceTensorDescriptor(
        input_shape.dim_size(0), input_shape.dim_size(1),
        input_shape.dim_size(2), data_type);
    OP_REQUIRES_OK(context, FromExecutorStatus(input_desc_s));
    auto input_desc = input_desc_s.ConsumeValueOrDie();

    auto hidden_state_desc_s = executor->createRnnStateTensorDescriptor(
        hidden_state_shape.dim_size(0), hidden_state_shape.dim_size(1),
        hidden_state_shape.dim_size(2), data_type);
    OP_REQUIRES_OK(context, FromExecutorStatus(hidden_state_desc_s));
    auto hidden_state_desc = hidden_state_desc_s.ConsumeValueOrDie();

    auto output_desc_s = executor->createRnnSequenceTensorDescriptor(
        output_shape.dim_size(0), output_shape.dim_size(1),
        output_shape.dim_size(2), data_type);
    OP_REQUIRES_OK(context, FromExecutorStatus(output_desc_s));
    auto output_desc = output_desc_s.ConsumeValueOrDie();

    auto input_data = AsDeviceMemory<T>(input);
    auto input_h_data = AsDeviceMemory<T>(input_h);
    DeviceMemory<T> input_c_data;
    if (HasInputC()) {
      input_c_data = AsDeviceMemory<T>(input_c);
    }
    auto params_data = AsDeviceMemory<T>(params);
    auto output_data = AsDeviceMemory<T>(output);
    auto output_h_data = AsDeviceMemory<T>(output_h);
    DeviceMemory<T> output_c_data;
    if (HasInputC()) {
      output_c_data = AsDeviceMemory<T>(output_c);
    }

    // Creates a memory callback for the reserve_space. The memory lives in the
    // output of this kernel. And it will be fed into the backward pass when
    // needed.
    CudnnRNNReserveSpaceAllocator<T> reserve_space_allocator(context, 3);
    if (!is_training_) {
      Tensor* dummy_reserve_space = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(3, {}, &dummy_reserve_space));
    }
    // Creates a memory callback for the workspace. The memory lives to the end
    // of this kernel calls.
    CudnnRNNWorkspaceAllocator workspace_allocator(context);
    bool launch_status = false;
    {
      mutex_lock l(mu_);
      launch_status =
          stream
              ->ThenRnnForward(
                  *rnn_desc_, *input_desc, input_data, *hidden_state_desc,
                  input_h_data, *hidden_state_desc, input_c_data, params_data,
                  *output_desc, &output_data, *hidden_state_desc,
                  &output_h_data, *hidden_state_desc, &output_c_data,
                  is_training_, &reserve_space_allocator, &workspace_allocator)
              .ok();
    }
    OP_REQUIRES(context, launch_status,
                errors::Internal("Failed to call ThenRnnForward"));
  }

 private:
  mutex mu_;
  bool is_training_;
  std::unique_ptr<CudnnModelShapes> model_shapes_ GUARDED_BY(mu_);
  std::unique_ptr<RnnDescriptor> rnn_desc_ GUARDED_BY(mu_);
  std::unique_ptr<CudnnRNNPersistentSpaceAllocator> dropout_state_allocator_
      GUARDED_BY(mu_);
};

#define REGISTER_GPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("CudnnRNN").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CudnnRNNForwardOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

// Run the backward operation of the RNN model.
template <typename T>
class CudnnRNNBackwardOp<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  typedef GPUDevice Device;

  explicit CudnnRNNBackwardOp(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    CudnnModelShapes model_shapes;
    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, model_types(), &input, &input_h,
                                       &input_c, &params, &model_shapes));

    const auto& input_shape = model_shapes.input_shape;
    const auto& hidden_state_shape = model_shapes.hidden_state_shape;
    const auto& output_shape = model_shapes.output_shape;

    auto data_type = ToDataType<T>::value;
    const Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->input("output", &output));
    OP_REQUIRES(context, output_shape == output->shape(),
                errors::InvalidArgument(
                    "input_h and input_c must have the same shape: ",
                    input_h->shape().DebugString(), " ",
                    input_c->shape().DebugString()));
    const Tensor* output_h = nullptr;
    OP_REQUIRES_OK(context, context->input("output_h", &output_h));
    OP_REQUIRES(context, output_h->shape() == hidden_state_shape,
                errors::InvalidArgument(
                    "Invalid output_h shape: ", output_h->shape().DebugString(),
                    " ", hidden_state_shape.DebugString()));
    const Tensor* output_c = nullptr;
    if (HasInputC()) {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      OP_REQUIRES_OK(context, context->input("output_c", &output_c));
      OP_REQUIRES(context, output_c->shape() == hidden_state_shape,
                  errors::InvalidArgument("Invalid output_c shape: ",
                                          output_c->shape().DebugString(), " ",
                                          hidden_state_shape.DebugString()));
    }

    const Tensor* output_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->input("output_backprop", &output_backprop));
    OP_REQUIRES(context, output_backprop->shape() == output_shape,
                errors::InvalidArgument("Invalid output_backprop shapes: ",
                                        output_backprop->shape().DebugString(),
                                        " ", output_shape.DebugString()));

    const Tensor* output_h_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->input("output_h_backprop", &output_h_backprop));
    OP_REQUIRES(
        context, output_h_backprop->shape() == hidden_state_shape,
        errors::InvalidArgument("Invalid output_h_backprop shapes: ",
                                output_h_backprop->shape().DebugString(), " ",
                                hidden_state_shape.DebugString()));
    const Tensor* output_c_backprop = nullptr;
    if (HasInputC()) {
      OP_REQUIRES_OK(context,
                     context->input("output_c_backprop", &output_c_backprop));
      OP_REQUIRES(
          context, output_c_backprop->shape() == hidden_state_shape,
          errors::InvalidArgument("Invalid output_c_backprop shapes: ",
                                  output_c_backprop->shape().DebugString(), " ",
                                  hidden_state_shape.DebugString()));
    }
    const Tensor* reserve_space_const = nullptr;
    // This is the same "reserve_space" created by the forward op.
    // It can also be modified by this backward operation.
    OP_REQUIRES_OK(context,
                   context->input("reserve_space", &reserve_space_const));
    // Cudnn needs the reserve space to be writeable. This is fine because they
    // are opaque.
    Tensor* reserve_space = const_cast<Tensor*>(reserve_space_const);

    Tensor* input_backprop = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input->shape(), &input_backprop));
    Tensor* input_h_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_h->shape(),
                                                     &input_h_backprop));
    Tensor* input_c_backprop = nullptr;
    if (HasInputC()) {
      OP_REQUIRES_OK(context, context->allocate_output(2, input_c->shape(),
                                                       &input_c_backprop));
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(2, {}, &input_c_backprop));
    }
    Tensor* params_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, params->shape(),
                                                     &params_backprop));

    auto* stream = context->op_device_context()->stream();
    auto* executor = stream->parent();
    RnnInputMode input_mode;
    OP_REQUIRES_OK(context,
                   ToRNNInputMode(rnn_input_mode(), model_shapes.num_units,
                                  model_shapes.input_size, &input_mode));
    // TODO(zhengxq): cache the descriptor so we don't have to create them all
    // the time.
    {
      mutex_lock l(mu_);
      if (model_shapes_ == nullptr) {
        model_shapes_.reset(new CudnnModelShapes(model_shapes));
      } else {
        OP_REQUIRES(context, model_shapes_->IsCompatibleWith(model_shapes),
                    errors::InvalidArgument(
                        "Incompatible rnn model shapes inferred: expecting ",
                        model_shapes_->RnnDescDebugString(), ", getting ",
                        model_shapes.RnnDescDebugString(), "."));
      }

      if (rnn_desc_ == nullptr || ResetRndGenState()) {
        dropout_state_allocator_.reset(
            new CudnnRNNPersistentSpaceAllocator(context));
        auto rnn_desc_s = executor->createRnnDescriptor(
            model_shapes.num_layers, model_shapes.num_units,
            model_shapes.input_size, input_mode, rnn_direction_mode(),
            rnn_mode(), data_type, dropout(), seed(),
            dropout_state_allocator_.get());
        OP_REQUIRES_OK(context, FromExecutorStatus(rnn_desc_s));
        rnn_desc_ = std::move(rnn_desc_s.ConsumeValueOrDie());
      }
    }

    auto input_desc_s = executor->createRnnSequenceTensorDescriptor(
        input_shape.dim_size(0), input_shape.dim_size(1),
        input_shape.dim_size(2), data_type);
    OP_REQUIRES_OK(context, FromExecutorStatus(input_desc_s));
    auto input_desc = input_desc_s.ConsumeValueOrDie();

    auto hidden_state_desc_s = executor->createRnnStateTensorDescriptor(
        hidden_state_shape.dim_size(0), hidden_state_shape.dim_size(1),
        hidden_state_shape.dim_size(2), data_type);
    OP_REQUIRES_OK(context, FromExecutorStatus(hidden_state_desc_s));
    auto hidden_state_desc = hidden_state_desc_s.ConsumeValueOrDie();

    auto output_desc_s = executor->createRnnSequenceTensorDescriptor(
        output_shape.dim_size(0), output_shape.dim_size(1),
        output_shape.dim_size(2), data_type);
    OP_REQUIRES_OK(context, FromExecutorStatus(output_desc_s));
    auto output_desc = output_desc_s.ConsumeValueOrDie();

    auto input_data = AsDeviceMemory<T>(input);
    auto input_h_data = AsDeviceMemory<T>(input_h);
    DeviceMemory<T> input_c_data;
    if (HasInputC()) {
      input_c_data = AsDeviceMemory<T>(input_c);
    }
    auto params_data = AsDeviceMemory<T>(params);
    auto output_data = AsDeviceMemory<T>(output);
    auto output_h_data = AsDeviceMemory<T>(output_h);
    DeviceMemory<T> output_c_data;
    if (HasInputC()) {
      output_c_data = AsDeviceMemory<T>(output_c);
    }
    auto output_backprop_data = AsDeviceMemory<T>(output_backprop);
    auto output_h_backprop_data = AsDeviceMemory<T>(output_h_backprop);
    DeviceMemory<T> output_c_backprop_data;
    if (HasInputC()) {
      output_c_backprop_data = AsDeviceMemory<T>(output_c_backprop);
    }
    auto input_backprop_data = AsDeviceMemory<T>(input_backprop);
    auto input_h_backprop_data = AsDeviceMemory<T>(input_h_backprop);
    DeviceMemory<T> input_c_backprop_data;
    if (HasInputC()) {
      input_c_backprop_data = AsDeviceMemory<T>(input_c_backprop);
    }
    auto params_backprop_data = AsDeviceMemory<T>(params_backprop);
    auto reserve_space_uint8 = CastDeviceMemory<uint8, T>(reserve_space);
    // Creates a memory callback for the workspace. The memory lives to the end
    // of this kernel calls.
    CudnnRNNWorkspaceAllocator workspace_allocator(context);
    bool launch_status = false;
    {
      mutex_lock l(mu_);
      launch_status =
          stream
              ->ThenRnnBackward(
                  *rnn_desc_, *input_desc, input_data, *hidden_state_desc,
                  input_h_data, *hidden_state_desc, input_c_data, params_data,
                  *output_desc, output_data, *hidden_state_desc, output_h_data,
                  *hidden_state_desc, output_c_data, output_backprop_data,
                  output_h_backprop_data, output_c_backprop_data,
                  &input_backprop_data, &input_h_backprop_data,
                  &input_c_backprop_data, &params_backprop_data,
                  &reserve_space_uint8, &workspace_allocator)
              .ok();
    }
    OP_REQUIRES(context, launch_status,
                errors::Internal("Failed to call ThenRnnBackward"));
  }

 private:
  mutex mu_;
  std::unique_ptr<CudnnModelShapes> model_shapes_ GUARDED_BY(mu_);
  std::unique_ptr<RnnDescriptor> rnn_desc_ GUARDED_BY(mu_);
  std::unique_ptr<CudnnRNNPersistentSpaceAllocator> dropout_state_allocator_
      GUARDED_BY(mu_);
};

#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CudnnRNNBackprop").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CudnnRNNBackwardOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

// TODO(zhengxq): Add the conversion of Cudnn RNN Params from and to
// its canonical form.

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
