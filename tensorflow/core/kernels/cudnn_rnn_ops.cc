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
#include <utility>

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
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using GPUDevice = Eigen::GpuDevice;
using se::Stream;
using se::StreamExecutor;
using se::dnn::RnnDescriptor;

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

template <typename Device, typename T>
class CudnnRNNForwardOpV2;

template <typename Device, typename T>
class CudnnRNNBackwardOpV2;

template <typename Device, typename T>
class CudnnRNNForwardOpV3;

template <typename Device, typename T>
class CudnnRNNBackwardOpV3;

enum class TFRNNInputMode {
  kRNNLinearInput = 0,
  kRNNSkipInput = 1,
  kAutoSelect = 9999999
};

namespace {
using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::ScratchAllocator;
using se::dnn::AlgorithmConfig;
using se::dnn::AlgorithmDesc;
using se::dnn::ProfileResult;
using se::dnn::RnnDirectionMode;
using se::dnn::RnnInputMode;
using se::dnn::RnnMode;
using se::dnn::RnnSequenceTensorDescriptor;
using se::dnn::RnnStateTensorDescriptor;
using se::dnn::ToDataType;
using tsl::StatusOr;

uint64 HashList(const std::vector<int>& list) {
  if (list.empty()) {
    return 0;
  }
  uint64 hash_code = list[0];
  for (int i = 1; i < list.size(); i++) {
    hash_code = Hash64Combine(hash_code, list[i]);
  }
  return hash_code;
}

// Encapsulate all the shape information that is used in both forward and
// backward rnn operations.
class CudnnRnnParameters {
 public:
  CudnnRnnParameters(int num_layers, int input_size, int num_units,
                     int max_seq_length, int batch_size, int dir_count,
                     bool has_dropout, bool is_training, RnnMode rnn_mode,
                     TFRNNInputMode rnn_input_mode, DataType dtype)
      : num_layers_(num_layers),
        input_size_(input_size),
        num_units_(num_units),
        seq_length_(max_seq_length),
        batch_size_(batch_size),
        dir_count_(dir_count),
        has_dropout_(has_dropout),
        is_training_(is_training),
        rnn_mode_(rnn_mode),
        rnn_input_mode_(rnn_input_mode),
        dtype_(dtype) {
    hash_code_ =
        HashList({num_layers, input_size, num_units, max_seq_length, batch_size,
                  dir_count, static_cast<int>(has_dropout),
                  static_cast<int>(is_training), static_cast<int>(rnn_mode),
                  static_cast<int>(rnn_input_mode), dtype});
  }

  bool operator==(const CudnnRnnParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const CudnnRnnParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    std::vector<string> fields = {
        std::to_string(num_layers_),
        std::to_string(input_size_),
        std::to_string(num_units_),
        std::to_string(seq_length_),
        std::to_string(batch_size_),
        std::to_string(dir_count_),
        std::to_string(has_dropout_),
        std::to_string(is_training_),
        std::to_string(static_cast<int>(rnn_mode_)),
        std::to_string(static_cast<int>(rnn_input_mode_)),
        std::to_string(static_cast<int>(dtype_))};
    return absl::StrJoin(fields, ", ");
  }

 private:
  using ParameterDataType = std::tuple<int, int, int, int, int, int, bool, bool,
                                       RnnMode, TFRNNInputMode, DataType>;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(num_layers_, input_size_, num_units_, seq_length_,
                           batch_size_, dir_count_, has_dropout_, is_training_,
                           rnn_mode_, rnn_input_mode_, dtype_);
  }

  const int num_layers_;
  const int input_size_;
  const int num_units_;
  const int seq_length_;
  const int batch_size_;
  const int dir_count_;
  const bool has_dropout_;
  const bool is_training_;
  const RnnMode rnn_mode_;
  const TFRNNInputMode rnn_input_mode_;
  const DataType dtype_;
  uint64 hash_code_;
};

struct RnnAutotuneGroup {
  static string name() { return "Rnn"; }
};

using AutotuneRnnConfigMap =
    AutotuneSingleton<RnnAutotuneGroup, CudnnRnnParameters, AlgorithmConfig>;

Status ParseRNNMode(const string& str, RnnMode* rnn_mode) {
  if (str == "rnn_relu") {
    *rnn_mode = RnnMode::kRnnRelu;
    return OkStatus();
  } else if (str == "rnn_tanh") {
    *rnn_mode = RnnMode::kRnnTanh;
    return OkStatus();
  } else if (str == "lstm") {
    *rnn_mode = RnnMode::kRnnLstm;
    return OkStatus();
  } else if (str == "gru") {
    *rnn_mode = RnnMode::kRnnGru;
    return OkStatus();
  }
  return errors::InvalidArgument("Invalid RNN mode: ", str);
}

Status ParseTFRNNInputMode(const string& str, TFRNNInputMode* rnn_input_mode) {
  if (str == "linear_input") {
    *rnn_input_mode = TFRNNInputMode::kRNNLinearInput;
    return OkStatus();
  } else if (str == "skip_input") {
    *rnn_input_mode = TFRNNInputMode::kRNNSkipInput;
    return OkStatus();
  } else if (str == "auto_select") {
    *rnn_input_mode = TFRNNInputMode::kAutoSelect;
    return OkStatus();
  }
  return errors::InvalidArgument("Invalid RNN input mode: ", str);
}

Status ParseRNNDirectionMode(const string& str,
                             RnnDirectionMode* rnn_dir_mode) {
  if (str == "unidirectional") {
    *rnn_dir_mode = RnnDirectionMode::kRnnUnidirectional;
    return OkStatus();
  } else if (str == "bidirectional") {
    *rnn_dir_mode = RnnDirectionMode::kRnnBidirectional;
    return OkStatus();
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
  return OkStatus();
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
                                   int64_t offset, int64_t size) {
  const void* base_ptr = device_memory.opaque();
  void* offset_ptr =
      const_cast<char*>(reinterpret_cast<const char*>(base_ptr) + offset);
  CHECK(offset + size <= device_memory.size())
      << "The slice is not within the region of DeviceMemory.";
  return DeviceMemoryBase(offset_ptr, size);
}

inline Status FromExecutorStatus(const tsl::Status& s) {
  return s.ok() ? OkStatus() : Status(s.code(), s.error_message());
}

template <typename T>
inline Status FromExecutorStatus(const tsl::StatusOr<T>& s) {
  return FromExecutorStatus(s.status());
}

inline tsl::Status ToExecutorStatus(const Status& s) {
  return s.ok() ? OkStatus() : Status(s.code(), s.error_message());
}

template <typename>
struct ToTFDataType;

template <>
struct ToTFDataType<Eigen::half> : std::integral_constant<DataType, DT_HALF> {};

template <>
struct ToTFDataType<Eigen::bfloat16>
    : std::integral_constant<DataType, DT_BFLOAT16> {};

template <>
struct ToTFDataType<float> : std::integral_constant<DataType, DT_FLOAT> {};

template <>
struct ToTFDataType<double> : std::integral_constant<DataType, DT_DOUBLE> {};

template <>
struct ToTFDataType<uint8> : std::integral_constant<DataType, DT_UINT8> {};

// A helper to allocate temporary scratch memory for Cudnn RNN models. It
// takes the ownership of the underlying memory. The expectation is that the
// memory should be alive for the span of the Cudnn RNN itself.
template <typename T>
class CudnnRnnAllocatorInTemp : public ScratchAllocator {
 public:
  ~CudnnRnnAllocatorInTemp() override = default;

  explicit CudnnRnnAllocatorInTemp(OpKernelContext* context)
      : context_(context) {}
  int64_t GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64_t>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override {
    Tensor temporary_memory;
    const DataType tf_data_type = ToTFDataType<T>::value;
    int64_t allocate_count =
        Eigen::divup(byte_size, static_cast<int64_t>(sizeof(T)));
    Status allocation_status(context_->allocate_temp(
        tf_data_type, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory.template flat<T>().data(),
        temporary_memory.template flat<T>().size() * sizeof(T));
  }

  int64_t TotalByteSize() const { return total_byte_size_; }

  Tensor get_allocated_tensor(int index) const {
    return allocated_tensors_[index];
  }

 private:
  int64_t total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  std::vector<Tensor> allocated_tensors_;
};

// A helper to allocate memory for Cudnn RNN models as a kernel output. It is
// used by forward pass kernel to feed the output to the backward pass.
// The memory is expected to live long enough after the backward pass is
// finished.
template <typename T>
class CudnnRnnAllocatorInOutput : public ScratchAllocator {
 public:
  ~CudnnRnnAllocatorInOutput() override {}
  CudnnRnnAllocatorInOutput(OpKernelContext* context, int output_index)
      : context_(context), output_index_(output_index) {}
  int64_t GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64_t>::max();
  }
  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override {
    CHECK(total_byte_size_ == 0)
        << "Reserve space allocator can only be called once";
    int64_t allocate_count =
        Eigen::divup(byte_size, static_cast<int64_t>(sizeof(T)));

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
  int64_t TotalByteSize() { return total_byte_size_; }

 private:
  int64_t total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  int output_index_;
};

// A helper to allocate memory for Cudnn RNN models, which is
// expected to live between kernel invocations.
// This class is not thread-safe.
class CudnnRNNSpaceAllocator : public ScratchAllocator {
 public:
  explicit CudnnRNNSpaceAllocator(OpKernelContext* context)
      : context_(context) {}

  ~CudnnRNNSpaceAllocator() override {}

  int64_t GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64_t>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override {
    if (total_byte_size_ != 0) {
      return Status(absl::StatusCode::kFailedPrecondition,
                    "Space allocator can only be called once");
    }

    Status allocation_status =
        context_->allocate_temp(DT_UINT8, TensorShape({byte_size}), &tensor_);
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    total_byte_size_ += byte_size;
    return AsDeviceMemory<uint8>(&tensor_);
  }
  int64_t TotalByteSize() { return total_byte_size_; }

 private:
  int64_t total_byte_size_ = 0;
  Tensor tensor_;
  OpKernelContext* context_;  // not owned
};

struct CudnnModelTypes {
  RnnMode rnn_mode;
  TFRNNInputMode rnn_input_mode;
  RnnDirectionMode rnn_direction_mode;
  bool HasInputC() const {
    // For Cudnn 5.0, only LSTM has input-c. All other models use only
    // input-h.
    return rnn_mode == RnnMode::kRnnLstm;
  }

  string DebugString() const {
    return strings::Printf(
        "[rnn_mode, rnn_input_mode, rnn_direction_mode]: %d, %d, %d ",
        static_cast<int>(rnn_mode), static_cast<int>(rnn_input_mode),
        static_cast<int>(rnn_direction_mode));
  }
};

// A helper class that collects the shapes to describe a RNN model.
struct CudnnRnnModelShapes {
  int num_layers;
  int input_size;
  int num_units;
  int dir_count;
  int max_seq_length;
  int batch_size;
  int cell_num_units = 0;
  // If you add new field to this structure, please take care of
  // updating IsCompatibleWith() below as well as the hash function in
  // CudnnRnnConfigHasher.
  TensorShape input_shape;
  TensorShape output_shape;
  TensorShape hidden_state_shape;
  TensorShape cell_state_shape;
  // At present only fields related to cached RnnDescriptor are concerned.
  bool IsCompatibleWith(const CudnnRnnModelShapes& rhs) const {
    return num_layers == rhs.num_layers && input_size == rhs.input_size &&
           num_units == rhs.num_units && dir_count == rhs.dir_count &&
           cell_num_units == rhs.cell_num_units &&
           max_seq_length == rhs.max_seq_length;
  }
  string DebugString() const {
    return strings::Printf(
        "[num_layers, input_size, num_units, dir_count, max_seq_length, "
        "batch_size, cell_num_units]: [%d, %d, %d, %d, %d, %d, %d] ",
        num_layers, input_size, num_units, dir_count, max_seq_length,
        batch_size, cell_num_units);
  }
};

// Utility class for using CudnnRnnConfig and AlgorithmDesc pair a hash table
// key.
struct CudnnRnnConfigHasher {
  uint64 operator()(
      const std::pair<CudnnRnnModelShapes, absl::optional<AlgorithmDesc>>&
          to_hash) const {
    auto& shapes = to_hash.first;
    auto& algo_desc = to_hash.second;

    uint64 hash =
        HashList({shapes.num_layers, shapes.input_size, shapes.num_units,
                  shapes.dir_count, shapes.max_seq_length, shapes.batch_size});
    if (algo_desc.has_value()) {
      hash = Hash64Combine(hash, algo_desc->hash());
    }
    return hash;
  }
};

// Utility class for using CudnnRnnModelShapes and AlgorithmDesc pair as a hash
// table key.
struct CudnnRnnConfigComparator {
  bool operator()(
      const std::pair<CudnnRnnModelShapes, absl::optional<AlgorithmDesc>>& lhs,
      const std::pair<CudnnRnnModelShapes, absl::optional<AlgorithmDesc>>& rhs)
      const {
    return lhs.first.IsCompatibleWith(rhs.first) && lhs.second == rhs.second;
  }
};

// Pointers to RNN scratch space for a specific set of shape parameters (used as
// a hash table value in CudnnRNNForwardOp and CudnnRNNBackwardOp).
struct RnnScratchSpace {
  std::unique_ptr<RnnDescriptor> rnn_desc;
  std::unique_ptr<CudnnRNNSpaceAllocator> dropout_state_allocator;
};

// Extract and checks the forward input tensors, parameters, and shapes from the
// OpKernelContext.
Status ExtractForwardInput(OpKernelContext* context,
                           const CudnnModelTypes& model_types, bool time_major,
                           const Tensor** input, const Tensor** input_h,
                           const Tensor** input_c, const Tensor** params,
                           const int num_proj,
                           CudnnRnnModelShapes* model_shapes) {
  TF_RETURN_IF_ERROR(context->input("input", input));
  TF_RETURN_IF_ERROR(context->input("input_h", input_h));
  if (model_types.HasInputC()) {
    TF_RETURN_IF_ERROR(context->input("input_c", input_c));
  }
  TF_RETURN_IF_ERROR(context->input("params", params));

  if ((*input)->dims() != 3) {
    return errors::InvalidArgument("RNN input must be a 3-D vector.");
  }
  if (time_major) {
    model_shapes->max_seq_length = (*input)->dim_size(0);
    model_shapes->batch_size = (*input)->dim_size(1);
  } else {
    model_shapes->max_seq_length = (*input)->dim_size(1);
    model_shapes->batch_size = (*input)->dim_size(0);
  }
  model_shapes->input_size = (*input)->dim_size(2);
  model_shapes->input_shape = (*input)->shape();
  model_shapes->dir_count =
      (model_types.rnn_direction_mode == RnnDirectionMode::kRnnBidirectional)
          ? 2
          : 1;

  if ((*input_h)->dims() != 3) {
    return errors::InvalidArgument("RNN input_h must be a 3-D vector.");
  }
  if (time_major) {
    model_shapes->num_layers =
        (*input_h)->dim_size(0) / model_shapes->dir_count;
  } else {
    model_shapes->num_layers =
        (*input_h)->dim_size(1) / model_shapes->dir_count;
  }
  model_shapes->num_units = (*input_h)->dim_size(2);

  if (time_major) {
    model_shapes->hidden_state_shape =
        TensorShape({model_shapes->dir_count * model_shapes->num_layers,
                     model_shapes->batch_size, model_shapes->num_units});
  } else {
    model_shapes->hidden_state_shape =
        TensorShape({model_shapes->batch_size,
                     model_shapes->dir_count * model_shapes->num_layers,
                     model_shapes->num_units});
  }
  if ((*input_h)->shape() != model_shapes->hidden_state_shape) {
    return errors::InvalidArgument(
        "Invalid input_h shape: ", (*input_h)->shape().DebugString(), " ",
        model_shapes->hidden_state_shape.DebugString());
  }
  if (model_types.HasInputC()) {
    model_shapes->cell_num_units = (*input_c)->dim_size(2);
    if (time_major) {
      model_shapes->cell_state_shape =
          TensorShape({model_shapes->dir_count * model_shapes->num_layers,
                       model_shapes->batch_size, model_shapes->cell_num_units});
    } else {
      model_shapes->cell_state_shape =
          TensorShape({model_shapes->batch_size,
                       model_shapes->dir_count * model_shapes->num_layers,
                       model_shapes->cell_num_units});
    }
    if (num_proj == 0) {
      if ((*input_h)->shape() != (*input_c)->shape()) {
        return errors::InvalidArgument(
            "input_h and input_c must have the same shape w/o projection: ",
            (*input_h)->shape().DebugString(), " ",
            (*input_c)->shape().DebugString());
      }
    } else {
      if ((*input_h)->dim_size(2) > (*input_c)->dim_size(2) ||
          num_proj != (*input_h)->dim_size(2) ||
          (*input_h)->dim_size(0) != (*input_c)->dim_size(0) ||
          (*input_h)->dim_size(1) != (*input_c)->dim_size(1)) {
        return errors::InvalidArgument(
            "Invalid input_h and input_c w/ projection size: ", num_proj, " ",
            (*input_h)->shape().DebugString(), " ",
            (*input_c)->shape().DebugString());
      }
    }
  } else {
    // dummy cell_state_shape TODO(kaixih): remove the time_major branch
    if (time_major) {
      model_shapes->cell_state_shape =
          TensorShape({model_shapes->dir_count * model_shapes->num_layers,
                       model_shapes->batch_size, model_shapes->num_units});
    } else {
      model_shapes->cell_state_shape =
          TensorShape({model_shapes->batch_size,
                       model_shapes->dir_count * model_shapes->num_layers,
                       model_shapes->num_units});
    }
    model_shapes->cell_num_units = 0;
  }
  if (time_major) {
    model_shapes->output_shape =
        TensorShape({model_shapes->max_seq_length, model_shapes->batch_size,
                     model_shapes->dir_count * model_shapes->num_units});
  } else {
    model_shapes->output_shape =
        TensorShape({model_shapes->batch_size, model_shapes->max_seq_length,
                     model_shapes->dir_count * model_shapes->num_units});
  }
  return OkStatus();
}

// Overloaded function to process the sequence_lengths
Status ExtractForwardInput(OpKernelContext* context,
                           const CudnnModelTypes& model_types, bool time_major,
                           const Tensor** input, const Tensor** input_h,
                           const Tensor** input_c, const Tensor** params,
                           const Tensor** sequence_lengths, const int num_proj,
                           CudnnRnnModelShapes* model_shapes) {
  TF_RETURN_IF_ERROR(context->input("sequence_lengths", sequence_lengths));
  return ExtractForwardInput(context, model_types, time_major, input, input_h,
                             input_c, params, num_proj, model_shapes);
}

template <typename T>
Status CreateForwardAndBackwardIODescriptors(
    OpKernelContext* context, const CudnnRnnModelShapes& model_shapes,
    std::unique_ptr<RnnSequenceTensorDescriptor>* input_desc,
    std::unique_ptr<RnnStateTensorDescriptor>* h_state_desc,
    std::unique_ptr<RnnStateTensorDescriptor>* c_state_desc,
    std::unique_ptr<RnnSequenceTensorDescriptor>* output_desc,
    const absl::Span<const int> seq_lengths, bool time_major) {
  StreamExecutor* executor = context->op_device_context()->stream()->parent();
  se::dnn::DataType data_type = ToDataType<T>::value;

  const TensorShape& input_shape = model_shapes.input_shape;
  const TensorShape& hidden_state_shape = model_shapes.hidden_state_shape;
  const TensorShape& cell_state_shape = model_shapes.cell_state_shape;
  const TensorShape& output_shape = model_shapes.output_shape;

  DCHECK_EQ(input_shape.dims(), 3);
  if (seq_lengths.data() != nullptr) {
    if (time_major) {
      auto input_desc_s = executor->createRnnSequenceTensorDescriptor(
          input_shape.dim_size(0), input_shape.dim_size(1),
          input_shape.dim_size(2), seq_lengths, time_major, data_type);
      TF_RETURN_IF_ERROR(input_desc_s.status());
      *input_desc = std::move(input_desc_s).value();
    } else {
      auto input_desc_s = executor->createRnnSequenceTensorDescriptor(
          input_shape.dim_size(1), input_shape.dim_size(0),
          input_shape.dim_size(2), seq_lengths, time_major, data_type);
      TF_RETURN_IF_ERROR(input_desc_s.status());
      *input_desc = std::move(input_desc_s).value();
    }
  } else {
    auto input_desc_s = executor->createRnnSequenceTensorDescriptor(
        input_shape.dim_size(0), input_shape.dim_size(1),
        input_shape.dim_size(2), data_type);
    TF_RETURN_IF_ERROR(input_desc_s.status());
    *input_desc = std::move(input_desc_s).value();
  }

  DCHECK_EQ(hidden_state_shape.dims(), 3);
  if (time_major) {
    auto hidden_state_desc_s = executor->createRnnStateTensorDescriptor(
        hidden_state_shape.dim_size(0), hidden_state_shape.dim_size(1),
        hidden_state_shape.dim_size(2), data_type);
    TF_RETURN_IF_ERROR(hidden_state_desc_s.status());
    *h_state_desc = std::move(hidden_state_desc_s).value();
  } else {
    auto hidden_state_desc_s = executor->createRnnStateTensorDescriptor(
        hidden_state_shape.dim_size(1), hidden_state_shape.dim_size(0),
        hidden_state_shape.dim_size(2), data_type);
    TF_RETURN_IF_ERROR(hidden_state_desc_s.status());
    *h_state_desc = std::move(hidden_state_desc_s).value();
  }

  DCHECK_EQ(cell_state_shape.dims(), 3);
  if (time_major) {
    auto cell_state_desc_s = executor->createRnnStateTensorDescriptor(
        cell_state_shape.dim_size(0), cell_state_shape.dim_size(1),
        cell_state_shape.dim_size(2), data_type);
    TF_RETURN_IF_ERROR(cell_state_desc_s.status());
    *c_state_desc = std::move(cell_state_desc_s).value();
  } else {
    auto cell_state_desc_s = executor->createRnnStateTensorDescriptor(
        cell_state_shape.dim_size(1), cell_state_shape.dim_size(0),
        cell_state_shape.dim_size(2), data_type);
    TF_RETURN_IF_ERROR(cell_state_desc_s.status());
    *c_state_desc = std::move(cell_state_desc_s).value();
  }

  DCHECK_EQ(output_shape.dims(), 3);
  if (seq_lengths.data() != nullptr) {
    if (time_major) {
      auto output_desc_s = executor->createRnnSequenceTensorDescriptor(
          output_shape.dim_size(0), output_shape.dim_size(1),
          output_shape.dim_size(2), seq_lengths, time_major, data_type);
      TF_RETURN_IF_ERROR(output_desc_s.status());
      *output_desc = std::move(output_desc_s).value();
    } else {
      auto output_desc_s = executor->createRnnSequenceTensorDescriptor(
          output_shape.dim_size(1), output_shape.dim_size(0),
          output_shape.dim_size(2), seq_lengths, time_major, data_type);
      TF_RETURN_IF_ERROR(output_desc_s.status());
      *output_desc = std::move(output_desc_s).value();
    }
  } else {
    auto output_desc_s = executor->createRnnSequenceTensorDescriptor(
        output_shape.dim_size(0), output_shape.dim_size(1),
        output_shape.dim_size(2), data_type);
    TF_RETURN_IF_ERROR(output_desc_s.status());
    *output_desc = std::move(output_desc_s).value();
  }

  return OkStatus();
}

template <typename T>
Status DoForwardImpl(OpKernelContext* context, const RnnDescriptor& rnn_desc,
                     const CudnnModelTypes& model_types,
                     const CudnnRnnModelShapes& model_shapes,
                     /* forward inputs */
                     const Tensor* input, const Tensor* input_h,
                     const Tensor* input_c, const Tensor* params,
                     const bool is_training,
                     /* forward outputs, outputs of the function */
                     Tensor* output, Tensor* output_h, Tensor* output_c,
                     const Tensor* sequence_lengths, bool time_major,
                     ScratchAllocator* reserve_space_allocator,
                     ScratchAllocator* workspace_allocator,
                     ProfileResult* output_profile_result) {
  std::unique_ptr<RnnSequenceTensorDescriptor> input_desc;
  std::unique_ptr<RnnStateTensorDescriptor> h_state_desc;
  std::unique_ptr<RnnStateTensorDescriptor> c_state_desc;
  std::unique_ptr<RnnSequenceTensorDescriptor> output_desc;

  absl::Span<const int> seq_lengths;
  if (sequence_lengths != nullptr) {
    seq_lengths = absl::Span<const int>(
        sequence_lengths->template flat<int>().data(), model_shapes.batch_size);
  }
  TF_RETURN_IF_ERROR(CreateForwardAndBackwardIODescriptors<T>(
      context, model_shapes, &input_desc, &h_state_desc, &c_state_desc,
      &output_desc, seq_lengths, time_major));

  auto input_data = AsDeviceMemory<T>(input);
  auto input_h_data = AsDeviceMemory<T>(input_h);
  DeviceMemory<T> input_c_data;
  if (model_types.HasInputC()) {
    input_c_data = AsDeviceMemory<T>(input_c);
  }

  auto params_data = AsDeviceMemory<T>(params);
  auto output_data = AsDeviceMemory<T>(output);
  auto output_h_data = AsDeviceMemory<T>(output_h);
  DeviceMemory<T> output_c_data;
  if (model_types.HasInputC()) {
    output_c_data = AsDeviceMemory<T>(output_c);
  }

  Stream* stream = context->op_device_context()->stream();

  Tensor seq_lengths_tensor;
  DeviceMemory<int> seq_lengths_ptr;
  if (sequence_lengths != nullptr) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, {static_cast<long>(seq_lengths.size())},
        &seq_lengths_tensor));
    seq_lengths_ptr = AsDeviceMemory<int>(&seq_lengths_tensor);
    if (!stream
             ->ThenMemcpy(&seq_lengths_ptr, seq_lengths.data(),
                          seq_lengths.size() * sizeof(int))
             .ok()) {
      return errors::InvalidArgument(
          "Failed to copy memory from host to "
          "device for sequence_lengths in "
          "CudnnRNNV3");
    }
  }

  bool launch_success =
      stream
          ->ThenRnnForward(rnn_desc, *input_desc, input_data, seq_lengths_ptr,
                           *h_state_desc, input_h_data, *c_state_desc,
                           input_c_data, params_data, *output_desc,
                           &output_data, *h_state_desc, &output_h_data,
                           *c_state_desc, &output_c_data, is_training,
                           reserve_space_allocator, workspace_allocator,
                           output_profile_result)
          .ok();
  return launch_success
             ? OkStatus()
             : errors::Internal(
                   "Failed to call ThenRnnForward with model config: ",
                   model_types.DebugString(), ", ", model_shapes.DebugString());
}

template <typename T>
Status DoForward(OpKernelContext* context, const RnnDescriptor& rnn_desc,
                 const CudnnModelTypes& model_types,
                 const CudnnRnnModelShapes& model_shapes,
                 /* forward inputs */
                 const Tensor* input, const Tensor* input_h,
                 const Tensor* input_c, const Tensor* params,
                 const bool is_training,
                 /* forward outputs, outputs of the function */
                 Tensor* output, Tensor* output_h, Tensor* output_c,
                 const Tensor* sequence_lengths, bool time_major,
                 ScratchAllocator* reserve_space_allocator,
                 ScratchAllocator* workspace_allocator,
                 ProfileResult* output_profile_result) {
  return DoForwardImpl<T>(context, rnn_desc, model_types, model_shapes, input,
                          input_h, input_c, params, is_training, output,
                          output_h, output_c, sequence_lengths, time_major,
                          reserve_space_allocator, workspace_allocator,
                          output_profile_result);
}

template <>
Status DoForward<Eigen::bfloat16>(
    OpKernelContext* context, const RnnDescriptor& rnn_desc,
    const CudnnModelTypes& model_types, const CudnnRnnModelShapes& model_shapes,
    /* forward inputs */
    const Tensor* input, const Tensor* input_h, const Tensor* input_c,
    const Tensor* params, const bool is_training,
    /* forward outputs, outputs of the function */
    Tensor* output, Tensor* output_h, Tensor* output_c,
    const Tensor* sequence_lengths, bool time_major,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator,
    ProfileResult* output_profile_result) {
  // CudnnRnn doesn't support bfloat16 yet, so cast to float. This avoids errors
  // with mixed_bfloat16 policy.
  Tensor casted_input;
  Tensor casted_input_h;
  Tensor casted_input_c;
  Tensor casted_params;
  Tensor casted_output;
  Tensor casted_output_h;
  Tensor casted_output_c;
  const GPUDevice& device = context->eigen_device<GPUDevice>();
  functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
  auto allocate_and_cast_to_float = [&](const Tensor* tensor,
                                        Tensor* casted_tensor) {
    TF_RETURN_IF_ERROR(context, context->allocate_temp(
                                    DT_FLOAT, tensor->shape(), casted_tensor));
    cast(device, casted_tensor->template flat<float>(),
         tensor->template flat<Eigen::bfloat16>());
    return OkStatus();
  };
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(input, &casted_input));
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(input_h, &casted_input_h));
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(params, &casted_params));
  TF_RETURN_IF_ERROR(context, context->allocate_temp(DT_FLOAT, output->shape(),
                                                     &casted_output));
  TF_RETURN_IF_ERROR(
      context,
      context->allocate_temp(DT_FLOAT, output_h->shape(), &casted_output_h));
  if (model_types.HasInputC()) {
    TF_RETURN_IF_ERROR(allocate_and_cast_to_float(input_c, &casted_input_c));
    TF_RETURN_IF_ERROR(
        context,
        context->allocate_temp(DT_FLOAT, output_c->shape(), &casted_output_c));
  }

  TF_RETURN_IF_ERROR(DoForwardImpl<float>(
      context, rnn_desc, model_types, model_shapes, &casted_input,
      &casted_input_h, &casted_input_c, &casted_params, is_training,
      &casted_output, &casted_output_h, &casted_output_c, sequence_lengths,
      time_major, reserve_space_allocator, workspace_allocator,
      output_profile_result));

  functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
  cast_back(device, output->template flat<Eigen::bfloat16>(),
            const_cast<const Tensor*>(&casted_output)->template flat<float>());
  cast_back(
      device, output_h->template flat<Eigen::bfloat16>(),
      const_cast<const Tensor*>(&casted_output_h)->template flat<float>());
  if (model_types.HasInputC()) {
    cast_back(
        device, output_c->template flat<Eigen::bfloat16>(),
        const_cast<const Tensor*>(&casted_output_c)->template flat<float>());
  }
  return OkStatus();
}

template <typename T>
Status DoBackwardImpl(
    OpKernelContext* context, const RnnDescriptor& rnn_desc,
    const CudnnModelTypes& model_types, const CudnnRnnModelShapes& model_shapes,
    /* forward inputs */
    const Tensor* input, const Tensor* input_h, const Tensor* input_c,
    const Tensor* params,
    /* forward outputs */
    const Tensor* output, const Tensor* output_h, const Tensor* output_c,
    /* backprop inputs */
    const Tensor* output_backprop, const Tensor* output_h_backprop,
    const Tensor* output_c_backprop, const Tensor* reserve_space,
    /* backprop outputs, output of the function */
    Tensor* input_backprop, Tensor* input_h_backprop, Tensor* input_c_backprop,
    Tensor* params_backprop, const Tensor* sequence_lengths, bool time_major,
    ScratchAllocator* workspace_allocator,
    ProfileResult* output_profile_result) {
  std::unique_ptr<RnnSequenceTensorDescriptor> input_desc;
  std::unique_ptr<RnnStateTensorDescriptor> h_state_desc;
  std::unique_ptr<RnnStateTensorDescriptor> c_state_desc;
  std::unique_ptr<RnnSequenceTensorDescriptor> output_desc;

  absl::Span<const int> seq_lengths;
  if (sequence_lengths != nullptr) {
    seq_lengths = absl::Span<const int>(
        sequence_lengths->template flat<int>().data(), model_shapes.batch_size);
  }
  TF_RETURN_IF_ERROR(CreateForwardAndBackwardIODescriptors<T>(
      context, model_shapes, &input_desc, &h_state_desc, &c_state_desc,
      &output_desc, seq_lengths, time_major));

  auto input_data = AsDeviceMemory<T>(input);
  auto input_h_data = AsDeviceMemory<T>(input_h);
  DeviceMemory<T> input_c_data;
  if (model_types.HasInputC()) {
    input_c_data = AsDeviceMemory<T>(input_c);
  }
  auto params_data = AsDeviceMemory<T>(params);
  auto output_data = AsDeviceMemory<T>(output);
  auto output_h_data = AsDeviceMemory<T>(output_h);
  DeviceMemory<T> output_c_data;
  if (model_types.HasInputC()) {
    output_c_data = AsDeviceMemory<T>(output_c);
  }
  auto output_backprop_data = AsDeviceMemory<T>(output_backprop);
  auto output_h_backprop_data = AsDeviceMemory<T>(output_h_backprop);
  DeviceMemory<T> output_c_backprop_data;
  if (model_types.HasInputC()) {
    output_c_backprop_data = AsDeviceMemory<T>(output_c_backprop);
  }
  auto input_backprop_data = AsDeviceMemory<T>(input_backprop);
  auto input_h_backprop_data = AsDeviceMemory<T>(input_h_backprop);
  DeviceMemory<T> input_c_backprop_data;
  if (model_types.HasInputC()) {
    input_c_backprop_data = AsDeviceMemory<T>(input_c_backprop);
  }
  auto params_backprop_data = AsDeviceMemory<T>(params_backprop);
  auto reserve_space_uint8 =
      CastDeviceMemory<uint8, T>(const_cast<Tensor*>(reserve_space));

  // Creates a memory callback for the workspace. The memory lives to the end
  // of this kernel calls.
  Stream* stream = context->op_device_context()->stream();

  Tensor seq_lengths_tensor;
  DeviceMemory<int> seq_lengths_ptr;
  if (sequence_lengths != nullptr) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, {static_cast<long>(seq_lengths.size())},
        &seq_lengths_tensor));
    seq_lengths_ptr = AsDeviceMemory<int>(&seq_lengths_tensor);
    if (!stream
             ->ThenMemcpy(&seq_lengths_ptr, seq_lengths.data(),
                          seq_lengths.size() * sizeof(int))
             .ok()) {
      return errors::InvalidArgument(
          "Failed to copy memory from host to "
          "device for sequence_lengths in "
          "CudnnRNNBackwardOpV3");
    }
  }

  bool launch_success =
      stream
          ->ThenRnnBackward(
              rnn_desc, *input_desc, input_data, seq_lengths_ptr, *h_state_desc,
              input_h_data, *c_state_desc, input_c_data, params_data,
              *output_desc, output_data, *h_state_desc, output_h_data,
              *c_state_desc, output_c_data, output_backprop_data,
              output_h_backprop_data, output_c_backprop_data,
              &input_backprop_data, &input_h_backprop_data,
              &input_c_backprop_data, &params_backprop_data,
              &reserve_space_uint8, workspace_allocator, output_profile_result)
          .ok();
  return launch_success
             ? OkStatus()
             : errors::Internal(
                   "Failed to call ThenRnnBackward with model config: ",
                   model_types.DebugString(), ", ", model_shapes.DebugString());
}

template <typename T>
Status DoBackward(
    OpKernelContext* context, const RnnDescriptor& rnn_desc,
    const CudnnModelTypes& model_types, const CudnnRnnModelShapes& model_shapes,
    /* forward inputs */
    const Tensor* input, const Tensor* input_h, const Tensor* input_c,
    const Tensor* params,
    /* forward outputs */
    const Tensor* output, const Tensor* output_h, const Tensor* output_c,
    /* backprop inputs */
    const Tensor* output_backprop, const Tensor* output_h_backprop,
    const Tensor* output_c_backprop, const Tensor* reserve_space,
    /* backprop outputs, output of the function */
    Tensor* input_backprop, Tensor* input_h_backprop, Tensor* input_c_backprop,
    Tensor* params_backprop, const Tensor* sequence_lengths, bool time_major,
    ScratchAllocator* workspace_allocator,
    ProfileResult* output_profile_result) {
  return DoBackwardImpl<T>(
      context, rnn_desc, model_types, model_shapes, input, input_h, input_c,
      params, output, output_h, output_c, output_backprop, output_h_backprop,
      output_c_backprop, reserve_space, input_backprop, input_h_backprop,
      input_c_backprop, params_backprop, sequence_lengths, time_major,
      workspace_allocator, output_profile_result);
}

template <>
Status DoBackward<Eigen::bfloat16>(
    OpKernelContext* context, const RnnDescriptor& rnn_desc,
    const CudnnModelTypes& model_types, const CudnnRnnModelShapes& model_shapes,
    /* forward inputs */
    const Tensor* input, const Tensor* input_h, const Tensor* input_c,
    const Tensor* params,
    /* forward outputs */
    const Tensor* output, const Tensor* output_h, const Tensor* output_c,
    /* backprop inputs */
    const Tensor* output_backprop, const Tensor* output_h_backprop,
    const Tensor* output_c_backprop, const Tensor* reserve_space,
    /* backprop outputs, output of the function */
    Tensor* input_backprop, Tensor* input_h_backprop, Tensor* input_c_backprop,
    Tensor* params_backprop, const Tensor* sequence_lengths, bool time_major,
    ScratchAllocator* workspace_allocator,
    ProfileResult* output_profile_result) {
  // CudnnRnn doesn't support bfloat16 yet, so cast to float. This avoids errors
  // with mixed_bfloat16 policy.
  Tensor casted_input;
  Tensor casted_input_h;
  Tensor casted_input_c;
  Tensor casted_params;
  Tensor casted_output;
  Tensor casted_output_h;
  Tensor casted_output_c;
  Tensor casted_output_backprop;
  Tensor casted_output_h_backprop;
  Tensor casted_output_c_backprop;
  Tensor casted_input_backprop;
  Tensor casted_input_h_backprop;
  Tensor casted_input_c_backprop;
  Tensor casted_params_backprop;
  const GPUDevice& device = context->eigen_device<GPUDevice>();
  functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
  auto allocate_and_cast_to_float = [&](const Tensor* tensor,
                                        Tensor* casted_tensor) {
    TF_RETURN_IF_ERROR(context, context->allocate_temp(
                                    DT_FLOAT, tensor->shape(), casted_tensor));
    cast(device, casted_tensor->template flat<float>(),
         tensor->template flat<Eigen::bfloat16>());
    return OkStatus();
  };
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(input, &casted_input));
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(input_h, &casted_input_h));
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(params, &casted_params));
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(output, &casted_output));
  TF_RETURN_IF_ERROR(allocate_and_cast_to_float(output_h, &casted_output_h));
  TF_RETURN_IF_ERROR(
      allocate_and_cast_to_float(output_backprop, &casted_output_backprop));
  TF_RETURN_IF_ERROR(
      allocate_and_cast_to_float(output_h_backprop, &casted_output_h_backprop));
  TF_RETURN_IF_ERROR(context,
                     context->allocate_temp(DT_FLOAT, input_backprop->shape(),
                                            &casted_input_backprop));
  TF_RETURN_IF_ERROR(context,
                     context->allocate_temp(DT_FLOAT, input_h_backprop->shape(),
                                            &casted_input_h_backprop));
  TF_RETURN_IF_ERROR(context,
                     context->allocate_temp(DT_FLOAT, params_backprop->shape(),
                                            &casted_params_backprop));
  if (model_types.HasInputC()) {
    TF_RETURN_IF_ERROR(allocate_and_cast_to_float(input_c, &casted_input_c));
    TF_RETURN_IF_ERROR(allocate_and_cast_to_float(output_c, &casted_output_c));
    TF_RETURN_IF_ERROR(allocate_and_cast_to_float(output_c_backprop,
                                                  &casted_output_c_backprop));
    TF_RETURN_IF_ERROR(
        context, context->allocate_temp(DT_FLOAT, input_c_backprop->shape(),
                                        &casted_input_c_backprop));
  }

  TF_RETURN_IF_ERROR(DoBackwardImpl<float>(
      context, rnn_desc, model_types, model_shapes, &casted_input,
      &casted_input_h, &casted_input_c, &casted_params, &casted_output,
      &casted_output_h, &casted_output_c, &casted_output_backprop,
      &casted_output_h_backprop, &casted_output_c_backprop, reserve_space,
      &casted_input_backprop, &casted_input_h_backprop,
      &casted_input_c_backprop, &casted_params_backprop, sequence_lengths,
      time_major, workspace_allocator, output_profile_result));

  functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
  cast_back(device, input_backprop->template flat<Eigen::bfloat16>(),
            const_cast<const Tensor*>(&casted_input_backprop)
                ->template flat<float>());
  cast_back(device, input_h_backprop->template flat<Eigen::bfloat16>(),
            const_cast<const Tensor*>(&casted_input_h_backprop)
                ->template flat<float>());
  cast_back(device, input_c_backprop->template flat<Eigen::bfloat16>(),
            const_cast<const Tensor*>(&casted_input_c_backprop)
                ->template flat<float>());
  cast_back(device, params_backprop->template flat<Eigen::bfloat16>(),
            const_cast<const Tensor*>(&casted_params_backprop)
                ->template flat<float>());
  return OkStatus();
}

template <typename T>
void RestoreParams(const OpInputList params_input,
                   const std::vector<RnnDescriptor::ParamsRegion>& params,
                   DeviceMemoryBase* data_dst, Stream* stream) {
  int num_params = params.size();
  CHECK(params_input.size() == num_params)
      << "Number of params mismatch. Expected " << params_input.size()
      << ", got " << num_params;
  for (int i = 0; i < params.size(); i++) {
    int64_t size_in_bytes = params[i].size;
    int64_t size = size_in_bytes / sizeof(T);
    CHECK(size == params_input[i].NumElements())
        << "Params size mismatch. Expected " << size << ", got "
        << params_input[i].NumElements();
    auto data_src_ptr = StreamExecutorUtil::AsDeviceMemory<T>(params_input[i]);
    DeviceMemoryBase data_dst_ptr =
        SliceDeviceMemory(*data_dst, params[i].offset, size_in_bytes);
    stream->ThenMemcpy(&data_dst_ptr, data_src_ptr, size_in_bytes);
  }
}

bool ShouldUsePaddedIO(const Tensor* sequence_lengths,
                       const CudnnRnnModelShapes& model_shapes,
                       bool time_major) {
  auto seq_array = sequence_lengths->template flat<int>().data();
  bool all_max_seq_length = true;
  for (int i = 0; i < model_shapes.batch_size; i++) {
    if (seq_array[i] != model_shapes.max_seq_length) {
      all_max_seq_length = false;
      break;
    }
  }
  return !(time_major && all_max_seq_length);
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
  const CudnnModelTypes& model_types() const { return model_types_; }
  float dropout() const { return dropout_; }
  uint64 seed() { return (static_cast<uint64>(seed_) << 32) | seed2_; }
  bool ResetRndGenState() { return reset_rnd_gen_state_; }

  template <typename T>
  Status ExtractCudnnRNNParamsInfo(OpKernelContext* context, int num_proj,
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

    int h_num_units = (num_proj == 0 ? num_units : num_proj);
    int c_num_units = (num_proj == 0 ? 0 : num_units);

    RnnInputMode input_mode;
    TF_RETURN_IF_ERROR(
        ToRNNInputMode(rnn_input_mode(), num_units, input_size, &input_mode));

    Stream* stream = context->op_device_context()->stream();
    // ExtracCudnnRNNParamsInfo is only called by op_kernels that do not require
    // random number generator, therefore set state_allocator to nullptr.
    const AlgorithmConfig algo_config;
    auto rnn_desc_s = stream->parent()->createRnnDescriptor(
        num_layers, h_num_units, input_size, /*cell_size=*/c_num_units,
        /*batch_size=*/0, input_mode, rnn_direction_mode(), rnn_mode(),
        ToDataType<T>::value, algo_config, dropout(), seed(),
        /* state_allocator=*/nullptr, /*use_padded_io=*/false);
    if (!rnn_desc_s.ok()) {
      return FromExecutorStatus(rnn_desc_s);
    }
    *rnn_desc = std::move(rnn_desc_s).value();
    return OkStatus();
  }

  template <typename T>
  Status CreateRnnDescriptor(OpKernelContext* context,
                             const CudnnRnnModelShapes& model_shapes,
                             const RnnInputMode& input_mode,
                             const AlgorithmConfig& algo_config,
                             ScratchAllocator* dropout_state_allocator,
                             std::unique_ptr<RnnDescriptor>* rnn_desc,
                             bool use_padded_io) {
    StreamExecutor* executor = context->op_device_context()->stream()->parent();
    se::dnn::DataType data_type = ToDataType<T>::value;
    auto rnn_desc_s = executor->createRnnDescriptor(
        model_shapes.num_layers, model_shapes.num_units,
        model_shapes.input_size, model_shapes.cell_num_units,
        model_shapes.batch_size, input_mode, rnn_direction_mode(), rnn_mode(),
        data_type, algo_config, dropout(), seed(), dropout_state_allocator,
        use_padded_io);
    TF_RETURN_IF_ERROR(rnn_desc_s.status());

    *rnn_desc = std::move(rnn_desc_s).value();
    return OkStatus();
  }

  using RnnStateCache = gtl::FlatMap<
      std::pair<CudnnRnnModelShapes, absl::optional<AlgorithmDesc>>,
      RnnScratchSpace, CudnnRnnConfigHasher, CudnnRnnConfigComparator>;
  // Returns a raw rnn descriptor pointer. The cache owns the rnn descriptor and
  // should outlive the returned pointer.
  template <typename T>
  Status GetCachedRnnDescriptor(OpKernelContext* context,
                                const CudnnRnnModelShapes& model_shapes,
                                const RnnInputMode& input_mode,
                                const AlgorithmConfig& algo_config,
                                RnnStateCache* cache, RnnDescriptor** rnn_desc,
                                bool use_padded_io) {
    auto key = std::make_pair(model_shapes, algo_config.algorithm());
    RnnScratchSpace& rnn_state = (*cache)[key];
    if (rnn_state.rnn_desc == nullptr || ResetRndGenState()) {
      CudnnRNNSpaceAllocator* dropout_state_allocator =
          new CudnnRNNSpaceAllocator(context);
      rnn_state.dropout_state_allocator.reset(dropout_state_allocator);
      Status status = CreateRnnDescriptor<T>(
          context, model_shapes, input_mode, algo_config,
          dropout_state_allocator, &rnn_state.rnn_desc, use_padded_io);
      TF_RETURN_IF_ERROR(status);
    }
    *rnn_desc = rnn_state.rnn_desc.get();
    return OkStatus();
  }

 private:
  int seed_;
  int seed2_;
  float dropout_;
  bool reset_rnd_gen_state_;

  CudnnModelTypes model_types_;
};

template <>
Status CudnnRNNKernelCommon::CreateRnnDescriptor<bfloat16>(
    OpKernelContext* context, const CudnnRnnModelShapes& model_shapes,
    const RnnInputMode& input_mode, const AlgorithmConfig& algo_config,
    ScratchAllocator* dropout_state_allocator,
    std::unique_ptr<RnnDescriptor>* rnn_desc, bool use_padded_io) {
  StreamExecutor* executor = context->op_device_context()->stream()->parent();
  se::dnn::DataType data_type = ToDataType<float>::value;
  auto rnn_desc_s = executor->createRnnDescriptor(
      model_shapes.num_layers, model_shapes.num_units, model_shapes.input_size,
      model_shapes.cell_num_units, model_shapes.batch_size, input_mode,
      rnn_direction_mode(), rnn_mode(), data_type, algo_config, dropout(),
      seed(), dropout_state_allocator, use_padded_io);
  TF_RETURN_IF_ERROR(rnn_desc_s.status());

  *rnn_desc = std::move(rnn_desc_s).value();
  return OkStatus();
}

template <>
Status CudnnRNNKernelCommon::GetCachedRnnDescriptor<bfloat16>(
    OpKernelContext* context, const CudnnRnnModelShapes& model_shapes,
    const RnnInputMode& input_mode, const AlgorithmConfig& algo_config,
    RnnStateCache* cache, RnnDescriptor** rnn_desc, bool use_padded_io) {
  auto key = std::make_pair(model_shapes, algo_config.algorithm());
  RnnScratchSpace& rnn_state = (*cache)[key];
  if (rnn_state.rnn_desc == nullptr || ResetRndGenState()) {
    CudnnRNNSpaceAllocator* dropout_state_allocator =
        new CudnnRNNSpaceAllocator(context);
    rnn_state.dropout_state_allocator.reset(dropout_state_allocator);
    Status status = CreateRnnDescriptor<float>(
        context, model_shapes, input_mode, algo_config, dropout_state_allocator,
        &rnn_state.rnn_desc, use_padded_io);
    TF_RETURN_IF_ERROR(status);
  }
  *rnn_desc = rnn_state.rnn_desc.get();
  return OkStatus();
}

// A class that returns the size of the opaque parameter buffer. The user should
// use that to create the actual parameter buffer for training. However, it
// should not be used for saving and restoring.
template <typename T, typename Index>
class CudnnRNNParamsSizeOp<GPUDevice, T, Index> : public CudnnRNNKernelCommon {
 public:
  explicit CudnnRNNParamsSizeOp(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {
    if (context->HasAttr("num_proj")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_proj", &num_proj_));
    } else {
      num_proj_ = 0;
    }
  }

  void Compute(OpKernelContext* context) override {
    std::unique_ptr<RnnDescriptor> rnn_desc;
    OP_REQUIRES_OK(context,
                   ExtractCudnnRNNParamsInfo<T>(context, num_proj_, &rnn_desc));
    int64_t params_size_in_bytes = rnn_desc->ParamsSizeInBytes();
    CHECK(params_size_in_bytes % sizeof(T) == 0)
        << "params_size_in_bytes must be multiple of element size";
    int64_t params_size = params_size_in_bytes / sizeof(T);

    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_t));
    *output_t->template flat<Index>().data() = params_size;
  }

 private:
  int num_proj_;
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

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// Convert weight and bias params from a platform-specific layout to the
// canonical form.
template <typename T>
class CudnnRNNParamsToCanonical<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  explicit CudnnRNNParamsToCanonical(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {
    if (context->HasAttr("num_params")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_params", &num_params_));
    } else {
      num_params_ = 0;
    }
    if (context->HasAttr("num_params_weights")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_params_weights",
                                               &num_params_weights_));
    } else {
      num_params_weights_ = 0;
    }
    if (context->HasAttr("num_params_biases")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("num_params_biases", &num_params_biases_));
    } else {
      num_params_biases_ = 0;
    }
    if (context->HasAttr("num_proj")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_proj", &num_proj_));
    } else {
      num_proj_ = 0;
    }
    if (num_proj_ == 0) {
      num_params_weights_ = num_params_;
      num_params_biases_ = num_params_;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(3);
    auto input_ptr = StreamExecutorUtil::AsDeviceMemory<T>(input);
    Stream* stream = context->op_device_context()->stream();

    std::unique_ptr<RnnDescriptor> rnn_desc;
    OP_REQUIRES_OK(context,
                   ExtractCudnnRNNParamsInfo<T>(context, num_proj_, &rnn_desc));
    int64_t params_size_in_bytes = rnn_desc->ParamsSizeInBytes();
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
    const int num_params_weights_per_layer =
        num_params_weights_ / num_layers / num_dirs;
    // Number of params applied on inputs. The rest are applied on recurrent
    // hidden states.
    const int num_params_input_state = num_params_weights_per_layer / 2;
    OP_REQUIRES(
        context, num_params_weights_ % (num_layers * num_dirs) == 0,
        errors::InvalidArgument("Number of params (weights) is not a multiple"
                                "of num_layers * num_dirs."));
    OP_REQUIRES(
        context, num_params_biases_ % (num_layers * num_dirs) == 0,
        errors::InvalidArgument("Number of params (biases) is not a multiple"
                                "of num_layers * num_dirs."));
    if (num_proj_ == 0) {
      OP_REQUIRES(
          context, num_params_weights_per_layer % 2 == 0,
          errors::InvalidArgument("Number of params (weights) per layer is not"
                                  "an even number with no projection."));
    } else {
      OP_REQUIRES(
          context, num_params_weights_per_layer % 2 != 0,
          errors::InvalidArgument("Number of params (weights) per layer is not"
                                  "an odl number with projection."));
    }

    OP_REQUIRES(
        context, num_params_weights_ == rnn_desc->ParamsWeightRegions().size(),
        errors::InvalidArgument("C Number of params mismatch. Expected ",
                                num_params_weights_, ", got ",
                                rnn_desc->ParamsWeightRegions().size()));
    int h_num_units = (num_proj_ == 0 ? num_units : num_proj_);
    int c_num_units = (num_proj_ == 0 ? 0 : num_units);
    for (int i = 0; i < rnn_desc->ParamsWeightRegions().size(); i++) {
      int64_t size_in_bytes = rnn_desc->ParamsWeightRegions()[i].size;
      int64_t size = size_in_bytes / sizeof(T);
      const int layer_idx = i / num_params_weights_per_layer;
      const int index_within_layer = i % num_params_weights_per_layer;
      int width = 0, height = (num_proj_ == 0 ? h_num_units : c_num_units);
      // In CuDNN layout, each layer has num_params_weights_per_layer params,
      // with the
      // first half a.k.a num_params_input_state params applied on the inputs,
      // and the second half on the recurrent hidden states.
      bool apply_on_input_state = index_within_layer < num_params_input_state;
      if (rnn_direction_mode() == RnnDirectionMode::kRnnUnidirectional) {
        if (layer_idx == 0 && apply_on_input_state) {
          width = input_size;
        } else {
          width = h_num_units;
        }
      } else {
        if (apply_on_input_state) {
          if (layer_idx <= 1) {
            // First fwd or bak layer.
            width = input_size;
          } else {
            // Following layers, cell inputs are concatenated outputs of
            // its prior layer.
            width = 2 * h_num_units;
          }
        } else {
          width = h_num_units;
        }
      }
      CHECK(size == width * height) << "Params size mismatch. Expected "
                                    << width * height << ", got " << size;
      Tensor* output = nullptr;
      int id_in_layer = i % num_params_weights_per_layer;
      if (num_proj_ != 0 && id_in_layer == num_params_weights_per_layer - 1) {
        std::swap(height, width);
      }
      OP_REQUIRES_OK(context, context->allocate_output(
                                  i, TensorShape({height, width}), &output));
      DeviceMemoryBase data_src_ptr = SliceDeviceMemory(
          input_ptr, rnn_desc->ParamsWeightRegions()[i].offset, size_in_bytes);
      auto data_dst_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*output);
      stream->ThenMemcpy(&data_dst_ptr, data_src_ptr, size_in_bytes);
    }

    OP_REQUIRES(
        context, num_params_biases_ == rnn_desc->ParamsBiasRegions().size(),
        errors::InvalidArgument("A Number of params mismatch. Expected ",
                                num_params_biases_, ", got ",
                                rnn_desc->ParamsBiasRegions().size()));
    for (int i = 0; i < rnn_desc->ParamsBiasRegions().size(); i++) {
      int64_t size_in_bytes = rnn_desc->ParamsBiasRegions()[i].size;
      int64_t size = size_in_bytes / sizeof(T);
      OP_REQUIRES(context, size == num_units,
                  errors::InvalidArgument("Params size mismatch. Expected ",
                                          num_units, ", got ", size));

      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(num_params_weights_ + i,
                                              TensorShape({size}), &output));
      DeviceMemoryBase data_src_ptr = SliceDeviceMemory(
          input_ptr, rnn_desc->ParamsBiasRegions()[i].offset, size_in_bytes);
      auto data_dst_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*output);
      stream->ThenMemcpy(&data_dst_ptr, data_src_ptr, size_in_bytes);
    }
  }

 private:
  int num_params_;
  int num_params_weights_;
  int num_params_biases_;
  int num_proj_;
};

#define REGISTER_GPU(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNParamsToCanonical") \
                              .Device(DEVICE_GPU)           \
                              .HostMemory("num_layers")     \
                              .HostMemory("num_units")      \
                              .HostMemory("input_size")     \
                              .TypeConstraint<T>("T"),      \
                          CudnnRNNParamsToCanonical<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#define REGISTER_GPU(T)                                       \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNParamsToCanonicalV2") \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("num_layers")       \
                              .HostMemory("num_units")        \
                              .HostMemory("input_size")       \
                              .TypeConstraint<T>("T"),        \
                          CudnnRNNParamsToCanonical<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// Convert weight and bias params from the canonical form to a
// platform-specific layout.
template <typename T>
class CudnnRNNCanonicalToParams<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  explicit CudnnRNNCanonicalToParams(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {
    if (context->HasAttr("num_proj")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_proj", &num_proj_));
    } else {
      num_proj_ = 0;
    }
  }

  void Compute(OpKernelContext* context) override {
    std::unique_ptr<RnnDescriptor> rnn_desc;
    OP_REQUIRES_OK(context,
                   ExtractCudnnRNNParamsInfo<T>(context, num_proj_, &rnn_desc));
    int64_t params_size_in_bytes = rnn_desc->ParamsSizeInBytes();
    CHECK(params_size_in_bytes % sizeof(T) == 0)
        << "params_size_in_bytes must be multiple of element size";
    Tensor* output = nullptr;
    int params_size = params_size_in_bytes / sizeof(T);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {params_size}, &output));
    auto output_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*output);
    Stream* stream = context->op_device_context()->stream();

    OpInputList weights;
    OP_REQUIRES_OK(context, context->input_list("weights", &weights));
    RestoreParams<T>(weights, rnn_desc->ParamsWeightRegions(), &output_ptr,
                     stream);

    OpInputList biases;
    OP_REQUIRES_OK(context, context->input_list("biases", &biases));
    RestoreParams<T>(biases, rnn_desc->ParamsBiasRegions(), &output_ptr,
                     stream);
  }

 private:
  int num_proj_;
};

#define REGISTER_GPU(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNCanonicalToParams") \
                              .Device(DEVICE_GPU)           \
                              .HostMemory("num_layers")     \
                              .HostMemory("num_units")      \
                              .HostMemory("input_size")     \
                              .TypeConstraint<T>("T"),      \
                          CudnnRNNCanonicalToParams<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#define REGISTER_GPU(T)                                       \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNCanonicalToParamsV2") \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("num_layers")       \
                              .HostMemory("num_units")        \
                              .HostMemory("input_size")       \
                              .TypeConstraint<T>("T"),        \
                          CudnnRNNCanonicalToParams<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// Run the forward operation of the RNN model.
template <typename T>
class CudnnRNNForwardOp<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  explicit CudnnRNNForwardOp(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    // Read debug env variables.
    is_debug_mode_ = DebugCudnnRnn();
    debug_cudnn_rnn_algo_ = DebugCudnnRnnAlgo();
    debug_use_tensor_ops_ = DebugCudnnRnnUseTensorOps();
  }

  void Compute(OpKernelContext* context) override {
    AlgorithmConfig algo_config;
    ComputeAndReturnAlgorithm(context, &algo_config, /*var_seq_lengths=*/false,
                              /*time_major=*/true, /*num_proj=*/0);
  }

 protected:
  virtual void ComputeAndReturnAlgorithm(OpKernelContext* context,
                                         AlgorithmConfig* output_algo_config,
                                         bool var_seq_lengths, bool time_major,
                                         int num_proj) {
    CHECK_NE(output_algo_config, nullptr);

    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    const Tensor* sequence_lengths = nullptr;
    CudnnRnnModelShapes model_shapes;
    bool use_padded_io = false;
    if (var_seq_lengths) {
      OP_REQUIRES_OK(context, ExtractForwardInput(
                                  context, model_types(), time_major, &input,
                                  &input_h, &input_c, &params,
                                  &sequence_lengths, num_proj, &model_shapes));
      use_padded_io =
          ShouldUsePaddedIO(sequence_lengths, model_shapes, time_major);
    } else {
      OP_REQUIRES_OK(context,
                     ExtractForwardInput(context, model_types(), time_major,
                                         &input, &input_h, &input_c, &params,
                                         num_proj, &model_shapes));
    }
    RnnInputMode input_mode;
    OP_REQUIRES_OK(context,
                   ToRNNInputMode(rnn_input_mode(), model_shapes.num_units,
                                  model_shapes.input_size, &input_mode));

    Tensor* output = nullptr;
    Tensor* output_h = nullptr;
    Tensor* output_c = nullptr;
    OP_REQUIRES_OK(context, AllocateOutputs(context, model_shapes, &output,
                                            &output_h, &output_c));

    // Creates a memory callback for the reserve_space. The memory lives in the
    // output of this kernel. And it will be fed into the backward pass when
    // needed.
    CudnnRnnAllocatorInOutput<T> reserve_space_allocator(context, 3);
    // Creates a memory callback for the workspace. The memory lives to the end
    // of this kernel calls.
    CudnnRnnAllocatorInTemp<uint8> workspace_allocator(context);

    if (is_debug_mode_) {
      AlgorithmDesc algo_desc(debug_cudnn_rnn_algo_, debug_use_tensor_ops_,
                              absl::nullopt);
      output_algo_config->set_algorithm(algo_desc);
    } else {
      OP_REQUIRES_OK(context,
                     MaybeAutotune(context, model_shapes, input_mode, input,
                                   input_h, input_c, params, output, output_h,
                                   output_c, output_algo_config));
    }

    Status launch_status;
    {
      mutex_lock l(mu_);
      RnnDescriptor* rnn_desc_ptr = nullptr;
      OP_REQUIRES_OK(context,
                     GetCachedRnnDescriptor<T>(
                         context, model_shapes, input_mode, *output_algo_config,
                         &rnn_state_cache_, &rnn_desc_ptr, use_padded_io));
      launch_status = DoForward<T>(
          context, *rnn_desc_ptr, model_types(), model_shapes, input, input_h,
          input_c, params, is_training_, output, output_h, output_c,
          sequence_lengths, time_major, &reserve_space_allocator,
          &workspace_allocator, /*output_profile_result=*/nullptr);
    }
    OP_REQUIRES_OK(context, launch_status);
  }

 protected:
  virtual Status MaybeAutotune(OpKernelContext* context,
                               const CudnnRnnModelShapes& model_shapes,
                               const RnnInputMode& input_mode,
                               const Tensor* input, const Tensor* input_h,
                               const Tensor* input_c, const Tensor* params,
                               Tensor* output, Tensor* output_h,
                               Tensor* output_c,
                               AlgorithmConfig* best_algo_config) {
    CHECK_NE(best_algo_config, nullptr);
    *best_algo_config = AlgorithmConfig();
    return OkStatus();
  }

  bool is_training() const { return is_training_; }
  bool is_debug_mode_;
  bool debug_use_tensor_ops_;
  int64_t debug_cudnn_rnn_algo_;

 private:
  Status AllocateOutputs(OpKernelContext* context,
                         const CudnnRnnModelShapes& model_shapes,
                         Tensor** output, Tensor** output_h,
                         Tensor** output_c) {
    const TensorShape& hidden_state_shape = model_shapes.hidden_state_shape;
    const TensorShape& output_shape = model_shapes.output_shape;
    const TensorShape& cell_state_shape = model_shapes.cell_state_shape;

    TF_RETURN_IF_ERROR(context->allocate_output(0, output_shape, output));
    TF_RETURN_IF_ERROR(
        context->allocate_output(1, hidden_state_shape, output_h));
    if (HasInputC()) {
      TF_RETURN_IF_ERROR(
          context->allocate_output(2, cell_state_shape, output_c));
    } else {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      TF_RETURN_IF_ERROR(context->allocate_output(2, {}, output_c));
    }
    if (!is_training_) {
      Tensor* dummy_reserve_space = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(3, {}, &dummy_reserve_space));
    }
    return OkStatus();
  }

  mutex mu_;
  bool is_training_;
  RnnStateCache rnn_state_cache_ TF_GUARDED_BY(mu_);
};

#define REGISTER_GPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("CudnnRNN").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CudnnRNNForwardOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

template <typename T>
class CudnnRNNForwardOpV2<GPUDevice, T>
    : public CudnnRNNForwardOp<GPUDevice, T> {
 private:
  using CudnnRNNForwardOp<GPUDevice, T>::is_training;
  using CudnnRNNKernelCommon::CreateRnnDescriptor;
  using CudnnRNNKernelCommon::dropout;
  using CudnnRNNKernelCommon::HasInputC;
  using CudnnRNNKernelCommon::model_types;

 public:
  explicit CudnnRNNForwardOpV2(OpKernelConstruction* context)
      : CudnnRNNForwardOp<GPUDevice, T>(context) {}

  void Compute(OpKernelContext* context) override {
    AlgorithmConfig best_algo_config;
    CudnnRNNForwardOp<GPUDevice, T>::ComputeAndReturnAlgorithm(
        context, &best_algo_config, /*var_seq_lengths=*/false,
        /*time_major=*/true, /*num_proj=*/0);
    if (!context->status().ok()) {
      return;
    }

    Tensor* output_host_reserved = nullptr;
    // output_host_reserved stores opaque info used for backprop when running
    // in training mode. At present, it includes a serialization of the best
    // AlgorithmDesc picked during rnn forward pass autotune.
    // int8 algorithm_id
    // int8 use_tensor_op
    // If autotune is not enabled, the algorithm_id is
    // stream_executor::dnn::kDefaultAlgorithm and use_tensor_op is false. If
    // running in inference mode, the output_host_reserved is currently not
    // populated.
    if (is_training()) {
      OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({2}),
                                                       &output_host_reserved));
      auto output_host_reserved_int8 = output_host_reserved->vec<int8>();
      output_host_reserved_int8(0) = best_algo_config.algorithm()->algo_id();
      output_host_reserved_int8(1) =
          best_algo_config.algorithm()->tensor_ops_enabled();
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(4, {}, &output_host_reserved));
    }
  }

 protected:
  Status MaybeAutotune(OpKernelContext* context,
                       const CudnnRnnModelShapes& model_shapes,
                       const RnnInputMode& input_mode, const Tensor* input,
                       const Tensor* input_h, const Tensor* input_c,
                       const Tensor* params, Tensor* output, Tensor* output_h,
                       Tensor* output_c,
                       AlgorithmConfig* algo_config) override {
    CHECK_NE(algo_config, nullptr);
    if (!CudnnRnnUseAutotune() || this->is_debug_mode_) {
      *algo_config = AlgorithmConfig();
      return OkStatus();
    }

    std::vector<AlgorithmDesc> algorithms;
    auto* stream = context->op_device_context()->stream();
    CHECK(stream->parent()->GetRnnAlgorithms(&algorithms));
    if (algorithms.empty()) {
      LOG(WARNING) << "No Rnn algorithm found";
      return OkStatus();
    }

    const auto& modeltypes = model_types();
    CudnnRnnParameters rnn_params(
        model_shapes.num_layers, model_shapes.input_size,
        model_shapes.num_units, model_shapes.max_seq_length,
        model_shapes.batch_size, model_shapes.dir_count,
        /*has_dropout=*/std::abs(dropout()) > 1e-8, is_training(),
        modeltypes.rnn_mode, modeltypes.rnn_input_mode, input->dtype());

    if (AutotuneRnnConfigMap::GetInstance()->Find(rnn_params, algo_config)) {
      VLOG(1) << "Using existing best Cudnn RNN algorithm "
              << "(algo, tensor_op_enabled) = ("
              << algo_config->algorithm()->algo_id() << ", "
              << algo_config->algorithm()->tensor_ops_enabled() << ").";
      return OkStatus();
    }
    profiler::ScopedAnnotation trace("cudnn_autotuning");

    // Create temp tensors when profiling backprop pass.
    auto data_type = input->dtype();
    Tensor output_backprop;
    Tensor output_h_backprop;
    Tensor output_c_backprop;
    Tensor input_backprop;
    Tensor input_h_backprop;
    Tensor input_c_backprop;
    Tensor params_backprop;
    if (is_training()) {
      TF_RETURN_IF_ERROR(context->allocate_temp(
          data_type, model_shapes.output_shape, &output_backprop));
      TF_RETURN_IF_ERROR(context->allocate_temp(
          data_type, model_shapes.hidden_state_shape, &output_h_backprop));

      TF_RETURN_IF_ERROR(
          context->allocate_temp(data_type, params->shape(), &params_backprop));
      TF_RETURN_IF_ERROR(context->allocate_temp(
          data_type, model_shapes.input_shape, &input_backprop));
      TF_RETURN_IF_ERROR(context->allocate_temp(
          data_type, model_shapes.hidden_state_shape, &input_h_backprop));
      if (HasInputC()) {
        TF_RETURN_IF_ERROR(context->allocate_temp(
            data_type, model_shapes.hidden_state_shape, &output_c_backprop));
        TF_RETURN_IF_ERROR(context->allocate_temp(
            data_type, model_shapes.hidden_state_shape, &input_c_backprop));
      }
    }
    ProfileResult best_result;
    for (auto& algo : algorithms) {
      VLOG(1) << "Profile Cudnn RNN algorithm (algo, tensor_op_enabled) =  ("
              << algo.algo_id() << ", " << algo.tensor_ops_enabled() << ").";
      Status status;
      ProfileResult final_profile_result;

      ProfileResult fwd_profile_result;
      ProfileResult bak_profile_result;

      // RnnDescriptor is algorithm-dependent, thus not reusable.
      std::unique_ptr<RnnDescriptor> rnn_desc;
      // Use a temp scratch allocator for the random num generator.
      CudnnRnnAllocatorInTemp<uint8> dropout_state_allocator(context);
      if (!this->template CreateRnnDescriptor<T>(
                   context, model_shapes, input_mode, AlgorithmConfig(algo),
                   &dropout_state_allocator, &rnn_desc,
                   /*use_padded_io=*/false)
               .ok()) {
        continue;
      }

      // Again use temp scratch allocator during profiling.
      CudnnRnnAllocatorInTemp<T> reserve_space_allocator(context);
      CudnnRnnAllocatorInTemp<uint8> workspace_allocator(context);
      status = DoForward<T>(context, *rnn_desc, model_types(), model_shapes,
                            input, input_h, input_c, params, is_training(),
                            output, output_h, output_c, nullptr, true,
                            &reserve_space_allocator, &workspace_allocator,
                            &fwd_profile_result);
      if (!status.ok()) {
        continue;
      }

      if (is_training()) {
        // Get reserve space from the forward pass.
        Tensor reserve_space = reserve_space_allocator.get_allocated_tensor(0);
        status = DoBackward<T>(
            context, *rnn_desc, model_types(), model_shapes, input, input_h,
            input_c, params, output, output_h, output_c, &output_backprop,
            &output_h_backprop, &output_c_backprop, &reserve_space,
            &input_backprop, &input_h_backprop, &input_c_backprop,
            &params_backprop, nullptr, true, &workspace_allocator,
            &bak_profile_result);
        if (!status.ok()) {
          continue;
        }
        final_profile_result.set_elapsed_time_in_ms(
            fwd_profile_result.elapsed_time_in_ms() +
            bak_profile_result.elapsed_time_in_ms());
      } else {
        final_profile_result = fwd_profile_result;
      }

      auto total_time = final_profile_result.elapsed_time_in_ms();
      VLOG(1) << "Cudnn RNN algorithm (algo, tensor_op_enabled) =  ("
              << algo.algo_id() << ", " << algo.tensor_ops_enabled() << ")"
              << " run time: " << total_time << " ms.";
      if (total_time < best_result.elapsed_time_in_ms()) {
        best_result.set_elapsed_time_in_ms(total_time);
        best_result.set_algorithm(algo);
      }
    }

    if (!best_result.is_valid()) {
      return Status(absl::StatusCode::kInternal, "No algorithm worked!");
    }
    algo_config->set_algorithm(best_result.algorithm());
    VLOG(1) << "Best Cudnn RNN algorithm (algo, tensor_op_enabled) =  ("
            << best_result.algorithm().algo_id() << ", "
            << best_result.algorithm().tensor_ops_enabled() << ").";
    AutotuneRnnConfigMap::GetInstance()->Insert(rnn_params, *algo_config);
    return OkStatus();
  }
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNV2")               \
                              .Device(DEVICE_GPU)          \
                              .HostMemory("host_reserved") \
                              .TypeConstraint<T>("T"),     \
                          CudnnRNNForwardOpV2<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

template <typename T>
class CudnnRNNForwardOpV3<GPUDevice, T>
    : public CudnnRNNForwardOp<GPUDevice, T> {
 private:
  using CudnnRNNForwardOp<GPUDevice, T>::is_training;
  using CudnnRNNKernelCommon::CreateRnnDescriptor;
  using CudnnRNNKernelCommon::dropout;
  using CudnnRNNKernelCommon::HasInputC;
  using CudnnRNNKernelCommon::model_types;
  bool time_major_;

 protected:
  bool time_major() { return time_major_; }

 public:
  explicit CudnnRNNForwardOpV3(OpKernelConstruction* context)
      : CudnnRNNForwardOp<GPUDevice, T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("time_major", &time_major_));
    if (context->HasAttr("num_proj")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_proj", &num_proj_));
    } else {
      num_proj_ = 0;
    }
  }

  void Compute(OpKernelContext* context) override {
    AlgorithmConfig best_algo_config;
    CudnnRNNForwardOp<GPUDevice, T>::ComputeAndReturnAlgorithm(
        context, &best_algo_config, /*var_seq_lengths=*/true,
        /*time_major=*/time_major(), num_proj_);
    if (!context->status().ok()) {
      return;
    }

    Tensor* output_host_reserved = nullptr;
    // TODO: Current V3 only uses the default standard algorithm to process
    // batches with variable sequences and the inputs should be padded.
    // Autotune is not supported yet.
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, {}, &output_host_reserved));
  }

 private:
  int num_proj_;
};

#define REGISTER_GPU(T)                                       \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNV3")                  \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("sequence_lengths") \
                              .HostMemory("host_reserved")    \
                              .TypeConstraint<T>("T"),        \
                          CudnnRNNForwardOpV3<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// Run the backward operation of the RNN model.
template <typename T>
class CudnnRNNBackwardOp<GPUDevice, T> : public CudnnRNNKernelCommon {
 public:
  explicit CudnnRNNBackwardOp(OpKernelConstruction* context)
      : CudnnRNNKernelCommon(context) {}

  void Compute(OpKernelContext* context) override {
    ComputeImpl(context, false, true, 0);
  }

 protected:
  virtual void ComputeImpl(OpKernelContext* context, bool var_seq_lengths,
                           bool time_major, int num_proj) {
    const Tensor* input = nullptr;
    const Tensor* input_h = nullptr;
    const Tensor* input_c = nullptr;
    const Tensor* params = nullptr;
    const Tensor* sequence_lengths = nullptr;
    CudnnRnnModelShapes model_shapes;
    bool use_padded_io = false;
    if (var_seq_lengths) {
      OP_REQUIRES_OK(context, ExtractForwardInput(
                                  context, model_types(), time_major, &input,
                                  &input_h, &input_c, &params,
                                  &sequence_lengths, num_proj, &model_shapes));
      use_padded_io =
          ShouldUsePaddedIO(sequence_lengths, model_shapes, time_major);
    } else {
      OP_REQUIRES_OK(context,
                     ExtractForwardInput(context, model_types(), time_major,
                                         &input, &input_h, &input_c, &params,
                                         num_proj, &model_shapes));
    }
    RnnInputMode input_mode;
    OP_REQUIRES_OK(context,
                   ToRNNInputMode(rnn_input_mode(), model_shapes.num_units,
                                  model_shapes.input_size, &input_mode));

    const Tensor* output = nullptr;
    const Tensor* output_h = nullptr;
    const Tensor* output_c = nullptr;
    const Tensor* output_backprop = nullptr;
    const Tensor* output_h_backprop = nullptr;
    const Tensor* output_c_backprop = nullptr;
    const Tensor* reserve_space = nullptr;
    OP_REQUIRES_OK(context,
                   ExtractBackwardInputs(context, model_shapes, model_types(),
                                         &output, &output_h, &output_c,
                                         &output_backprop, &output_h_backprop,
                                         &output_c_backprop, &reserve_space));

    Tensor* input_backprop = nullptr;
    Tensor* input_h_backprop = nullptr;
    Tensor* input_c_backprop = nullptr;
    Tensor* params_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   AllocateOutputs(context, model_shapes, params->shape(),
                                   &input_backprop, &input_h_backprop,
                                   &input_c_backprop, &params_backprop));

    // Creates a memory callback for the workspace. The memory lives to the end
    // of this kernel calls.
    CudnnRnnAllocatorInTemp<uint8> workspace_allocator(context);
    AlgorithmConfig algo_config;
    OP_REQUIRES_OK(context, GetAlgorithm(context, &algo_config));
    Status launch_status;
    {
      mutex_lock l(mu_);
      RnnDescriptor* rnn_desc_ptr = nullptr;
      OP_REQUIRES_OK(
          context, GetCachedRnnDescriptor<T>(context, model_shapes, input_mode,
                                             algo_config, &rnn_state_cache_,
                                             &rnn_desc_ptr, use_padded_io));
      launch_status = DoBackward<T>(
          context, *rnn_desc_ptr, model_types(), model_shapes, input, input_h,
          input_c, params, output, output_h, output_c, output_backprop,
          output_h_backprop, output_c_backprop, reserve_space, input_backprop,
          input_h_backprop, input_c_backprop, params_backprop, sequence_lengths,
          time_major, &workspace_allocator,
          /*output_profile_result=*/nullptr);
    }
    OP_REQUIRES_OK(context, launch_status);
  }

 protected:
  virtual Status GetAlgorithm(OpKernelContext* context,
                              AlgorithmConfig* algo_config) {
    CHECK_NE(algo_config, nullptr);
    *algo_config = AlgorithmConfig();
    return OkStatus();
  }

 private:
  mutex mu_;
  RnnStateCache rnn_state_cache_ TF_GUARDED_BY(mu_);

  Status ExtractBackwardInputs(
      OpKernelContext* context, const CudnnRnnModelShapes& model_shapes,
      const CudnnModelTypes& model_types, const Tensor** output,
      const Tensor** output_h, const Tensor** output_c,
      const Tensor** output_backprop, const Tensor** output_h_backprop,
      const Tensor** output_c_backprop, const Tensor** reserve_space) {
    TF_RETURN_IF_ERROR(context->input("output", output));
    TF_RETURN_IF_ERROR(context->input("output_backprop", output_backprop));
    TF_RETURN_IF_ERROR(context->input("output_h", output_h));
    TF_RETURN_IF_ERROR(context->input("output_h_backprop", output_h_backprop));
    if (model_types.HasInputC()) {
      TF_RETURN_IF_ERROR(context->input("output_c", output_c));
      TF_RETURN_IF_ERROR(
          context->input("output_c_backprop", output_c_backprop));
    }
    TF_RETURN_IF_ERROR(context->input("reserve_space", reserve_space));
    const TensorShape& hidden_state_shape = model_shapes.hidden_state_shape;
    const TensorShape& output_shape = model_shapes.output_shape;
    const TensorShape& cell_state_shape = model_shapes.cell_state_shape;

    if (output_shape != (*output)->shape()) {
      return errors::InvalidArgument(
          "Invalid output shape: ", (*output)->shape().DebugString(), " ",
          output_shape.DebugString());
    }
    if (hidden_state_shape != (*output_h)->shape()) {
      return errors::InvalidArgument(
          "Invalid output_h shape: ", (*output_h)->shape().DebugString(), " ",
          hidden_state_shape.DebugString());
    }

    if (output_shape != (*output_backprop)->shape()) {
      return errors::InvalidArgument("Invalid output_backprop shape: ",
                                     (*output_backprop)->shape().DebugString(),
                                     " ", output_shape.DebugString());
    }
    if (hidden_state_shape != (*output_h_backprop)->shape()) {
      return errors::InvalidArgument(
          "Invalid output_h_backprop shape: ",
          (*output_h_backprop)->shape().DebugString(), " ",
          hidden_state_shape.DebugString());
    }

    if (model_types.HasInputC()) {
      if (cell_state_shape != (*output_c)->shape()) {
        return errors::InvalidArgument(
            "Invalid output_c shape: ", (*output_c)->shape().DebugString(), " ",
            cell_state_shape.DebugString());
      }
      if (cell_state_shape != (*output_c_backprop)->shape()) {
        return errors::InvalidArgument(
            "Invalid output_c_backprop shape: ",
            (*output_c_backprop)->shape().DebugString(), " ",
            cell_state_shape.DebugString());
      }
    }
    return OkStatus();
  }

  Status AllocateOutputs(OpKernelContext* context,
                         const CudnnRnnModelShapes& model_shapes,
                         const TensorShape& params_shape,
                         Tensor** input_backprop, Tensor** input_h_backprop,
                         Tensor** input_c_backprop, Tensor** params_backprop) {
    const TensorShape& input_shape = model_shapes.input_shape;
    const TensorShape& hidden_state_shape = model_shapes.hidden_state_shape;
    const TensorShape& cell_state_shape = model_shapes.cell_state_shape;

    TF_RETURN_IF_ERROR(
        context->allocate_output(0, input_shape, input_backprop));
    TF_RETURN_IF_ERROR(
        context->allocate_output(1, hidden_state_shape, input_h_backprop));
    if (HasInputC()) {
      TF_RETURN_IF_ERROR(
          context->allocate_output(2, cell_state_shape, input_c_backprop));
    } else {
      // Only LSTM uses input_c and output_c. So for all other models, we only
      // need to create dummy outputs.
      TF_RETURN_IF_ERROR(context->allocate_output(2, {}, input_c_backprop));
    }
    TF_RETURN_IF_ERROR(
        context->allocate_output(3, params_shape, params_backprop));
    return OkStatus();
  }
};

#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CudnnRNNBackprop").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CudnnRNNBackwardOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

template <typename T>
class CudnnRNNBackwardOpV2<GPUDevice, T>
    : public CudnnRNNBackwardOp<GPUDevice, T> {
 public:
  explicit CudnnRNNBackwardOpV2(OpKernelConstruction* context)
      : CudnnRNNBackwardOp<GPUDevice, T>(context) {}

 protected:
  Status GetAlgorithm(OpKernelContext* context,
                      AlgorithmConfig* algo_config) override {
    CHECK_NE(algo_config, nullptr);
    const Tensor* host_reserved = nullptr;
    TF_RETURN_IF_ERROR(context->input("host_reserved", &host_reserved));

    auto host_reserved_int8 = host_reserved->vec<int8>();
    const AlgorithmDesc algo_desc(host_reserved_int8(0), host_reserved_int8(1),
                                  absl::nullopt);
    algo_config->set_algorithm(algo_desc);
    return OkStatus();
  }
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNBackpropV2")       \
                              .Device(DEVICE_GPU)          \
                              .HostMemory("host_reserved") \
                              .TypeConstraint<T>("T"),     \
                          CudnnRNNBackwardOpV2<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

template <typename T>
class CudnnRNNBackwardOpV3<GPUDevice, T>
    : public CudnnRNNBackwardOp<GPUDevice, T> {
 private:
  bool time_major_;

 protected:
  bool time_major() { return time_major_; }

 public:
  explicit CudnnRNNBackwardOpV3(OpKernelConstruction* context)
      : CudnnRNNBackwardOp<GPUDevice, T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("time_major", &time_major_));
    if (context->HasAttr("num_proj")) {
      OP_REQUIRES_OK(context, context->GetAttr("num_proj", &num_proj_));
    } else {
      num_proj_ = 0;
    }
  }

  void Compute(OpKernelContext* context) override {
    CudnnRNNBackwardOp<GPUDevice, T>::ComputeImpl(context, true, time_major(),
                                                  num_proj_);
  }

 private:
  int num_proj_;
};

#define REGISTER_GPU(T)                                       \
  REGISTER_KERNEL_BUILDER(Name("CudnnRNNBackpropV3")          \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("sequence_lengths") \
                              .HostMemory("host_reserved")    \
                              .TypeConstraint<T>("T"),        \
                          CudnnRNNBackwardOpV3<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// TODO(zhengxq): Add the conversion of Cudnn RNN Params from and to
// its canonical form.

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
