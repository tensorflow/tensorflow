/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/bias_op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace {

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width, int32* depth,
                      int32* channel) {
  *batch = 1;
  *height = 1;
  *width = 1;
  *depth = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    *batch = static_cast<int32>(value_tensor.dim_size(0));
    *channel = static_cast<int32>(value_tensor.dim_size(1));
    *height = static_cast<int32>(value_tensor.dim_size(2));
    if (value_tensor.dims() > 3) {
      *width = static_cast<int32>(value_tensor.dim_size(3));
    }
    if (value_tensor.dims() > 4) {
      *depth = static_cast<int32>(value_tensor.dim_size(4));
    }
  }
}

template <class T>
struct AccumulatorType {
  typedef T type;
};

// float is faster on the CPU than half, and also more precise,
// so use float for the temporary accumulators.
template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

}  // namespace

template <typename Device, typename T>
class BiasOp : public BinaryOp<T> {
 public:
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));

    // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
    size_t channel_dim;
    if (data_format_ == FORMAT_NCHW) {
      channel_dim = 1;  // NCHW always have channel dim in 1 (with 3, 4, 5
                        // dimensions data).
    } else {
      channel_dim = input.shape().dims() - 1;  // End of code by intel_tf.
    }

    OP_REQUIRES(
        context,
        bias.shape().dim_size(0) == input.shape().dim_size(channel_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
    if (data_format_ == FORMAT_NCHW) {
      int32 batch, height, width, depth, channel;
      GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                       &channel);
      switch (input.shape().dims()) {
        case 3: {
          Eigen::DSizes<int32, 3> three_dims(1, channel, 1);
          Eigen::DSizes<int32, 3> broad_cast_dims(batch, 1, height);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 3>().device(d) = input.tensor<T, 3>() +
                                             bias.tensor<T, 1>()
                                                 .reshape(three_dims)
                                                 .broadcast(broad_cast_dims);
        } break;
        case 4: {
          Eigen::DSizes<int32, 4> four_dims(1, channel, 1, 1);
          Eigen::DSizes<int32, 4> broad_cast_dims(batch, 1, height, width);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 4>().device(d) =
              input.tensor<T, 4>() +
              bias.tensor<T, 1>().reshape(four_dims).broadcast(broad_cast_dims);
        } break;
        case 5: {
          Eigen::DSizes<int32, 5> five_dims(1, channel, 1, 1, 1);
          Eigen::DSizes<int32, 5> broad_cast_dims(batch, 1, height, width,
                                                  depth);
          const Device& d = context->eigen_device<Device>();
          output->tensor<T, 5>().device(d) =
              input.tensor<T, 5>() +
              bias.tensor<T, 1>().reshape(five_dims).broadcast(broad_cast_dims);
        } break;
        default:
          OP_REQUIRES(context, false,
                      errors::InvalidArgument("Only ranks up to 5 supported: ",
                                              input.shape().DebugString()));
      }
      return;
    }  // End of code by intel_tf.

    switch (input.shape().dims()) {
      case 2:
        Compute<2>(context, input, bias, output);
        break;
      case 3:
        Compute<3>(context, input, bias, output);
        break;
      case 4:
        Compute<4>(context, input, bias, output);
        break;
      case 5:
        Compute<5>(context, input, bias, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 5 supported: ",
                                            input.shape().DebugString()));
    }
  }

  // Add biases for an input matrix of rank Dims, by using the Bias.
  template <int Dims>
  void Compute(OpKernelContext* ctx, const Tensor& input, const Tensor& bias,
               Tensor* output) {
    functor::Bias<Device, T, Dims> functor;
    functor(ctx->eigen_device<Device>(), input.tensor<T, Dims>(), bias.vec<T>(),
            output->tensor<T, Dims>());
  }

 private:
  TensorFormat data_format_;
};

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      BiasOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BiasAdd").Device(DEVICE_SYCL).TypeConstraint<type>("T"),   \
      BiasOp<SYCLDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BiasAddV1").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      BiasOp<SYCLDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNEL);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class BiasGradOp : public OpKernel {
 public:
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));

    OP_REQUIRES(
        context, FastBoundsCheck(output_backprop.NumElements(),
                                 std::numeric_limits<int32>::max()),
        errors::InvalidArgument("BiasGrad requires tensor size <= int32 max"));

    int32 batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    if (channel == 0) {
      return;  // Nothing to do
    } else if (output_backprop.NumElements() == 0) {
      // Eigen often crashes by design on empty tensors, but setZero is safe
      output->template flat<T>().setZero();
    } else {
      //******************************************************************************
      //  Divide the input tensor into several blocks.
      //  As we don't know anything about the detail shape of incoming input
      //  tensor, and it will be too complex to deal with different shapes
      //  separately, so we just evenly distribute total workloads to each
      //  block.
      //******************************************************************************
      // Init the output to zero
      output->template flat<T>().setZero();
      // Get the location of input/output data.
      const T* input_ptr = output_backprop.template flat<T>().data();

      // Get the format of input/output data.
      const bool format_is_nhwc = (data_format_ == FORMAT_NHWC);

      // Get the intra-thread pool handle.
      auto worker_threads =
          *(context->device()->tensorflow_cpu_worker_threads());
      const int num_threads = worker_threads.num_threads;
      auto workers = worker_threads.workers;

      // Get the workload parameters.
      const int reduce_dims = batch * height * width * depth;
      const int hwd_size = height * width * depth;
      const int total_workload = reduce_dims * channel;

      // For small workloads, large block number will waste
      // scheduling and compute resources.
      // Use minimum workload and max_parallelism to limit total
      // thread number and guarantee the workload for a each thread.
      // Roughly, persume the CPU is 2GHz, using 1ns as a block,
      // then each block gets about 2000 FLOP.
      const int min_block_workloads = 2000;
      // For NHWC format, we use each channel layer as a scheduling unit,
      // while for NCHW format, we use each FLOP as a scheduling unit.
      int parallel_cell_size = 1;
      if ((format_is_nhwc) || ((!format_is_nhwc) && (hwd_size == 1))) {
        parallel_cell_size = channel;
      }
      const int max_parallelism = total_workload / parallel_cell_size;
      const int min_block_size =
          (min_block_workloads + parallel_cell_size - 1) / parallel_cell_size;
      const int max_num_blocks =
          std::min(max_parallelism,
                   (total_workload + min_block_size - 1) / min_block_size);
      // As the BiasAddGradOp is a reducing op,
      // it is necessary to build buffer for each block to avoid hazard.
      // To minimize the buffer, the block number is no more than thread number.
      const int num_blocks = std::min(max_num_blocks, num_threads);

      // Build&initialize buffers for blocks.
      TensorShape output_buffer_shape({num_blocks, channel});
      Tensor block_results_buffer(output->dtype(), output_buffer_shape);
      block_results_buffer.template flat<T>().setZero();
      T* block_results_buffer_ptr =
          block_results_buffer.template flat<T>().data();

      using Shell = Eigen::TensorMap<Eigen::Tensor<T, 1>>;
      using ConstShell = Eigen::TensorMap<Eigen::Tensor<const T, 1>>;

      //******************************************************************************
      //  Job func for each thread
      //******************************************************************************
      auto BiasGradWorker = [this, &total_workload, &num_blocks,
                             &format_is_nhwc, &input_ptr,
                             &block_results_buffer_ptr, &hwd_size, &channel](
          int my_job_begin, int my_job_end) mutable -> void {
        // We generate a cover of [0,total_workload), which is comprised of
        // num_blocks non-overlapping divisions of [0,total_workload)
        // EXP: If we get 22 elements in input tensor, which are divided
        // into 4 blocks:
        //
        // lockId  :   0   |   1   |   2   |   3   | res
        // Elements: $$$$$ | $$$$$ | $$$$$ | $$$$$ | **
        //                        ↓
        // BlockId :   0   |   1    |    2    |    3
        // Elements: $$$$$ | $$$$$* |  $$$$$  |  $$$$$*
        // Range   : [0,5) | [5,11) | [11,16) | [16,22)
        //     22*0/4=0 22*1/4=5 22*2/4=11 22*3/4=16 22*4/4=22
        const int block_begin = total_workload * my_job_begin / num_blocks;
        const int block_end = total_workload * my_job_end / num_blocks;

        T* buffer_ptr = &block_results_buffer_ptr[my_job_begin * channel];
        Shell my_buffer(buffer_ptr, channel);

        if ((format_is_nhwc) || ((!format_is_nhwc) && (hwd_size == 1))) {
          // For NHWC, it is easy to divide workload, because the parallelism
          // mainly comes from layers outside channel (N*H*W).
          // So we just divide NHW layers.
          // Align the calculation by inner most layer (channel).
          const int align_begin = (block_begin / channel) * channel;
          const int align_end = (block_end / channel) * channel;
          // Apply the calculation.
          for (int i = align_begin; i < align_end; i += channel) {
            my_buffer += ConstShell(&input_ptr[i], channel);
          }
        } else {  // For NCHW format
          // A straight forward impl for NCHW could be like:
          //  for(int i=block_begin;i<block_end;i++) {
          //    my_buffer_ptr[(i/hwd_size)%channel] +=
          //    input_ptr[i];
          //  }
          // It is more complex than NHWC for there are calculations
          // both inside and outside channel layer.
          // There are two extreme situations:
          //   1. N is large and H*W is small;
          //   2. H*W is large and N is small.
          // The first one is more similar to NHWC, which easy to divid
          // workload. While for the second situation, the workload could
          // be heavy, because of the large H*W, while there is not enough
          // dimensions outside channel to divide (small N).
          // We divide the workload basing on total.
          const int align_begin =
              ((block_begin + hwd_size - 1) / hwd_size) * hwd_size;
          const int align_end = (block_end / hwd_size) * hwd_size;

          // Dealing with front residual.
          int channel_id = block_begin / hwd_size % channel;
          Eigen::Tensor<T, 0> sum =
              ConstShell(&input_ptr[block_begin], align_begin - block_begin)
                  .sum();
          my_buffer(channel_id) += sum(0);

          // Init channel_id to avoid the error when align_begin == block_begin.
          channel_id = align_begin / hwd_size % channel;

          for (int i = align_begin; i < align_end; i += hwd_size) {
            // Apply the reduction
            if (channel_id < channel) {
              // When channel_id is in channel,
              // just add the sum of inside dim to block buffer.
              sum = ConstShell(&input_ptr[i], hwd_size).sum();
              my_buffer(channel_id) += sum(0);
              channel_id++;
            } else {
              // When channel_id exceed the range of channel,
              // go back to the beginning of block buffer.
              channel_id = channel_id - channel;
              sum = ConstShell(&input_ptr[i], hwd_size).sum();
              my_buffer(channel_id) += sum(0);
              channel_id++;
            }
          }
          // Dealing with back residual.
          sum = ConstShell(&input_ptr[align_end], block_end - align_end).sum();
          my_buffer(channel_id) += sum(0);
        }
      };
      // Run multi-threads
      // We use Sharder::Do here to make sure each block matches one thread
      const int total_units = num_blocks;
      // As we have pretreated workload,
      // set cost_per_unit to 10000 to override the defalt.
      const int cost_per_unit = 10000;
      Sharder::Do(
          total_units, cost_per_unit, BiasGradWorker,
          [&workers](Sharder::Closure c) -> void { workers->Schedule(c); },
          max_parallelism);

      //******************************************************************************
      //  Now sum block results up
      //******************************************************************************
      for (int i = 0; i < num_blocks; i++) {
        Shell buffer_i(&block_results_buffer_ptr[channel * i], channel);
        output->template flat<T>() += buffer_i;
      }
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasGradOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BiasAddGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      BiasGradOp<SYCLDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNEL);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
template <typename T>
class BiasOp<GPUDevice, T> : public BinaryOp<T> {
 public:
  typedef GPUDevice Device;
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));
    int32 batch, height, width, depth, channel;
    GetBiasValueDims(input, data_format_, &batch, &height, &width, &depth,
                     &channel);
    OP_REQUIRES(context, bias.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel dimension "
                    "of the input tensor: ",
                    bias.shape().DebugString(), " vs. ", channel, " in ",
                    input.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (input.NumElements() > 0) {
      BiasGPU<T>::compute(context->template eigen_device<Device>(),
                          input.flat<T>().data(), bias.flat<T>().data(),
                          output->flat<T>().data(), batch, width, height, depth,
                          channel, data_format_);
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      BiasOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

struct BiasGradAutotuneGroup {
  static string name() { return "BiasGrad"; }
};

class BiasAddGradGPUConfig {
 public:
  BiasAddGradGPUConfig() : mode_(BiasAddGradGPUMode::kReduction) {}
  string ToString() const {
    if (mode_ == BiasAddGradGPUMode::kNative) {
      return "native CUDA kernel.";
    }
    if (mode_ == BiasAddGradGPUMode::kReduction) {
      return "cub reduction kernel.";
    }
    return "unknown kernel.";
  }
  BiasAddGradGPUMode get_mode() const { return mode_; }
  void set_mode(BiasAddGradGPUMode val) { mode_ = val; }

  bool operator==(const BiasAddGradGPUConfig& other) const {
    return this->mode_ == other.get_mode();
  }

  bool operator!=(const BiasAddGradGPUConfig& other) const {
    return !(*this == other);
  }

 private:
  BiasAddGradGPUMode mode_;
};

// Encapsulate all the shape information that is used in bias add grad
// operations.
class BiasAddParams {
 public:
  // We use a list to maintain both the shape value and the order (data format).
  using SpatialArray = gtl::InlinedVector<int64, 4>;
  BiasAddParams(const SpatialArray& in_shape, TensorFormat data_format,
                DataType dtype, int device_id)
      : in_shape_(in_shape),
        data_format_(data_format),
        dtype_(dtype),
        device_id_(device_id) {
    for (int64 val : in_shape_) {
      hash_code_ = Hash64Combine(hash_code_, val);
    }
    hash_code_ = Hash64Combine(hash_code_, data_format);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const BiasAddParams& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const BiasAddParams& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    // clang-format off
    return strings::StrCat(
        "(", str_util::Join(in_shape_, ", "), "), ",
        data_format_, ", ", dtype_, ", ", device_id_);
    // clang-format on
  }

 protected:
  using ParamsDataType = std::tuple<SpatialArray, TensorFormat, DataType, int>;

  ParamsDataType get_data_as_tuple() const {
    return std::make_tuple(in_shape_, data_format_, dtype_, device_id_);
  }

  uint64 hash_code_ = 0;

 private:
  SpatialArray in_shape_;
  TensorFormat data_format_;
  DataType dtype_;
  int device_id_;
};

typedef AutoTuneSingleton<BiasGradAutotuneGroup, BiasAddParams,
                          BiasAddGradGPUConfig>
    AutotuneBiasGrad;

template <typename T>
class BiasGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NCHW;
    }
  }

  void ComputeWithCustomKernel(OpKernelContext* context,
                               const Tensor& output_backprop, int32 batch,
                               int32 width, int32 height, int32 depth,
                               int32 channel, Tensor* output) {
    BiasGradGPU<T>::compute(context->template eigen_device<Device>(),
                            output_backprop.template flat<T>().data(),
                            output->flat<T>().data(), batch, width, height,
                            depth, channel, data_format_);
  }

  void ComputeWithReduceSum(OpKernelContext* context,
                            const Tensor& output_backprop, int32 batch,
                            int32 width, int32 height, int32 depth,
                            int32 channel, Tensor* output) {
    if (data_format_ == FORMAT_NCHW) {
      int32 row_count = batch * channel;
      int32 col_count = height * width * depth;
      Tensor temp_grad_outputs;
      // For 'NCHW' format, we perform reduction twice: first HW, then N.
      TensorShape temp_grad_output_shape{row_count, col_count};
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                     temp_grad_output_shape,
                                                     &temp_grad_outputs));
      BiasGradGPU<T>::DoRowReduction(
          context, temp_grad_outputs.flat<T>().data(),
          output_backprop.template flat<T>().data(), row_count, col_count);

      row_count = batch;
      col_count = channel;
      BiasGradGPU<T>::DoColReduction(context, output->flat<T>().data(),
                                     temp_grad_outputs.flat<T>().data(),
                                     row_count, col_count);
    } else {
      // For 'NHWC', we simply apply reduction once on NHW.
      int32 row_count = batch * height * width * depth;
      int32 col_count = channel;
      BiasGradGPU<T>::DoColReduction(
          context, const_cast<T*>(output->flat<T>().data()),
          reinterpret_cast<const T*>(output_backprop.template flat<T>().data()),
          row_count, col_count);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));
    int32 batch, height, width, depth, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &depth, &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (channel == 0) return;
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    se::DeviceMemoryBase output_ptr(output->flat<T>().data(),
                                    output->NumElements() * sizeof(T));
    stream->ThenMemZero(&output_ptr, output->NumElements() * sizeof(T));
    if (output_backprop.NumElements() <= 0) return;

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = output_backprop.dtype();
    BiasAddParams bias_parameters = {
        {batch, height * width * depth, channel},
        data_format_,
        dtype,
        device_id,
    };

    // Autotune two algorithm: customized
    BiasAddGradGPUConfig algo_config;
    if (!AutotuneBiasGrad::GetInstance()->Find(bias_parameters, &algo_config)) {
      BiasGradGPUProfileResult best_result;
      // Initialize the timer.
      perftools::gputools::Timer timer(stream->parent());
      stream->InitTimer(&timer);
      stream->ThenStartTimer(&timer);
      ComputeWithCustomKernel(context, output_backprop, batch, width, height,
                              depth, channel, output);
      stream->ThenStopTimer(&timer);
      uint64 elapsed_microseconds = timer.Microseconds();
      VLOG(1) << "BiasAddGrad " << bias_parameters.ToString()
              << " Native algo latency: " << elapsed_microseconds;
      if (elapsed_microseconds < best_result.elapsed_time()) {
        best_result.set_algorithm(BiasAddGradGPUMode::kNative);
        best_result.set_elapsed_time(elapsed_microseconds);
      }

      // Try reduction and profile.
      stream->ThenStartTimer(&timer);
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
      stream->ThenStopTimer(&timer);

      elapsed_microseconds = timer.Microseconds();
      VLOG(1) << "BiasAddGrad " << bias_parameters.ToString()
              << " Reduction algo latency: " << elapsed_microseconds;
      if (elapsed_microseconds < best_result.elapsed_time()) {
        best_result.set_algorithm(BiasAddGradGPUMode::kReduction);
        best_result.set_elapsed_time(elapsed_microseconds);
      }

      algo_config.set_mode(best_result.algorithm());
      AutotuneBiasGrad::GetInstance()->Insert(bias_parameters, algo_config);

      // Results are already available during autotune, so no need to continue.
      return;
    }

    // Choose the best algorithm based on autotune results.
    if (algo_config.get_mode() == BiasAddGradGPUMode::kReduction) {
      ComputeWithReduceSum(context, output_backprop, batch, width, height,
                           depth, channel, output);
    } else {
      // Default to the customized kernel.
      ComputeWithCustomKernel(context, output_backprop, batch, width, height,
                              depth, channel, output);
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
