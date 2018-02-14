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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/util/padding.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace tensorflow {

template <typename Device, typename T>
struct LaunchUnpool;

template <typename T>
struct LaunchUnpool<CPUDevice,T>
{
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;

  static void launch(OpKernelContext* context, const Tensor& pooled_data, const Tensor& indices, Tensor* unpooled_data)
  {
    bool status = true;

    const DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    auto shard = [&pooled_data, &indices, &unpooled_data](int64 start, int64 limit)
    {
      const int64 batch_size = GetTensorDim(pooled_data.shape(), FORMAT_NHWC, 'N');
      const int64 numPooledPoints = pooled_data.shape().num_elements();
      const int64 num_pooled_points_per_batch = pooled_data.shape().num_elements()/batch_size;
      const int64 num_unpooledpoints_per_batch = unpooled_data->shape().num_elements()/batch_size;

      {
        const int64 output_start = start*num_unpooledpoints_per_batch;
        const int64 output_end = limit*num_unpooledpoints_per_batch;
        EigenMatrixMap unpooled_data_shard(unpooled_data->flat<T>().data()+output_start, 1, output_end-output_start);
        unpooled_data_shard.setConstant(T(0));

        auto pooled_dataFlat = pooled_data.flat<T>();
        auto unpooled_dataFlat = unpooled_data->flat<T>();
        auto indices_flat = indices.flat<int64>();
        for (int64 batch=start; batch<limit; batch++) {
          for (int64 index=0; index<num_pooled_points_per_batch; index++) {
            const int64 pooled_index = batch*num_pooled_points_per_batch+index;
            const int64 unpooled_index = indices_flat(pooled_index);
            CHECK(pooled_index<numPooledPoints) << "Invalid pooled index: " << pooled_index << ", total pooled points: " << numPooledPoints;
            unpooled_dataFlat(unpooled_index) = pooled_dataFlat(pooled_index);
          }
        }
      }
    };

    const int batch_size = GetTensorDim(pooled_data.shape(), FORMAT_NHWC, 'N');
    const int64 shard_cost = unpooled_data->shape().num_elements();
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, shard_cost, shard);

    if (!status) {
      context->SetStatus(errors::Internal("Failed launching Unpool on CPU"));
    }
  }
};

template <typename Device, typename T>
struct LaunchUnpoolGradient;

template <typename T>
struct LaunchUnpoolGradient<CPUDevice,T>
{
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;

  static void launch(tensorflow::OpKernelContext* context, const tensorflow::Tensor& unpooled_gradient, const tensorflow::Tensor& indices, tensorflow::Tensor* pooled_gradient)
  {
    bool status = true;

    const tensorflow::DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    auto shard = [&unpooled_gradient, &indices, &pooled_gradient](tensorflow::int64 start, tensorflow::int64 limit)
    {
      const tensorflow::int64 batch_size = tensorflow::GetTensorDim(unpooled_gradient.shape(), tensorflow::FORMAT_NHWC, 'N');
      const tensorflow::int64 num_pooled_points_per_batch = pooled_gradient->NumElements()/batch_size;

      {
        auto pooled_gradient_flat = pooled_gradient->flat<T>();
        auto unpooled_gradient_flat = unpooled_gradient.flat<T>();
        auto indices_flat = indices.flat<tensorflow::int64>();

        const tensorflow::int64 pooled_start = start*num_pooled_points_per_batch;
        const tensorflow::int64 pooled_end = limit*num_pooled_points_per_batch;
        EigenMatrixMap pooled_gradient_shard(pooled_gradient_flat.data()+pooled_start, 1, pooled_end-pooled_start);
        pooled_gradient_shard.setConstant(T(0));

        for (tensorflow::int64 batch=start; batch<limit; batch++) {
          for (tensorflow::int64 batch_pooled_index=0; batch_pooled_index<num_pooled_points_per_batch; batch_pooled_index++) {
            const tensorflow::int64 pooled_index = batch*num_pooled_points_per_batch + batch_pooled_index;
            CHECK(pooled_index<batch_size*num_pooled_points_per_batch) << "pooled index out of range: " << pooled_index << ">=" << batch_size*num_pooled_points_per_batch;
            const tensorflow::int64 unpooled_index = indices_flat(pooled_index);
            pooled_gradient_flat(pooled_index) += unpooled_gradient_flat(unpooled_index);
          }
        }
      }
    };

    const int batch_size = tensorflow::GetTensorDim(unpooled_gradient.shape(), tensorflow::FORMAT_NHWC, 'N');
    const tensorflow::int64 shard_cost = unpooled_gradient.shape().num_elements();
    tensorflow::Shard(worker_threads.num_threads, worker_threads.workers, batch_size, shard_cost, shard);

    if (!status) {
      context->SetStatus(tensorflow::errors::Internal("Failed launching Unpool on CPU"));
    }
  }
};

template <typename Device, typename T>
struct UnpoolOp : public OpKernel
{
public:
  explicit UnpoolOp(OpKernelConstruction* context) : OpKernel(context)
  {}

  void Compute(OpKernelContext* context) override
  {
    const Tensor& pooled_data = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& unpool_shape_tensor = context->input(2);

    if (!context->status().ok()) {
      return;
    }

    TensorShape unpool_shape;
    switch (unpool_shape_tensor.dtype()) {
      case DT_INT32:
        {
          auto unpool_shape_vector = unpool_shape_tensor.flat<int32>();
          Status status = TensorShapeUtils::MakeShape(unpool_shape_vector.data(), unpool_shape_vector.size(), &unpool_shape);
          if (!status.ok()) {
            context->SetStatus(errors::Internal("Failed getting unpool shape"));
          }
        }
        break;
      case DT_INT64:
        {
          auto unpool_shape_vector = unpool_shape_tensor.flat<int64>();
          Status status = TensorShapeUtils::MakeShape(unpool_shape_vector.data(), unpool_shape_vector.size(), &unpool_shape);
          if (!status.ok()) {
            context->SetStatus(errors::Internal("Failed getting unpool shape"));
          }
        }
        break;
      default:
        return;
    }

    Tensor* unpooled_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, unpool_shape, &unpooled_data));

    LaunchUnpool<Device,T>::launch(context, pooled_data, indices, unpooled_data);
  }
private:
  std::vector<int32> m_unpool_shape;
};

template <typename Device, typename T>
struct UnpoolGradientOp : public tensorflow::OpKernel
{
public:
  explicit UnpoolGradientOp(tensorflow::OpKernelConstruction* context) :
    tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override
  {
    const tensorflow::Tensor& unpooled_gradient = context->input(0);
    const tensorflow::Tensor& indices = context->input(1);

    if (!context->status().ok()) {
      return;
    }

    tensorflow::TensorShape pooled_shape = indices.shape();
    tensorflow::Tensor* pooled_gradient = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, pooled_shape, &pooled_gradient));

    LaunchUnpoolGradient<Device,T>::launch(context, unpooled_gradient, indices, pooled_gradient);
  }
};

REGISTER_KERNEL_BUILDER(Name("Unpool").Device(tensorflow::DEVICE_CPU), UnpoolOp<CPUDevice, float>)
REGISTER_KERNEL_BUILDER(Name("UnpoolGradient").Device(tensorflow::DEVICE_CPU), UnpoolGradientOp<CPUDevice, float>)

}
