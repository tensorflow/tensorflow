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

  static void launch(OpKernelContext* context, const Tensor& pooledData, const Tensor& indices, Tensor* unpooledData)
  {
    bool status = true;

    const DeviceBase::CpuWorkerThreads& workerThreads = *(context->device()->tensorflow_cpu_worker_threads());

    auto shard = [&pooledData, &indices, &unpooledData](int64 start, int64 limit)
    {
      const int64 batchSize = GetTensorDim(pooledData.shape(), FORMAT_NHWC, 'N');
      const int64 numPooledPoints = pooledData.shape().num_elements();
      const int64 numPooledPointsPerBatch = pooledData.shape().num_elements()/batchSize;
      const int64 numUnpooledPointsPerBatch = unpooledData->shape().num_elements()/batchSize;

      {
        const int64 outputStart = start*numUnpooledPointsPerBatch;
        const int64 outputEnd = limit*numUnpooledPointsPerBatch;
        EigenMatrixMap unpooledDataShard(unpooledData->flat<T>().data()+outputStart, 1, outputEnd-outputStart);
        unpooledDataShard.setConstant(T(0));

        auto pooledDataFlat = pooledData.flat<T>();
        auto unpooledDataFlat = unpooledData->flat<T>();
        auto indicesFlat = indices.flat<int64>();
        for (int64 batch=start; batch<limit; batch++) {
          for (int64 index=0; index<numPooledPointsPerBatch; index++) {
            const int64 pooledIndex = batch*numPooledPointsPerBatch+index;
            const int64 unpooledIndex = indicesFlat(pooledIndex);
            CHECK(pooledIndex<numPooledPoints) << "Invalid pooled index: " << pooledIndex << ", total pooled points: " << numPooledPoints;
            unpooledDataFlat(unpooledIndex) = pooledDataFlat(pooledIndex);
          }
        }
      }
    };

    const int batchSize = GetTensorDim(pooledData.shape(), FORMAT_NHWC, 'N');
    const int64 shardCost = unpooledData->shape().num_elements();
    Shard(workerThreads.num_threads, workerThreads.workers, batchSize, shardCost, shard);

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

  static void launch(tensorflow::OpKernelContext* context, const tensorflow::Tensor& unpooledGradient, const tensorflow::Tensor& indices, tensorflow::Tensor* pooledGradient)
  {
    bool status = true;

    const tensorflow::DeviceBase::CpuWorkerThreads& workerThreads = *(context->device()->tensorflow_cpu_worker_threads());

    auto shard = [&unpooledGradient, &indices, &pooledGradient](tensorflow::int64 start, tensorflow::int64 limit)
    {
      const tensorflow::int64 batchSize = tensorflow::GetTensorDim(unpooledGradient.shape(), tensorflow::FORMAT_NHWC, 'N');
      const tensorflow::int64 numPooledPointsPerBatch = pooledGradient->NumElements()/batchSize;

      {
        auto pooledGradientFlat = pooledGradient->flat<T>();
        auto unpooledGradientFlat = unpooledGradient.flat<T>();
        auto indicesFlat = indices.flat<tensorflow::int64>();

        const tensorflow::int64 pooledStart = start*numPooledPointsPerBatch;
        const tensorflow::int64 pooledEnd = limit*numPooledPointsPerBatch;
        EigenMatrixMap pooledGradientShard(pooledGradientFlat.data()+pooledStart, 1, pooledEnd-pooledStart);
        pooledGradientShard.setConstant(T(0));

        for (tensorflow::int64 batch=start; batch<limit; batch++) {
          for (tensorflow::int64 batchPooledIndex=0; batchPooledIndex<numPooledPointsPerBatch; batchPooledIndex++) {
            const tensorflow::int64 pooledIndex = batch*numPooledPointsPerBatch + batchPooledIndex;
            CHECK(pooledIndex<batchSize*numPooledPointsPerBatch) << "pooled index out of range: " << pooledIndex << ">=" << batchSize*numPooledPointsPerBatch;
            const tensorflow::int64 unpooledIndex = indicesFlat(pooledIndex);
            pooledGradientFlat(pooledIndex) += unpooledGradientFlat(unpooledIndex);
          }
        }
      }
    };

    const int batchSize = tensorflow::GetTensorDim(unpooledGradient.shape(), tensorflow::FORMAT_NHWC, 'N');
    const tensorflow::int64 shardCost = unpooledGradient.shape().num_elements();
    tensorflow::Shard(workerThreads.num_threads, workerThreads.workers, batchSize, shardCost, shard);

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
    const Tensor& pooledData = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& unpoolShapeTensor = context->input(2);

    if (!context->status().ok()) {
      return;
    }

    TensorShape unpoolShape;
    switch (unpoolShapeTensor.dtype()) {
      case DT_INT32:
        {
          auto unpoolShapeVector = unpoolShapeTensor.flat<int32>();
          Status status = TensorShapeUtils::MakeShape(unpoolShapeVector.data(), unpoolShapeVector.size(), &unpoolShape);
          if (!status.ok()) {
            context->SetStatus(errors::Internal("Failed getting unpool shape"));
          }
        }
        break;
      case DT_INT64:
        {
          auto unpoolShapeVector = unpoolShapeTensor.flat<int64>();
          Status status = TensorShapeUtils::MakeShape(unpoolShapeVector.data(), unpoolShapeVector.size(), &unpoolShape);
          if (!status.ok()) {
            context->SetStatus(errors::Internal("Failed getting unpool shape"));
          }
        }
        break;
      default:
        return;
    }

    Tensor* unpooledData = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, unpoolShape, &unpooledData));

    LaunchUnpool<Device,T>::launch(context, pooledData, indices, unpooledData);
  }
private:
  std::vector<int32> m_unpoolShape;
};

template <typename Device, typename T>
struct UnpoolGradientOp : public tensorflow::OpKernel
{
public:
  explicit UnpoolGradientOp(tensorflow::OpKernelConstruction* context) :
    tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override
  {
    const tensorflow::Tensor& unpooledGradient = context->input(0);
    const tensorflow::Tensor& indices = context->input(1);

    if (!context->status().ok()) {
      return;
    }

    tensorflow::TensorShape pooledShape = indices.shape();
    tensorflow::Tensor* pooledGradient = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, pooledShape, &pooledGradient));

    LaunchUnpoolGradient<Device,T>::launch(context, unpooledGradient, indices, pooledGradient);
  }
};

REGISTER_KERNEL_BUILDER(Name("Unpool").Device(tensorflow::DEVICE_CPU), UnpoolOp<CPUDevice, float>)
REGISTER_KERNEL_BUILDER(Name("UnpoolGradient").Device(tensorflow::DEVICE_CPU), UnpoolGradientOp<CPUDevice, float>)

}
