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

// See docs for ImageConnectedComponents in ../ops/image_ops.cc, and description
// of the algorithm in segmentation_ops.h.

#define EIGEN_USE_THREADS

#include "tensorflow/contrib/image/kernels/segmentation_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using tensorflow::functor::BlockedImageUnionFindFunctor;
using tensorflow::functor::FindRootFunctor;
using tensorflow::functor::ImageConnectedComponentsFunctor;
using tensorflow::functor::TensorRangeFunctor;

using OutputType = typename BlockedImageUnionFindFunctor<bool>::OutputType;

// Computes connected components on batches of 2D images.
template <typename Device, typename T>
class ImageConnectedComponents : public OpKernel {
 public:
  explicit ImageConnectedComponents(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images_t = ctx->input(0);
    OP_REQUIRES(ctx, images_t.shape().dims() == 3,
                errors::InvalidArgument("Input images must have rank 3"));
    Tensor forest_t, rank_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_INT64,
                                           images_t.shape(), &forest_t));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_INT64,
                                           images_t.shape(), &rank_t));
    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, images_t.shape(), &output_t));

    // Fill forest with values from 0 to n - 1, so that each node points to
    // itself.
    TensorRangeFunctor<Device>()(ctx->eigen_device<Device>(),
                                 forest_t.flat<OutputType>());
    auto rank = rank_t.tensor<OutputType, 3>();
    rank.device(ctx->eigen_device<Device>()) = rank.constant(OutputType(0));

    const auto images = images_t.tensor<T, 3>();
    auto forest = forest_t.tensor<OutputType, 3>();
    ImageConnectedComponentsFunctor<Device, T>()(
        ctx, output_t->flat<OutputType>(), images, forest, rank);
  }
};

using CPUDevice = Eigen::ThreadPoolDevice;

namespace functor {

// Connected components CPU implementation. See `segmentation_ops.h` for a
// description of the algorithm.
template <typename T>
struct ImageConnectedComponentsFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<OutputType>::Flat output,
                  typename TTypes<T, 3>::ConstTensor images,
                  typename TTypes<OutputType, 3>::Tensor forest,
                  typename TTypes<OutputType, 3>::Tensor rank) {
    const int64 num_images = images.dimension(0),
                num_rows = images.dimension(1), num_cols = images.dimension(2),
                num_elements = images.size();
    // Bail out early for an empty image--no work to do.
    if (num_elements == 0) {
      return;
    }
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    BlockedImageUnionFindFunctor<T> union_find(
        images.data(), num_rows, num_cols, forest.data(), rank.data());
    while (union_find.can_merge()) {
      union_find.merge_blocks();
      int64 num_blocks_vertically = union_find.num_blocks_vertically();
      int64 num_blocks_horizontally = union_find.num_blocks_horizontally();
      // Merging each block calls union_down for each pixel in a row of the
      // block, and union_right for each pixel in a column of the block. Assume
      // 20 instructions for each call to union_down or union_right. find() may
      // loop more while searching for the root, but this should not be very
      // significant.
      int cost = (union_find.block_height() + union_find.block_width()) * 20;
      Shard(worker_threads->num_threads, worker_threads->workers,
            num_images * num_blocks_vertically * num_blocks_horizontally, cost,
            [&union_find, num_blocks_vertically, num_blocks_horizontally](
                int64 start_block, int64 limit_block) {
              for (int64 i = start_block; i < limit_block; i++) {
                int64 block_x = i % num_blocks_horizontally;
                int64 block_y =
                    (i / num_blocks_horizontally) % num_blocks_vertically;
                int64 image =
                    i / (num_blocks_horizontally * num_blocks_vertically);
                union_find.merge_internal_block_edges(image, block_y, block_x);
              }
            });
    }
    FindRootFunctor<CPUDevice, T>()(ctx->eigen_device<CPUDevice>(), output,
                                    images.data(), union_find);
  }
};

}  // end namespace functor

#define REGISTER_IMAGE_CONNECTED_COMPONENTS(TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("ImageConnectedComponents")    \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageConnectedComponents<CPUDevice, TYPE>)
// Connected components (arguably) make sense for number, bool, and string types
TF_CALL_NUMBER_TYPES(REGISTER_IMAGE_CONNECTED_COMPONENTS);
TF_CALL_bool(REGISTER_IMAGE_CONNECTED_COMPONENTS);
TF_CALL_string(REGISTER_IMAGE_CONNECTED_COMPONENTS);
#undef REGISTER_IMAGE_CONNECTED_COMPONENTS

// TODO(ringwalt): Implement on GPU. We probably want to stick to the original
// algorithm by Stava and Benes there for efficiency (computing small blocks in
// shared memory in CUDA thread blocks, instead of starting with single-pixel
// blocks).

}  // end namespace tensorflow
