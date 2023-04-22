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

#ifndef TENSORFLOW_KERNELS_SCATTER_ND_OP_H_
#define TENSORFLOW_KERNELS_SCATTER_ND_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class OpKernelContext;

namespace scatter_nd_op {

enum class UpdateOp { ASSIGN, ADD, SUB, MIN, MAX };

}  // namespace scatter_nd_op

namespace functor {

// Functor used by ScatterOp to do the computations.
template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdFunctor {
  // Returns -1 on success or a nonnegative i s.t. indices[i] is a bad index.
  Index operator()(
      const Device& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput);
};

// Scatter updates into indices in Tensor out.  The argument allocate
// controls whether 'out' should be created.  If allocate is true,
// *out will be updated to the scattered tensor upon successful completion.
// If allocate is false, out must point to a Tensor allocated with the
// right type (T) and shape.  This tensor will not be zeroed out
// before the scatter is executed.
template <typename Device, typename T, typename Index,
          scatter_nd_op::UpdateOp Op>
Status DoScatterNd(OpKernelContext* c, const Tensor& indices,
                   const Tensor& updates, const TensorShape& shape, Tensor* out,
                   bool allocate);

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SCATTER_ND_OP_H_
