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

#ifndef TENSORFLOW_CORE_KERNELS_SPACETODEPTH_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPACETODEPTH_OP_H_
// Functor definition for XentOp, must be compilable by nvcc.

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

// Functor used by SpaceToDepthOp to do the computations.
// Implements a family of Space to Depth transforms for a 4D 'input' tensor
// to a 4D 'output' tensor, both tensors use type 'T' and layout 'data_format'.
// These transforms divide the vertical and horizontal image sizes by
// 'block_size', and multiply the depth dimension size by
// (block_size * block_size). The offset within each block_size * block_size
// patch within the image is combined with the input channel index to form
// the output channel index, with the Y, X coordinates within each block of
// the input image used as the high order component of the output channel.
// e.g. for data_format = NHWC:
//      Each element in the input tensor can be specified via 6 coordinates,
//      ordered by decreasing memory layout significance as:
//      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
//                         within the output image, bX, bY means coordinates
//                         within the input block, iC means input channels).
//      The output would be a transpose to the following layout:
//      n,oY,oX,bY,bX,iC
template <typename Device, typename T, TensorFormat data_format>
struct SpaceToDepthOpFunctor {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output);

  // This 5-D version is to support NCHW_VECT_C.
  void operator()(const Device& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPACETODEPTH_OP_H_
