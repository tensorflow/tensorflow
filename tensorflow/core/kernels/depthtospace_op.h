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

#ifndef TENSORFLOW_CORE_KERNELS_DEPTHTOSPACE_OP_H_
#define TENSORFLOW_CORE_KERNELS_DEPTHTOSPACE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

// Functor used by DepthToSpaceOp to do the computations.
// Implements a family of Depth to Space transforms for a 4D 'input' tensor
// to a 4D 'output' tensor, both tensors use type 'T' and layout 'data_format'.
// These transforms multiply the vertical and horizontal image sizes by
// 'block_size', and divide the depth dimension by (block_size * block_size)
// which must divide evenly.
// Each pixel in the input image is converted to a square block of pixels in
// the output image. The Y, X coordinates within each block comes from the
// high component of the input depth (channel) index.
// e.g. for data_format = NHWC:
//      Each element in the input tensor can be specified via 6 coordinates,
//      ordered by decreasing memory layout significance as:
//      n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
//                         within the input image, bX, bY means coordinates
//                         within the output block, oC means output channel).
//      The output would be a transpose to the following layout:
//      n,iY,bY,iX,bX,oC
template <typename Device, typename T, TensorFormat data_format>
struct DepthToSpaceOpFunctor {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output);

  // This 5-D version is to support NCHW_VECT_C.
  void operator()(const Device& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DEPTHTOSPACE_OP_H_
