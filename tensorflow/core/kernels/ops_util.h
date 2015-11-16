#ifndef TENSORFLOW_KERNELS_OPS_UTIL_H_
#define TENSORFLOW_KERNELS_OPS_UTIL_H_

// This file contains utilities for various operations.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// Call this function from a test if op kernels are not being
// registered.  This can happen if the test is linked in a shared
// mode and has no direct references to any code from this directory.
void RequireDefaultOps();

// Get2dOutputSize(): Given an input tensor, kernel, stride and padding
// type, the function computes the output and padding dimensions.
//
// Convolution layers take in an input tensor of shape (D, C, R, B), and
// convolve it with a set of filters, which can also be presented as a
// tensor (D, K, K, M), where M is the number of filters, K is the filter size,
// and each 3-dimensional tensor of size (D, K, K) is a filter. For
// simplicity we assume that we always use square filters (which is usually the
// case in images). It also takes in a few additional parameters:
//
// Stride (S): the stride with which we apply the filters. This is the offset
// between locations where we apply the filters. A larger stride
// means that the output will be spatially smaller.
//
// Padding (P): the padding we apply to the input tensor along the R and C
// dimensions. This is usually used to make sure that the spatial dimension
// do not shrink when we progress with convolutions. Two types of padding are
// often used:
//   SAME: the pad value is computed so that the output will have size R/S
//         and C/S.
//   VALID: no padding is carried out.
// The padded area is zero-filled.
//
// The output dimensions for convolution and many other operations, when given
// all the parameters above, are as follows:
// - When Padding = SAME: the output size is (B, R', C', M), where
//     R' = ceil(float(R) / float(S))
//     C' = ceil(float(C) / float(S))
//   where ceil is the ceiling function. The number of padded rows and columns
//   are computed as:
//     Pr = ((R' - 1) * S + K - R) / 2
//     Pc = ((C' - 1) * S + K - C) / 2
//   When the stride is 1, we have the simplified case
//     R'=R, C'=C, Pr=Pc=(K-1)/2.
//   This is where SAME comes from - the output has the same size as the input
//   has.
//
// - When Padding = VALID: the output size is computed as
//     R' = ceil(float(R - K + 1) / float(S))
//     C' = ceil(float(C - K + 1) / float(S))
//   and the number of padded rows and columns are computed in the same way.
//   When the stride is 1, we have the simplified case
//     R'=R-K+1, C'=C-K+1, Pr=0, Pc=0.
//
// For convolution, mathematically, the output value at location (b, r', c', m)
// is the inner product of two vectors: the chunk of input at
//    (b, (r'*S-Pr) : (r'*S-Pr+K), (c'*S-Pc) : (c'*S-Pc+K), :),
// and the filter at (m, :, :, :).
//
Status Get2dOutputSize(const int in_height, const int in_width,
                       int filter_height, int filter_width, int row_stride,
                       int col_stride, Padding padding, int* new_height,
                       int* new_width, int* pad_rows, int* pad_cols);

// Returns the same output dimensions as in Get2dOutputSize, but returns verbose
// padding dimensions (top/bottom/left/right). Any excess padding (caused by
// an odd padding size value) is added to the 'pad_bottom' and 'pad_right'
// dimensions.
Status Get2dOutputSizeVerbose(const int in_height, const int in_width,
                              int filter_height, int filter_width,
                              int row_stride, int col_stride, Padding padding,
                              int* new_height, int* new_width, int* pad_top,
                              int* pad_bottom, int* pad_left, int* pad_right);

// Calculates broadcast starting index and size.  For SAME padding, addition
// padding could be applied to right, left, top and bottom.  Depending on the
// current index, input size, kernel size, stride, padding size, the starting
// index and size for broadcast for that dimension are different from the
// current index and kernel size.
// This is mainly used by gradient algorithms for pooling operations.
Status GetBroadcastSize(const int index, const int in_size, const int ksize,
                        const int stride, const int pad_size, int* bindex,
                        int* bsize);

// Converts Brain's Padding to Eigen's PaddingType.
Eigen::PaddingType BrainPadding2EigenPadding(Padding padding);

// Given a shape 's' of a tensor of type T. Returns true iff the
// number of bytes occupied by each dim 0 (i.e., &tensor(i + 1, ...) -
// &tensor(i, ...)) is multiple of EIGEN_ALIGN_BYTES.
template <typename T>
bool IsInnerDimsSizeAligned(const TensorShape& s) {
  if (s.dims() == 0) return false;
  const int64 dim0_size = s.dim_size(0);
  if (dim0_size == 0) return false;
  const int64 bytes_per_dim0 = (s.num_elements() / dim0_size) * sizeof(T);
  return bytes_per_dim0 % EIGEN_MAX_ALIGN_BYTES == 0;
}

// Returns in 'col_data', image patches in storage order (height, width, depth)
// extracted from image at 'input_data', which is requred to be in storage
// order (batch, height, width, depth).
// Implementation written by Yangqing Jia (jiayq).
template <typename T>
void Im2col(const T* input_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* col_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(col_data, input_data + (ih * width + iw) * depth,
                   sizeof(T) * depth);
          } else {
            // This should be simply padded with zero.
            memset(col_data, 0, sizeof(T) * depth);
          }
          col_data += depth;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

// Returns in 'im_data' image patch in storage order (height, width, depth),
// constructed from patches in 'col_data', which is required to be in storage
// order (out_height * out_width, filter_height, filter_width, in_depth).
// Implementation by Yangqing Jia (jiayq).
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* im_data) {
  memset(im_data, 0, sizeof(T) * height * width * depth);
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      T* im_patch_data = im_data + (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            // TODO(andydavis) Vectorize this loop (if compiler does not).
            for (int i = 0; i < depth; ++i) {
              im_patch_data[i] += col_data[i];
            }
          }
          im_patch_data += depth;
          col_data += depth;
        }
        // Jump over remaining number of depth.
        im_patch_data += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_OPS_UTIL_H_
