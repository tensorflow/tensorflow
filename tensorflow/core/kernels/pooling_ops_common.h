#ifndef TENSORFLOW_KERNELS_POOLING_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_POOLING_OPS_COMMON_H_

#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/avgpooling_op.h"
#include "tensorflow/core/kernels/maxpooling_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/NeuralNetworks"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// A helper class to manage sizes and shapes for pooling operations.
struct PoolParameters {
  // Updates context->status if there is an invalid input.
  PoolParameters(OpKernelContext* context, const std::vector<int32>& ksize,
                 const std::vector<int32>& stride, Padding padding,
                 const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();

  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_rows;
  int window_cols;
  int depth_window;

  int row_stride;
  int col_stride;
  int depth_stride;

  int out_height;
  int out_width;
  int out_depth;

  int pad_rows;
  int pad_cols;
  int pad_depth;
};

// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MaxPoolingOp : public UnaryOp<T> {
 public:
  explicit MaxPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context, ksize_, stride_, padding_,
                          tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    if (params.depth_window > 1) {
      DepthwiseMaxPool(context, output, tensor_in, params);
    } else {
      SpatialMaxPool(context, output, tensor_in, params, padding_);
    }
  }

 private:
  // Single-threaded implementation of DepthwiseMaxPool which
  // does not handle all of the same options as SpatialMaxPool
  // (strict assumptions on no padding, stride).
  //
  // TODO(vrv): implement a more general depthwise-max pool that works
  // on GPU as well.
  void DepthwiseMaxPool(OpKernelContext* context, Tensor* output,
                        const Tensor& tensor_in, const PoolParameters& params) {
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        in_by_pool(tensor_in.flat<T>().data(), params.depth_window,
                   tensor_in.NumElements() / params.depth_window);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> out_by_pool(
        output->flat<T>().data(), 1, output->NumElements());
    out_by_pool = in_by_pool.colwise().maxCoeff();
  }

  void SpatialMaxPool(OpKernelContext* context, Tensor* output,
                      const Tensor& tensor_in, const PoolParameters& params,
                      const Padding& padding) {
    // On GPU, use Eigen's Spatial Max Pooling.  On CPU, use an
    // EigenMatrix version that is currently faster than Eigen's
    // Spatial MaxPooling implementation.
    //
    // TODO(vrv): Remove this once we no longer need it.
    if (std::is_same<Device, GPUDevice>::value) {
      Eigen::PaddingType pt = BrainPadding2EigenPadding(padding);
      functor::SpatialMaxPooling<Device, T>()(
          context->eigen_device<Device>(), output->tensor<T, 4>(),
          tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
          params.row_stride, params.col_stride, pt);
    } else {
      typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          ConstEigenMatrixMap;
      typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
          EigenMatrixMap;

      ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                                 params.tensor_in_cols * params.tensor_in_rows *
                                     params.tensor_in_batch);
      EigenMatrixMap out_mat(
          output->flat<T>().data(), params.depth,
          params.out_width * params.out_height * params.tensor_in_batch);

      // Initializes the output tensor with MIN<T>.
      output->flat<T>().setConstant(Eigen::NumTraits<T>::lowest());

      // The following code basically does the following:
      // 1. Flattens the input and output tensors into two dimensional arrays.
      //    tensor_in_as_matrix:
      //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
      //    output_as_matrix:
      //      depth by (out_width * out_height * tensor_in_batch)
      //
      // 2. Walks through the set of columns in the flattened
      // tensor_in_as_matrix,
      //    and updates the corresponding column(s) in output_as_matrix with the
      //    max value.
      for (int b = 0; b < params.tensor_in_batch; ++b) {
        for (int h = 0; h < params.tensor_in_rows; ++h) {
          for (int w = 0; w < params.tensor_in_cols; ++w) {
            // (h_start, h_end) * (w_start, w_end) is the range that the input
            // vector projects to.
            const int hpad = h + params.pad_rows;
            const int wpad = w + params.pad_cols;
            const int h_start =
                (hpad < params.window_rows)
                    ? 0
                    : (hpad - params.window_rows) / params.row_stride + 1;
            const int h_end =
                std::min(hpad / params.row_stride + 1, params.out_height);
            const int w_start =
                (wpad < params.window_cols)
                    ? 0
                    : (wpad - params.window_cols) / params.col_stride + 1;
            const int w_end =
                std::min(wpad / params.col_stride + 1, params.out_width);
            // compute elementwise max
            const int in_offset =
                (b * params.tensor_in_rows + h) * params.tensor_in_cols + w;
            for (int ph = h_start; ph < h_end; ++ph) {
              for (int pw = w_start; pw < w_end; ++pw) {
                const int out_offset =
                    (b * params.out_height + ph) * params.out_width + pw;
                out_mat.col(out_offset) =
                    out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
              }
            }
          }
        }
      }
    }
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

template <typename Device, typename T>
void SpatialAvgPool(OpKernelContext* context, Tensor* output,
                    const Tensor& input, const PoolParameters& params,
                    const Padding& padding) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      EigenMatrixMap;

  auto in_flat = input.flat<T>();
  auto out_flat = output->flat<T>();

  ConstEigenMatrixMap in_mat(
      in_flat.data(), params.depth,
      params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
  EigenMatrixMap out_mat(
      out_flat.data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);
  Eigen::Matrix<T, Eigen::Dynamic, 1> out_count(out_mat.cols());
  out_count.setZero();

  // Initializes output to zero.
  out_flat.setZero();

  // The following code basically does the following:
  // 1. Flattens the input and output tensors into two dimensional arrays.
  //    tensor_in_as_matrix:
  //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
  //    output_as_matrix:
  //      depth by (out_width * out_height * tensor_in_batch)
  //
  // 2. Walks through the set of columns in the flattened
  // tensor_in_as_matrix,
  //    and updates the corresponding column(s) in output_as_matrix with the
  //    average value.
  for (int b = 0; b < params.tensor_in_batch; ++b) {
    for (int h = 0; h < params.tensor_in_rows; ++h) {
      for (int w = 0; w < params.tensor_in_cols; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        const int hpad = h + params.pad_rows;
        const int wpad = w + params.pad_cols;
        const int h_start =
            (hpad < params.window_rows)
                ? 0
                : (hpad - params.window_rows) / params.row_stride + 1;
        const int h_end =
            std::min(hpad / params.row_stride + 1, params.out_height);
        const int w_start =
            (wpad < params.window_cols)
                ? 0
                : (wpad - params.window_cols) / params.col_stride + 1;
        const int w_end =
            std::min(wpad / params.col_stride + 1, params.out_width);
        const int in_offset =
            (b * params.tensor_in_rows + h) * params.tensor_in_cols + w;
        Eigen::DSizes<ptrdiff_t, 2> in_indices(0, in_offset);
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            const int out_offset =
                (b * params.out_height + ph) * params.out_width + pw;
            out_mat.col(out_offset) += in_mat.col(in_offset);
            out_count(out_offset)++;
          }
        }
      }
    }
  }
  DCHECK_GT(out_count.minCoeff(), 0);
  out_mat.array().rowwise() /= out_count.transpose().array();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_POOLING_OPS_COMMON_H_
