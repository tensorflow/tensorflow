// LRN = Local Response Normalization
// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifndef __ANDROID__
#include "tensorflow/core/util/work_sharder.h"
#endif

namespace tensorflow {

// Create a depth-by-depth band matrix with 1s along a swath of size (2 *
// depth_radius + 1) around the diagonal.
static void GetBandMatrix(int depth, int64 depth_radius,
                          Eigen::Tensor<float, 2, Eigen::RowMajor>* result) {
  result->setZero();
  for (int row = 0; row < depth; ++row) {
    const int begin = std::max<int>(0, row - depth_radius);
    const int end = std::min<int64>(depth, row + depth_radius + 1);
    Eigen::DSizes<ptrdiff_t, 2> start(row, begin);
    Eigen::DSizes<ptrdiff_t, 2> sizes(1, end - begin);
    result->slice(start, sizes).setConstant(1.0f);
  }
}

class LRNOp : public OpKernel {
 public:
  explicit LRNOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius_));
    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in = context->input(0);
    OP_REQUIRES(context, in.dims() == 4,
                errors::InvalidArgument("in must be 4-dimensional"));
    const int64 batch = in.dim_size(0);
    const int64 rows = in.dim_size(1);
    const int64 cols = in.dim_size(2);
    const int64 depth = in.dim_size(3);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch, rows, cols, depth}), &output));

#ifdef __ANDROID__
    MognetLRN(in, batch, rows, cols, depth, output);
#else
    const int nodes = cols * rows;
    auto in_shaped = in.shaped<float, 2>({nodes * batch, depth});

    // Multiplying the input with the band matrix has the effect of reducing the
    // correct patch along the depth.
    Eigen::Tensor<float, 2, Eigen::RowMajor> multiplier(depth, depth);
    GetBandMatrix(depth, depth_radius_, &multiplier);

    auto out_shaped = output->shaped<float, 2>({nodes * batch, depth});
    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    /// TODO(keveman): Optimize for beta in {0, 1, 0.5}
    out_shaped.device(context->eigen_cpu_device()) =
        in_shaped /
        in_shaped.square()
            .contract(multiplier, dims)
            .unaryExpr([this](float x) { return bias_ + alpha_ * x; })
            .pow(beta_);
#endif
  }

 private:
  typedef Eigen::Tensor<float, 1, Eigen::RowMajor>::DimensionPair DimPair;

  void MognetLRN(const Tensor& in, const int batch, const int rows,
                 const int cols, const int depth, Tensor* out) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
    data_in(in.flat<float>().data(), depth, batch * rows * cols);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> data_out(
        out->flat<float>().data(), depth, batch * rows * cols);

    const int double_depth_radius = depth_radius_ * 2;
    Eigen::VectorXf padded_square(data_in.rows() + double_depth_radius);
    padded_square.setZero();
    for (int r = 0; r < data_in.cols(); ++r) {
      // Do local response normalization for data_in(:, r)
      // first, compute the square and store them in buffer for repeated use
      padded_square.block(depth_radius_, 0, data_out.rows(), 1) =
          data_in.col(r).cwiseProduct(data_in.col(r)) * alpha_;
      // Then, compute the scale and writes them to data_out
      float accumulated_scale = 0;
      for (int i = 0; i < double_depth_radius; ++i) {
        accumulated_scale += padded_square(i);
      }
      for (int i = 0; i < data_in.rows(); ++i) {
        accumulated_scale += padded_square(i + double_depth_radius);
        data_out(i, r) = bias_ + accumulated_scale;
        accumulated_scale -= padded_square(i);
      }
    }

    // In a few cases, the pow computation could benefit from speedups.
    if (beta_ == 1) {
      data_out.array() = data_in.array() * data_out.array().inverse();
    } else if (beta_ == 0.5) {
      data_out.array() = data_in.array() * data_out.array().sqrt().inverse();
    } else {
      data_out.array() = data_in.array() * data_out.array().pow(-beta_);
    }
  }

  int64 depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_KERNEL_BUILDER(Name("LRN").Device(DEVICE_CPU), LRNOp);

#ifndef __ANDROID__

class LRNGradOp : public OpKernel {
 public:
  explicit LRNGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius_));
    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in_grads = context->input(0);
    const Tensor& in_image = context->input(1);
    const Tensor& out_image = context->input(2);

    OP_REQUIRES(context, in_grads.dims() == 4 && in_image.dims() == 4,
                errors::InvalidArgument("inputs must be 4-dimensional"));
    const int64 batch = in_grads.dim_size(0);
    const int64 rows = in_grads.dim_size(1);
    const int64 cols = in_grads.dim_size(2);
    const int64 depth = in_grads.dim_size(3);
    OP_REQUIRES(
        context,
        in_image.dim_size(0) == batch && in_image.dim_size(1) == rows &&
            in_image.dim_size(2) == cols && in_image.dim_size(3) == depth &&
            out_image.dim_size(0) == batch && out_image.dim_size(1) == rows &&
            out_image.dim_size(2) == cols && out_image.dim_size(3) == depth,
        errors::InvalidArgument(
            "input_grads, input_image, and out_image should have the same "
            "shape"));
    const auto nodes = cols * rows;
    auto grads_shaped = in_grads.shaped<float, 2>({nodes * batch, depth});
    auto in_shaped = in_image.shaped<float, 2>({nodes * batch, depth});
    auto activations = out_image.shaped<float, 2>({nodes * batch, depth});

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch, rows, cols, depth}), &output));
    auto out_shaped = output->shaped<float, 2>({nodes * batch, depth});
    out_shaped.setZero();

    auto shard = [this, activations, in_shaped, grads_shaped, out_shaped,
                  depth](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        for (int64 j = 0; j < depth; ++j) {
          // Let y be the LRN activations and x be the inputs along the depth
          // dimension. (LRN operates independently along rows, cols, and
          // batch).
          // We have
          // yi = xi / (bias + alpha(sum_j_{i - depth_radius}^{i + depth_radius}
          //      x_j^2))^beta
          //
          // Let N = (bias + alpha(sum_j_{i - depth_radius}^{i + depth_radius}
          //           x_j^2))
          // dy_i/dx_i = (N^beta - xi. beta*N^(beta-1)*2*alpha*xi)/N^(2*beta)
          // dy_i/dx_j = (       - xi. beta*N^(beta-1)*2*alpha*xj)/N^(2*beta)
          //
          // NOTE(keveman) : We can compute N by doing (yi/xi) ^ (1/beta).
          // However, this is numerically unstable for small values of xi. We
          // compute N explicitly here to avoid that.

          int64 depth_begin = std::max<int64>(0, j - depth_radius_);
          int64 depth_end = std::min<int64>(depth, j + depth_radius_ + 1);

          float norm = 0.0f;
          for (int64 k = depth_begin; k < depth_end; ++k) {
            norm += in_shaped(i, k) * in_shaped(i, k);
          }
          norm = alpha_ * norm + bias_;
          DCHECK_GT(norm, 1e-6);
          for (int64 k = depth_begin; k < depth_end; ++k) {
            float dyi = -2.0f * alpha_ * beta_ * in_shaped(i, k) *
                        activations(i, j) / norm;
            if (k == j) {
              dyi += std::pow(norm, -beta_);
            }
            dyi *= grads_shaped(i, j);
            const_cast<TTypes<float, 2>::Tensor&>(out_shaped)(i, k) += dyi;
          }
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, nodes * batch,
          depth * depth, shard);
  }

 private:
  typedef Eigen::Tensor<float, 1, Eigen::RowMajor>::DimensionPair DimPair;

  int64 depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

REGISTER_KERNEL_BUILDER(Name("LRNGrad").Device(DEVICE_CPU), LRNGradOp);

#endif  // __ANDROID__

}  // namespace tensorflow
