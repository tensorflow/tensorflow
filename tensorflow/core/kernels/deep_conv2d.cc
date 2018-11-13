/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/deep_conv2d.h"

#include <stdlib.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/winograd_transform.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// DeepConv2D is a Conv2D implementation specialized for deep convolutions (i.e
// large 'in_depth' and 'out_depth' product. See cost models below for details).
//
// DeepConv2D is implemented by computing the following equation:
//
//   y = C[Ad * Bg]
//
//   C: output transform matrix
//   A: input data transform matrix
//   B: filter transform matrix
//   d: vectorized data tile
//   g: vectorized filter tile
//   y: vectorized output tile
//
// The transform matrices and input, filter and output tile sizes are all
// specified by the DeepConv2DTransform implementation selected at the
// start of the DeepConv2D call, based on convolution parameters.

// Approximate cost models for direct and deep convolutions.
static int64 GetDeepConvCost(int input_tile_rows, int input_tile_cols,
                             int out_tile_rows, int out_tile_cols, int in_depth,
                             int out_depth, int out_rows, int out_cols) {
  // Input transform cost.
  const int64 input_tile_spatial_size = input_tile_rows * input_tile_cols;
  const int64 input_transform_cost =
      input_tile_spatial_size * input_tile_spatial_size * in_depth;

  // Element-wise products (each product is a MatMul across depth).
  const int64 product_cost = input_tile_spatial_size * in_depth * out_depth;

  // Output transform cost.
  const int64 output_tile_spatial_size = out_tile_rows * out_tile_cols;
  const int64 output_transform_cost =
      output_tile_spatial_size * input_tile_spatial_size * out_depth;

  // Calculate number of input tiles to process.
  const int64 row_tiles = (out_rows + out_tile_rows - 1) / out_tile_rows;
  const int64 col_tiles = (out_cols + out_tile_cols - 1) / out_tile_cols;
  const int64 num_tiles = row_tiles * col_tiles;

  // Return total cost.
  return num_tiles *
         (input_transform_cost + product_cost + output_transform_cost);
}

static int64 GetDirectConvCost(int filter_rows, int filter_cols, int in_depth,
                               int out_depth, int out_rows, int out_cols) {
  return filter_rows * filter_cols * in_depth * out_depth * out_rows * out_cols;
}

// Reads environment variable 'env_var_name'.
// Returns 'true' if environment variable is enabled, false otherwise.
static bool ReadBoolFromEnvVar(const char* env_var_name, bool default_val) {
  const char* tf_env_var_val = getenv(env_var_name);
  if (tf_env_var_val != nullptr) {
    StringPiece tf_env_var_val_str(tf_env_var_val);
    if (tf_env_var_val_str == "0") {
      return false;
    }
    return true;
  }
  return default_val;
}

// Returns true if convolution can be computed efficiently by DeepConv2D,
// returns false otherwise.
// TODO(andydavis) Add support for other filter sizes and strides.
// TODO(andydavis) Add support for autotuning.
bool CanUseDeepConv2D(int stride_rows, int stride_cols, int filter_rows,
                      int filter_cols, int in_depth, int out_depth,
                      int out_rows, int out_cols) {
  // Check if convolution parameters are supported.
  // TODO(andydavis) Add support for multiple filter sizes and strides.
  if (stride_rows > 1 || stride_cols > 1 || filter_rows != 3 ||
      filter_cols != 3) {
    return false;
  }

  // Check if deep convolution is enabled by environment variable.
  // NOTE: IF this environment variable name changes, update conv_ops_test.py.
  if (!ReadBoolFromEnvVar("TF_USE_DEEP_CONV2D", false)) {
    return false;
  }

  // Check if flop cost of deep convolution is less than direct convolution.
  WinogradTransform<float> t;
  const int64 deep_conv_cost = GetDeepConvCost(
      t.input_shape().rows, t.input_shape().cols, t.output_shape().rows,
      t.output_shape().cols, in_depth, out_depth, out_rows, out_cols);
  const int64 direct_conv_cost = GetDirectConvCost(
      filter_rows, filter_cols, in_depth, out_depth, out_rows, out_cols);

  VLOG(2) << "CanUseDeepConv2D"
          << " deep_conv_cost: " << deep_conv_cost
          << " direct_conv_cost: " << direct_conv_cost << " deep_direct_ratio: "
          << (static_cast<float>(deep_conv_cost) /
              static_cast<float>(direct_conv_cost))
          << " use_deep_conv: " << (deep_conv_cost < direct_conv_cost);
  return deep_conv_cost < direct_conv_cost;
}

typedef Eigen::ThreadPoolDevice CPUDevice;

// Copies data from 'filter_in' to 'filter_buf' along 'in_depth' dimension.
//
// filter_in:
//   [filter_rows, filter_cols, in_depth, out_depth]
//
// filter_buf:
//   [base_filter_rows, base_filter_cols, in_depth]
//
template <typename T>
struct CopyFilterDepth {
  void operator()(const Conv2DArgs& args, const T* filter_in, T* filter_buf) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static constexpr int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 vectorized_size = args.in_depth / kPacketSize;
    const int64 scalar_size = args.in_depth % kPacketSize;
    const int64 input_stride = args.out_depth * kPacketSize;

    // Copy vectorized portion of depth dimension.
    for (int64 d = 0; d < vectorized_size; ++d) {
      auto v = Eigen::internal::pgather<T, Packet>(filter_in + d * input_stride,
                                                   args.out_depth);
      Eigen::internal::pstoreu<T>(filter_buf + d * kPacketSize, v);
    }
    // Copy scalar portion of inner dimension.
    const int64 in_scalar_base = vectorized_size * input_stride;
    const int64 buf_scalar_base = vectorized_size * kPacketSize;
    for (int64 d = 0; d < scalar_size; ++d) {
      filter_buf[buf_scalar_base + d] =
          filter_in[in_scalar_base + d * args.out_depth];
    }
  }
};

// Computes transform of 'num_filters' from 'filter_in' starting at 'od_start'.
// Intermediate results (i.e. output of MatMul('transform_matrix', 'filter_in'))
// are stored in 'out_buffer'. The final result is copied from 'out_buffer' to
// 'filter_out' at the coordinate stride required by the transformed filter
// data layout.
//
// filter_in:
//   [base_filter_rows, base_filter_cols, num_filters, shard_rows, shard_cols,
//    in_depth]
//
// filter_out:
//   [tile_rows, tile_cols, out_depth, shard_rows, shard_cols, in_depth]
//
// transform_matrix:
//   [tile_spatial_size, base_filter_spatial_size]
//
// out_buffer:
//   [tile_spatial_size, num_filters, shard_rows, shard_cols, in_depth]

template <typename T>
struct ComputeFilterRangeTransform {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

  typedef Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      MatrixMap;
  typedef Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      ConstMatrixMap;

  void operator()(const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform, const int64 od_start,
                  const int64 num_filters, const int64 shard_rows,
                  const int64 shard_cols, const T* filter_in,
                  const int64 in_stride, const int64 out_stride,
                  const T* transform_matrix, T* out_buffer, T* filter_out) {
    namespace ei = Eigen::internal;

    const int64 in_depth = args.in_depth;
    const int64 base_filter_rows = transform->filter_shape().rows;
    const int64 base_filter_cols = transform->filter_shape().cols;
    const int64 base_filter_spatial_size = base_filter_rows * base_filter_cols;
    const int64 tile_rows = transform->input_shape().rows;
    const int64 tile_cols = transform->input_shape().cols;
    const int64 tile_spatial_size = tile_rows * tile_cols;

    // Compute transform of 'num_filters' by 'transform_matrix'.
    ConstMatrixMap A(transform_matrix, tile_spatial_size,
                     base_filter_spatial_size);
    ConstMatrixMap B(filter_in, base_filter_spatial_size, in_stride);
    MatrixMap C(out_buffer, tile_spatial_size, in_stride);

    C.noalias() = A * B;

    // Copy 'out_buffer' to 'filter_out' at required filter output stride.
    const int64 scalar_size = in_depth % kPacketSize;
    const int64 vectorized_size = in_depth / kPacketSize;

    const int64 shard_stride = args.in_depth;
    const int64 out_depth_stride = shard_rows * shard_cols * shard_stride;

    for (int64 od = 0; od < num_filters; ++od) {
      const int64 out_depth_buf_base = od * out_depth_stride;
      const int64 out_depth_base = (od_start + od) * out_depth_stride;

      // TODO(andydavis) Shard filters that are multiples of base filter sizes.
      for (int64 s_r = 0; s_r < shard_rows; ++s_r) {
        for (int64 s_c = 0; s_c < shard_cols; ++s_c) {
          const int64 shard_base = shard_stride * (s_r * shard_cols + s_c);

          for (int64 i = 0; i < tile_spatial_size; ++i) {
            const int64 in_base =
                i * in_stride + out_depth_buf_base + shard_base;
            const int64 out_base = i * out_stride + out_depth_base + shard_base;
            // Copy vectorized portion of 'in_depth'.
            for (int64 d = 0; d < vectorized_size; ++d) {
              auto v =
                  ei::ploadu<Packet>(out_buffer + in_base + d * kPacketSize);
              ei::pstoreu<T>(filter_out + out_base + d * kPacketSize, v);
            }
            // Transform scalar portion of 'in_depth'.
            const int64 scalar_base = vectorized_size * kPacketSize;
            for (int64 d = 0; d < scalar_size; ++d) {
              filter_out[out_base + scalar_base + d] =
                  out_buffer[in_base + scalar_base + d];
            }
          }
        }
      }
    }
  }
};

// Transforms 'num_filters' from 'filter_in', starting at 'od_start'.
// For each filter in 'num_filters', copies data for all filter shards from
// 'filter_in' into 'filter_buf', adding zero-padding as needed.
// Calls ComputeFilterRangeTransform to compute filter transform of data
// in 'filter_buf' by 'transform_matrix', storing the result in 'filter_out'.
//
// filter_in:
//   [filter_rows, filter_cols, in_depth, out_depth]
//
// filter_out:
//   [tile_rows, tile_cols, out_depth, shard_rows, shard_cols, in_depth]
//
// filter_buffer:
//   [base_filter_rows, base_filter_cols, num_filters, shard_rows, shard_cols,
//    in_depth]
//
// transform_matrix:
//   [tile_spatial_size, base_filter_spatial_size]
//
// out_buffer:
//   [tile_spatial_size, num_filters, shard_rows, shard_cols, in_depth]
//

template <typename T>
struct TransformFilterRange {
  void operator()(const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform, const int64 od_start,
                  const int64 od_limit, const T* filter_in,
                  const T* transform_matrix, T* out_buffer, T* filter_buf,
                  T* filter_out) {
    const int64 num_filters = od_limit - od_start;
    const int64 base_filter_rows = transform->filter_shape().rows;
    const int64 base_filter_cols = transform->filter_shape().cols;
    const int64 base_filter_spatial_size = base_filter_rows * base_filter_cols;

    // Compute number of filter shards.
    const int64 residual_row =
        std::max(int64{0}, args.filter_rows - base_filter_rows);
    const int64 shard_rows = 1 + (residual_row + 2 - 1) / 2;

    const int64 residual_col =
        std::max(int64{0}, args.filter_cols - base_filter_cols);
    const int64 shard_cols = 1 + (residual_col + 2 - 1) / 2;

    // Compute strides to be used for input and output IO.
    const int64 shard_stride = args.in_depth;
    const int64 out_depth_stride = shard_rows * shard_cols * shard_stride;
    const int64 coord_stride = out_depth_stride * args.out_depth;
    const int64 filter_buf_stride =
        num_filters * shard_rows * shard_cols * args.in_depth;
    const int64 tile_stride_rows = transform->output_shape().rows;
    const int64 tile_stride_cols = transform->output_shape().cols;

    const int64 filter_buf_size = base_filter_spatial_size * num_filters *
                                  shard_rows * shard_cols * args.in_depth;
    memset(filter_buf, 0, sizeof(T) * filter_buf_size);

    // Copy filter range into 'filter_buf'.
    for (int64 od = 0; od < num_filters; ++od) {
      const int64 out_depth_base = od * out_depth_stride;

      // TODO(andydavis) Shard filters that are multiples of base filter sizes.
      for (int64 s_r = 0; s_r < shard_rows; ++s_r) {
        const int64 row_offset = s_r == 0 ? 0 : 1;

        for (int64 s_c = 0; s_c < shard_cols; ++s_c) {
          const int64 col_offset = s_c == 0 ? 0 : 1;
          const int64 f_r_start = s_r * tile_stride_rows;
          const int64 f_c_start = s_c * tile_stride_cols;

          const int64 shard_base = shard_stride * (s_r * shard_cols + s_c);

          for (int64 b_r = row_offset; b_r < base_filter_rows; ++b_r) {
            const int64 f_r = f_r_start + b_r;
            if (f_r >= args.filter_rows) continue;

            for (int64 b_c = col_offset; b_c < base_filter_cols; ++b_c) {
              const int64 f_c = f_c_start + b_c;
              if (f_c >= args.filter_cols) continue;

              const int64 in_index =
                  args.out_depth *
                      (args.in_depth * (f_r * args.filter_cols + f_c)) +
                  (od_start + od);

              const int64 buf_index =
                  filter_buf_stride * (b_r * base_filter_cols + b_c) +
                  out_depth_base + shard_base;

              CopyFilterDepth<T>()(args, filter_in + in_index,
                                   filter_buf + buf_index);
            }
          }
        }
      }
    }

    // Compute filter transform of data in 'filter_buf' by 'transform_matrix'.
    // Intermediate results are stored in 'out_buffer'.
    // Final results are stored in 'filter_out'.
    ComputeFilterRangeTransform<T>()(args, transform, od_start, num_filters,
                                     shard_rows, shard_cols, filter_buf,
                                     filter_buf_stride, coord_stride,
                                     transform_matrix, out_buffer, filter_out);
  }
};

// Transforms all filters from 'filter_in', storing result in 'filter_out'.
//
// filter_in:
//   [filter_rows, filter_cols, in_depth, out_depth]
//
// filter_out:
//   [tile_rows, tile_cols, out_depth, shard_rows, shard_cols, in_depth]
//
template <typename T>
struct TransformFilters {
  void operator()(OpKernelContext* ctx, const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform,
                  const int64 filter_shards_row, const int64 filter_shards_col,
                  const T* filter_in, T* filter_out) {
    const int64 in_depth = args.in_depth;
    const int64 out_depth = args.out_depth;

    const int64 tile_rows = transform->input_shape().rows;
    const int64 tile_cols = transform->input_shape().cols;
    const int64 tile_spatial_size = tile_rows * tile_cols;

    const int64 base_filter_rows = transform->filter_shape().rows;
    const int64 base_filter_cols = transform->filter_shape().cols;
    const int64 base_filter_spatial_size = base_filter_rows * base_filter_cols;

    const int64 filter_shards_total = filter_shards_row * filter_shards_col;

    // Calculate filter transform batch based on cache/filter sizes.

    // Cache budget (based on L2 cache size = 256KB).
    // TODO(andydavis) Read cache size from system.
    const int64 cache_size = (256LL << 10) / sizeof(T);

    // Fixed cost.
    const int64 filter_transform_matrix_size =
        tile_spatial_size * base_filter_spatial_size;

    // Per-filter costs.
    const int64 filter_total_size =
        base_filter_spatial_size * in_depth * filter_shards_total;

    const int64 filter_transform_buffer_size =
        base_filter_spatial_size * filter_shards_total * in_depth;

    const int64 filter_out_buf_size =
        tile_spatial_size * filter_shards_total * in_depth;

    // Total per-filter costs.
    const int64 per_filter_cost =
        filter_total_size + filter_transform_buffer_size + filter_out_buf_size;

    // Remove fixed cost and divide by per-filter cost.
    const int64 num_filters_cache =
        std::max(int64{1},
                 (cache_size - filter_transform_matrix_size) / per_filter_cost);
    const int64 num_filters_transform = std::min(out_depth, num_filters_cache);

    // Allocate buffer for filter transform matrix:
    //   [tile_spatial_size, base_filter_spatial_size]
    Tensor filter_transform_matrix;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DataTypeToEnum<T>::value,
                 TensorShape({tile_spatial_size, base_filter_spatial_size}),
                 &filter_transform_matrix));
    T* transform_matrix = filter_transform_matrix.template flat<T>().data();
    transform->GetFilterTransformMatrix(
        tile_spatial_size, base_filter_spatial_size, transform_matrix);

    auto shard = [&ctx, &args, &transform, &base_filter_rows, &base_filter_cols,
                  &num_filters_transform, &in_depth, &out_depth,
                  &filter_shards_row, &filter_shards_col, &tile_spatial_size,
                  &filter_in, &transform_matrix,
                  &filter_out](int64 start, int64 limit) {
      // Allocate buffer for pre-processed filter:
      //   [base_filter_rows, base_filter_cols, num_filters_transform, in_depth]
      //
      Tensor filter_transform_buffer;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(
                         DataTypeToEnum<T>::value,
                         TensorShape({base_filter_rows, base_filter_cols,
                                      num_filters_transform, filter_shards_row,
                                      filter_shards_col, in_depth}),
                         &filter_transform_buffer));
      T* filter_buf = filter_transform_buffer.template flat<T>().data();

      // Allocate buffer for output filter transform matrix:
      //   [tile_rows, tile_cols, out_depth, shard_rows, shard_cols, in_depth]
      Tensor filter_output_buffer;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DataTypeToEnum<T>::value,
              TensorShape({tile_spatial_size, num_filters_transform,
                           filter_shards_row, filter_shards_col, in_depth}),
              &filter_output_buffer));
      T* out_buffer = filter_output_buffer.template flat<T>().data();

      const int64 num_filters = limit - start;
      const int64 od_unroll = num_filters_transform;
      const int64 od_unroll_limit = (num_filters / od_unroll) * od_unroll;

      for (int64 od = start; od < od_unroll_limit; od += od_unroll) {
        TransformFilterRange<T>()(args, transform, od, od + od_unroll,
                                  filter_in, transform_matrix, out_buffer,
                                  filter_buf, filter_out);
      }

      if (od_unroll_limit < limit) {
        TransformFilterRange<T>()(args, transform, od_unroll_limit, limit,
                                  filter_in, transform_matrix, out_buffer,
                                  filter_buf, filter_out);
      }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    const int64 shard_cost = args.filter_rows * args.filter_cols * in_depth *
                             filter_shards_total * tile_spatial_size;
    // TODO(andydavis) Resolve performance of multi-threaded filter transforms.
    Shard(1, worker_threads.workers, out_depth, shard_cost, shard);
  }
};

// Packs transformed filters stored in 'lhs_input' into 'lhs_block' in a
// gemm-kernel friendly data layout.
//
// Data layout for 'lhs_block':
//   [out_depth, shard_rows, shard_cols, in_depth].

template <typename T>
class GemmFilterPacker {
 public:
  typedef Eigen::internal::const_blas_data_mapper<T, int64, Eigen::RowMajor>
      LhsMapper;
  typedef Eigen::internal::gebp_traits<T, T> Traits;
  Eigen::internal::gemm_pack_lhs<
      T, int64, LhsMapper, Traits::mr, Traits::LhsProgress,
      typename Traits::LhsPacket4Packing, Eigen::RowMajor>
      pack_lhs;

  GemmFilterPacker(const int64 rows, const int64 depth, const T* lhs_input,
                   T* lhs_block)
      : rows_(rows),
        depth_(depth),
        lhs_block_(lhs_block),
        lhs_mapper_(lhs_input, depth_) {}

  void Run() { pack_lhs(lhs_block_, lhs_mapper_, depth_, rows_); }

 private:
  const int64 rows_;
  const int64 depth_;
  T* lhs_block_;
  LhsMapper lhs_mapper_;
};

// Packs transformed filter stored in 'filter_transform_data' into
// 'packed_filters' to be used by GemmState.
template <typename T>
struct PackFilters {
  void operator()(OpKernelContext* ctx, const Conv2DArgs& args,
                  const int64 tile_spatial_size, const int64 filter_shards_row,
                  const int64 filter_shards_col, const T* filter_transform_data,
                  std::vector<Tensor>* packed_filters) {
    const int64 in_depth = args.in_depth;
    const int64 out_depth = args.out_depth;
    const int64 num_filters = filter_shards_row * filter_shards_col * out_depth;

    auto shard = [&ctx, &packed_filters, &filter_transform_data,
                  &tile_spatial_size, &in_depth, &out_depth, &filter_shards_row,
                  &filter_shards_col, &num_filters](int64 start, int64 limit) {
      const int64 filter_coord_stride = num_filters * in_depth;
      for (int64 i = start; i < limit; ++i) {
        // Allocate filter buffer [out_depth, shard_rows, shard_cols, in_depth].
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                    TensorShape({out_depth, filter_shards_row,
                                                 filter_shards_col, in_depth}),
                                    &(*packed_filters)[i]));
        T* packed_filter = (*packed_filters)[i].template flat<T>().data();
        // Pack filters.
        GemmFilterPacker<T> packer(
            num_filters, in_depth,
            filter_transform_data + i * filter_coord_stride, packed_filter);
        packer.Run();
      }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, tile_spatial_size,
          num_filters * in_depth, shard);
  }
};

// Computes the product of filters stored in 'lhs_block' and input tiles
// stored in 'rhs_block', storing output in 'out_buffer'.
//
// Data layout for 'lhs_block':
//   [out_depth, shard_rows, shard_cols, in_depth].
//
// Data layout for 'rhs_block':
//   [num_tiles, in_depth]
//
// Data layout for 'out_buffer':
//   [num_tiles, out_depth, shard_rows, shard_cols]

template <typename T>
class GemmState {
 public:
  typedef Eigen::internal::const_blas_data_mapper<T, int64, Eigen::ColMajor>
      RhsMapper;
  typedef Eigen::internal::blas_data_mapper<T, int64, Eigen::ColMajor>
      OutputMapper;
  typedef Eigen::internal::gebp_traits<T, T> Traits;

  Eigen::internal::gemm_pack_rhs<T, int64, RhsMapper, Traits::nr,
                                 Eigen::ColMajor>
      pack_rhs;
  Eigen::internal::gebp_kernel<T, T, int64, OutputMapper, Traits::mr,
                               Traits::nr, false, false>
      gebp;

  GemmState(const int64 rows, const int64 cols, const int64 depth,
            const int64 out_buffer_size, const T* lhs_block, const T* rhs_input,
            T* rhs_block, T* out_buffer)
      : rows_(rows),
        cols_(cols),
        depth_(depth),
        out_buffer_size_(out_buffer_size),
        lhs_block_(lhs_block),
        rhs_block_(rhs_block),
        out_buffer_(out_buffer),
        rhs_mapper_(rhs_input, depth_),
        out_mapper_(out_buffer, rows_) {}

  void PackRhs() { pack_rhs(rhs_block_, rhs_mapper_, depth_, cols_); }

  void Compute() {
    memset(out_buffer_, 0, sizeof(T) * out_buffer_size_);
    gebp(out_mapper_, lhs_block_, rhs_block_, rows_, depth_, cols_, 1.0);
  }

 private:
  const int64 rows_;
  const int64 cols_;
  const int64 depth_;
  const int64 out_buffer_size_;
  const T* lhs_block_;
  T* rhs_block_;
  T* out_buffer_;
  RhsMapper rhs_mapper_;
  OutputMapper out_mapper_;
};

// Copies an input tile from 'input' into 'tile_buffer'.
//
// input:
//   [in_rows, in_cols, in_depth]
//
// tile_buffer:
//   [tile_rows, tile_cols, num_tiles, in_depth]

template <typename T>
struct CopyInputTile {
  void operator()(const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform,
                  const int64 num_tiles, const int64 in_r_start,
                  const int64 in_c_start, const T* input, T* tile_buffer) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 tile_rows = transform->input_shape().rows;
    const int64 tile_cols = transform->input_shape().cols;
    const int64 coord_stride = num_tiles * args.in_depth;

    // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
    const int64 input_vectorized_size =
        (args.in_depth / kPacketSize) * kPacketSize;
    const int64 input_scalar_size = args.in_depth % kPacketSize;

    for (int64 r = 0; r < tile_rows; ++r) {
      const int64 in_r = in_r_start + r;
      if (in_r < 0 || in_r >= args.in_rows) continue;

      for (int64 c = 0; c < tile_cols; ++c) {
        const int64 in_c = in_c_start + c;
        if (in_c < 0 || in_c >= args.in_cols) continue;

        auto* in = input + (in_r * args.in_cols + in_c) * args.in_depth;
        auto* tile = tile_buffer + coord_stride * (r * tile_rows + c);
        // Copy vectorized portion of depth dimension.
        for (int64 d = 0; d < input_vectorized_size; d += kPacketSize) {
          auto v = Eigen::internal::ploadu<Packet>(in + d);
          Eigen::internal::pstoreu<T>(tile, v);
          tile += kPacketSize;
        }
        // Copy scalar portion of inner dimension.
        for (int64 d = 0; d < input_scalar_size; ++d) {
          tile[d] = in[input_vectorized_size + d];
        }
      }
    }
  }
};

// Transforms 'num_tiles' tiles from 'input' by 'transform_matrix', storing the
// final result in 'tile_transform'.
// Intermediate results are stored in 'tile_buffer'.
//
// input:
//   [in_rows, in_cols, in_depth]
// tile_buffer:
//   [tile_rows, tile_cols, num_tiles, in_depth]
// tile_transform_matrix:
//   [tile_spatial_size, tile_spatial_size]
// tile_transform:
//   [tile_rows, tile_cols, num_tiles, in_depth]

template <typename T>
struct TransformInputTiles {
  typedef Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      MatrixMap;
  typedef Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      ConstMatrixMap;

  void operator()(const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform,
                  const int64 num_tiles, const int64 in_r_start,
                  const int64 in_c_start, const T* input,
                  const T* transform_matrix, T* tile_buffer,
                  T* tile_transform) {
    const int64 tile_rows = transform->input_shape().rows;
    const int64 tile_cols = transform->input_shape().cols;
    const int64 tile_spatial_size = tile_rows * tile_cols;
    const int64 tile_stride_cols = transform->output_shape().cols;
    const int64 coord_stride = num_tiles * args.in_depth;
    const int64 num_tiles_stride = args.in_depth;

    memset(tile_buffer, 0, sizeof(T) * tile_spatial_size * coord_stride);
    const int64 in_r = in_r_start;
    for (int64 t = 0; t < num_tiles; ++t) {
      const int64 num_tiles_base = t * num_tiles_stride;
      const int64 in_c = in_c_start + t * tile_stride_cols;
      CopyInputTile<T>()(args, transform, num_tiles, in_r, in_c, input,
                         tile_buffer + num_tiles_base);
    }

    ConstMatrixMap A(transform_matrix, tile_spatial_size, tile_spatial_size);
    ConstMatrixMap B(tile_buffer, tile_spatial_size, coord_stride);
    MatrixMap C(tile_transform, tile_spatial_size, coord_stride);

    C.noalias() = A * B;
  }
};

// Transforms output tiles from buffer by 'out_transform_matrix', storing
// final result in 'output' (intermediate results stored in 'out_buffer').
//
// out_buffer:
//   [tile_rows, tile_cols, num_tiles, out_depth, shard_rows, shard_cols]
//
// output transform buffer:
//  [out_tile_rows, out_tile_cols, num_tiles, out_depth, shard_rows, shard_cols]
//
// output:
//   [out_rows, out_cols, out_depth]
//

template <typename T>
struct TransformOutputTile {
  typedef Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      MatrixMap;
  typedef Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      ConstMatrixMap;

  void operator()(const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform,
                  const int64 num_tiles, const int64 in_r, const int64 in_c,
                  const int64 filter_shards_row, const int64 filter_shards_col,
                  const T* out_transform_matrix, const T* out_buffer,
                  T* out_transform_buffer, T* output) {
    const int64 tile_rows = transform->input_shape().rows;
    const int64 tile_cols = transform->input_shape().cols;
    const int64 tile_spatial_size = tile_rows * tile_cols;

    const int64 out_buf_stride =
        num_tiles * args.out_depth * filter_shards_row * filter_shards_col;

    const int64 out_tile_rows = transform->output_shape().rows;
    const int64 out_tile_cols = transform->output_shape().cols;
    const int64 out_tile_spatial_size = out_tile_rows * out_tile_cols;

    // Compute output transform.
    ConstMatrixMap A(out_transform_matrix, out_tile_spatial_size,
                     tile_spatial_size);
    ConstMatrixMap B(out_buffer, tile_spatial_size, out_buf_stride);
    MatrixMap C(out_transform_buffer, out_tile_spatial_size, out_buf_stride);

    C.noalias() = A * B;

    const int64 tile_stride_rows = transform->output_shape().rows;
    const int64 tile_stride_cols = transform->output_shape().cols;

    const int64 out_depth_stride = filter_shards_row * filter_shards_col;
    const int64 num_tiles_stride = args.out_depth * out_depth_stride;

    // Copy transformed output from 'out_transform_buffer' to proper index
    // in 'output'. Note that some outputs at boundaries can be discarded.
    for (int64 t = 0; t < num_tiles; ++t) {
      const int64 tile_base = t * num_tiles_stride;

      for (int64 od = 0; od < args.out_depth; ++od) {
        const int64 out_depth_base = od * out_depth_stride;

        // TODO(andydavis) Update filter sharding scheme in the next CL.
        for (int64 sr = 0; sr < filter_shards_row; ++sr) {
          for (int64 sc = 0; sc < filter_shards_col; ++sc) {
            const int64 shard_base = sr * filter_shards_col + sc;
            const int64 out_buf_base = tile_base + out_depth_base + shard_base;

            // Calcuate output indices and outputs to drop (if needed).
            const int64 out_r_start =
                in_r + args.pad_rows - sr * tile_stride_rows;
            // NOTE: The index 't' for 'num_tiles is used in index calculation
            // for 'out_c_start' because we 'num_tiles' progresses along the
            // column dimension.
            const int64 out_c_start = (in_c + t * tile_stride_cols) +
                                      args.pad_cols - sc * tile_stride_cols;

            if (out_r_start < 0 || out_r_start >= args.out_rows ||
                out_c_start < 0 || out_c_start >= args.out_cols) {
              continue;  // Skip un-needed outputs.
            }

            // Increment output if not first filter shard.
            const bool inc_output = (sr == 0 && sc == 0) ? false : true;

            for (int64 ot_row = 0; ot_row < out_tile_rows; ++ot_row) {
              const int64 out_r = out_r_start + ot_row;
              if (out_r >= args.out_rows) continue;

              for (int64 ot_col = 0; ot_col < out_tile_cols; ++ot_col) {
                const int64 out_c = out_c_start + ot_col;
                if (out_c >= args.out_cols) continue;

                // Calculate out tile indexl
                const int64 out_buf_index = ot_row * out_tile_cols + ot_col;
                // Read output value from buffer.
                const T out_val =
                    out_transform_buffer[out_buf_base +
                                         out_buf_index * out_buf_stride];
                // Calculate output index.
                const int64 output_index =
                    args.out_depth * (out_r * args.out_cols + out_c) + od;
                // Update output.
                if (inc_output) {
                  output[output_index] += out_val;
                } else {
                  output[output_index] = out_val;
                }
              }
            }
          }
        }
      }
    }
  }
};

template <typename T>
struct Conv2DState {
  Conv2DState(const int64 tile_spatial_size, const int64 filter_shards_row,
              const int64 filter_shards_col, const T* input,
              const T* tile_transform_matrix, const T* output_transform_matrix,
              T* buffer1, T* buffer2, T* packed_tile_buffer,
              T* gemm_output_buffer)
      : tile_spatial_size(tile_spatial_size),
        filter_shards_row(filter_shards_row),
        filter_shards_col(filter_shards_col),
        input(input),
        tile_transform_matrix(tile_transform_matrix),
        output_transform_matrix(output_transform_matrix),
        buffer1(buffer1),
        buffer2(buffer2),
        packed_tile_buffer(packed_tile_buffer),
        gemm_output_buffer(gemm_output_buffer) {}

  const int64 tile_spatial_size;
  const int64 filter_shards_row;
  const int64 filter_shards_col;
  const T* input;
  const T* tile_transform_matrix;
  const T* output_transform_matrix;
  T* buffer1;
  T* buffer2;
  T* packed_tile_buffer;
  T* gemm_output_buffer;
};

// Computes Conv2D for 'num_tiles' input tiles from 'input' starting at
// (in_r, in_c), storing the results of the computation in 'output'.
// Details:
// *) Transforms 'num_tiles' input tiles into 'tile_transform_buffer'.
// *) Computes point-wise MatMuls of 'num_tiles' input tiles with all filters.
// *) Transforms output tiles, and stores result to 'output'.

// TODO(andydavis) Maybe pass Conv2DState into TransformInput/Output functions.
template <typename T>
struct ComputeConv2D {
  void operator()(const Conv2DArgs& args,
                  const DeepConv2DTransform<T>* transform,
                  const Conv2DState<T>& cs, const int64 in_r, const int64 in_c,
                  const int64 num_tiles,
                  const std::vector<Tensor>& packed_filters, const T* input,
                  T* output) {
    // Transform input tiles.
    TransformInputTiles<T>()(args, transform, num_tiles, in_r, in_c, input,
                             cs.tile_transform_matrix, cs.buffer1, cs.buffer2);

    // Compute element-wise product (each a MatMul): input tiles X filters.
    const int64 in_depth = args.in_depth;
    const int64 out_depth = args.out_depth;
    const int64 num_filters =
        cs.filter_shards_row * cs.filter_shards_col * out_depth;
    const int64 tile_coord_stride = num_tiles * in_depth;
    const int64 gemm_out_buf_size = num_tiles * num_filters;
    const int64 gemm_out_buf_bytes = gemm_out_buf_size * sizeof(T);

    for (int64 i = 0; i < cs.tile_spatial_size; ++i) {
      GemmState<T> gemm(num_filters, num_tiles, in_depth, gemm_out_buf_size,
                        packed_filters[i].template flat<T>().data(),
                        cs.buffer2 + i * tile_coord_stride,
                        cs.packed_tile_buffer, cs.gemm_output_buffer);
      // Pack tile buffer.
      gemm.PackRhs();
      // Compute product.
      gemm.Compute();
      // Copy to larger output buffer without alignment requirements.
      memcpy(cs.buffer1 + i * gemm_out_buf_size, cs.gemm_output_buffer,
             gemm_out_buf_bytes);
    }

    // Transform output.
    TransformOutputTile<T>()(args, transform, num_tiles, in_r, in_c,
                             cs.filter_shards_row, cs.filter_shards_col,
                             cs.output_transform_matrix, cs.buffer1, cs.buffer2,
                             output);
  }
};

namespace functor {

// Conv2D operation specialized for deep convolutions (i.e. large
// in_depth * out_depth).
// Details:
// *) Transforms and packs filters from 'filter' in parallel.
// *) Computes Conv2D parallelized across 'batch' dimension.
//   *) Each thread loops over images in its batch shard, copying 'num_tiles'
//      input tiles into a local buffer, and computing the Conv2D output of
//      these tiles by all filters.

// TODO(andydavis) Improve the performance of boundary cases where the input
// tile extends past the limit, and wasted outputs are computed. This overhead
// is at most 2/n, where 'n' is the max(out_rows, out_cols), and so is worse
// for smaller spatial sizes.
// TODO(andydavis) Improve the performance of sharded filters.
template <typename T>
struct DeepConv2D<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, const Conv2DArgs& args, const T* input,
                  const T* filter, T* output) {
    // TODO(andydavis) Add function to select transform based on conv params.
    std::unique_ptr<DeepConv2DTransform<T>> transform(new WinogradTransform<T>);

    const int64 in_depth = args.in_depth;
    const int64 out_depth = args.out_depth;

    const int64 tile_rows = transform->input_shape().rows;
    const int64 tile_cols = transform->input_shape().cols;
    const int64 tile_spatial_size = tile_rows * tile_cols;

    const int64 out_tile_rows = transform->output_shape().rows;
    const int64 out_tile_cols = transform->output_shape().cols;
    const int64 out_tile_spatial_size = out_tile_rows * out_tile_cols;

    const int64 base_filter_rows = transform->filter_shape().rows;

    const int64 filter_residual_row =
        std::max(int64{0}, args.filter_rows - base_filter_rows);
    const int64 filter_shards_row = 1 + (filter_residual_row + 2 - 1) / 2;

    const int64 filter_residual_col =
        std::max(int64{0}, args.filter_cols - base_filter_rows);
    const int64 filter_shards_col = 1 + (filter_residual_col + 2 - 1) / 2;

    // Allocate buffer for transformed filters.
    Tensor filter_transform;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DataTypeToEnum<T>::value,
                 TensorShape({tile_rows, tile_cols, out_depth,
                              filter_shards_row, filter_shards_col, in_depth}),
                 &filter_transform));
    T* filter_transform_data = filter_transform.template flat<T>().data();

    // Transform filters.
    TransformFilters<T>()(ctx, args, transform.get(), filter_shards_row,
                          filter_shards_col, filter, filter_transform_data);

    // Pack filters.
    std::vector<Tensor> packed_filters(tile_spatial_size);
    PackFilters<T>()(ctx, args, tile_spatial_size, filter_shards_row,
                     filter_shards_col, filter_transform_data, &packed_filters);

    // Allocate buffer for tile transform matrix.
    Tensor tile_transform_matrix_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::value,
                            TensorShape({tile_spatial_size, tile_spatial_size}),
                            &tile_transform_matrix_tensor));
    T* tile_transform_matrix =
        tile_transform_matrix_tensor.template flat<T>().data();
    transform->GetInputTransformMatrix(tile_spatial_size, tile_spatial_size,
                                       tile_transform_matrix);

    // Allocate buffer for output transform matrix.
    Tensor output_transform_matrix_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           TensorShape({out_tile_spatial_size,
                                                        tile_spatial_size}),
                                           &output_transform_matrix_tensor));
    T* output_transform_matrix =
        output_transform_matrix_tensor.template flat<T>().data();
    transform->GetOutputTransformMatrix(
        out_tile_spatial_size, tile_spatial_size, output_transform_matrix);

    auto shard = [&ctx, &args, &transform, &packed_filters, &in_depth,
                  out_depth, tile_rows, tile_cols, out_tile_rows, out_tile_cols,
                  filter_shards_row, filter_shards_col, tile_spatial_size,
                  &input, &tile_transform_matrix, &output_transform_matrix,
                  &output](int64 batch_start, int64 batch_limit) {
      const int64 row_tiles =
          (args.out_rows + out_tile_rows - 1) / out_tile_rows +
          filter_shards_row - 1;
      const int64 col_tiles =
          (args.out_cols + out_tile_cols - 1) / out_tile_cols +
          filter_shards_col - 1;

      // Calculate number of tiles to process together.
      const int64 filter_shard_size = filter_shards_row * filter_shards_col;
      const int64 out_tile_spatial_size = out_tile_rows * out_tile_cols;

      // Cache budget (based on L2 cache size = 256KB).
      // TODO(andydavis) Read cache size from the system.
      const int64 cache_size = (256LL << 10) / sizeof(T);

      // Fixed costs.
      const int64 tile_transform_matrix_size =
          tile_spatial_size * tile_spatial_size;
      const int64 output_transform_matrix_size =
          out_tile_spatial_size * tile_spatial_size;
      // Calculate cache reserve size.
      const int64 filter_depth_size = in_depth * out_depth * filter_shard_size;
      const bool small_filter = ((filter_depth_size * 100) / cache_size) <= 25;
      const int64 cache_reserve_size = small_filter ? filter_depth_size : 1024;
      // Calculate total fixed cost.
      const int64 total_fixed_cost = tile_transform_matrix_size +
                                     output_transform_matrix_size +
                                     cache_reserve_size;

      // Per-tile costs.
      const int64 buffer1_per_tile_size =
          tile_spatial_size * std::max(in_depth, out_depth * filter_shard_size);
      const int64 buffer2_per_tile_size =
          std::max(tile_spatial_size * in_depth,
                   out_tile_spatial_size * out_depth * filter_shard_size);
      const int64 packed_tile_per_tile_size = in_depth;
      const int64 gemm_out_per_tile_size = out_depth * filter_shard_size;
      const int64 total_per_tile_cost =
          buffer1_per_tile_size + buffer2_per_tile_size +
          packed_tile_per_tile_size + gemm_out_per_tile_size;

      const int64 num_tiles_cache = std::max(
          int64{4}, (cache_size - total_fixed_cost) / total_per_tile_cost);
      const int64 num_tiles = std::min(num_tiles_cache, col_tiles);

      // Allocate temporary buffer 'buffer1', which is first used for copying
      // input tiles, then re-used to buffer gemm output. Calculate the
      // required buffer size for 'buffer1', based on max buffer size required
      // between copying input tiles and buffering gemm product output.
      //   buffer1: [max(buf1_tile_size, buf1_out_size)]
      const int64 buffer1_tile_size = tile_spatial_size * num_tiles * in_depth;
      const int64 buffer1_out_size =
          tile_spatial_size * num_tiles * out_depth * filter_shard_size;
      const int64 buffer1_size = std::max(buffer1_tile_size, buffer1_out_size);
      Tensor buffer1_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({buffer1_size}),
                                             &buffer1_tensor));
      T* buffer1 = buffer1_tensor.template flat<T>().data();

      // Allocate temporary buffer 'buffer2', which is first used for
      // transformed input tiles, then re-used for transformed output tiles.
      // Calculate required buffer size for 'buffer2' as max required buffer
      // between input and output transform buffer sizes.
      const int64 buffer2_tile_transform_size =
          tile_spatial_size * num_tiles * in_depth;
      const int64 buffer2_out_transform_size =
          out_tile_spatial_size * num_tiles * out_depth * filter_shard_size;
      const int64 buffer2_size =
          std::max(buffer2_tile_transform_size, buffer2_out_transform_size);
      Tensor buffer2_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({buffer2_size}),
                                             &buffer2_tensor));
      T* buffer2 = buffer2_tensor.template flat<T>().data();

      // Allocate temporary buffer to store packed tiles for one coordinate.
      // packed tile buffer: [num_tiles, in_depth].
      Tensor packed_tile_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({num_tiles, in_depth}),
                                             &packed_tile_tensor));
      T* packed_tile_buffer = packed_tile_tensor.template flat<T>().data();

      // Allocate temporary buffer for gemm output.
      // gemm output buffer [num_tiles, out_depth, shard_rows, shard_cols].
      Tensor gemm_output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape({num_tiles, out_depth,
                                                          filter_shards_row,
                                                          filter_shards_col}),
                                             &gemm_output_tensor));
      T* gemm_output_buffer = gemm_output_tensor.template flat<T>().data();

      // Capture state needed for ComputeConv2D inner loop.
      Conv2DState<T> conv_state(tile_spatial_size, filter_shards_row,
                                filter_shards_col, input, tile_transform_matrix,
                                output_transform_matrix, buffer1, buffer2,
                                packed_tile_buffer, gemm_output_buffer);

      const int64 row_pad = args.pad_rows;
      const int64 col_pad = args.pad_cols;
      const int64 unroll_col_limit = (col_tiles / num_tiles) * num_tiles;

      const int64 input_image_size = args.in_rows * args.in_cols * in_depth;
      const int64 output_image_size = args.out_rows * args.out_cols * out_depth;

      const int64 tile_stride_rows = transform->output_shape().rows;
      const int64 tile_stride_cols = transform->output_shape().cols;

      for (int64 b = batch_start; b < batch_limit; ++b) {
        const int64 in_base = b * input_image_size;
        const int64 out_base = b * output_image_size;

        for (int64 tile_r = 0; tile_r < row_tiles; ++tile_r) {
          const int64 in_r = tile_r * tile_stride_rows - row_pad;

          // Process unrolled tiles.
          for (int64 tile_c = 0; tile_c < unroll_col_limit;
               tile_c += num_tiles) {
            const int64 in_c = tile_c * tile_stride_cols - col_pad;
            ComputeConv2D<T>()(args, transform.get(), conv_state, in_r, in_c,
                               num_tiles, packed_filters, input + in_base,
                               output + out_base);
          }
          // Process remaining tiles.
          if (unroll_col_limit < col_tiles) {
            const int64 rem_tiles = col_tiles - unroll_col_limit;
            const int64 in_c = unroll_col_limit * tile_stride_cols - col_pad;
            ComputeConv2D<T>()(args, transform.get(), conv_state, in_r, in_c,
                               rem_tiles, packed_filters, input + in_base,
                               output + out_base);
          }
        }
      }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost = args.out_rows * args.out_cols * args.out_depth *
                             tile_spatial_size * args.in_depth;
    Shard(worker_threads.num_threads, worker_threads.workers, args.batch,
          shard_cost, shard);
  }
};

}  // namespace functor

template struct functor::DeepConv2D<CPUDevice, float>;

}  // namespace tensorflow
