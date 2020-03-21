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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/convex_hull_op.h"

#include <memory>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ConvexHullOp : public OpKernel {
 public:
  explicit ConvexHullOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("clockwise", &clockwise_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    OP_REQUIRES(context, input.dims() == 3,
                errors::InvalidArgument("input shape must be 3-dimensional",
                                        input.shape().DebugString()));

    OP_REQUIRES(context,
                input.dim_size(1) > 0 &&
                    FastBoundsCheck(input.dim_size(1),
                                    std::numeric_limits<int64>::max()),
                errors::InvalidArgument(
                    "point number must be between 0 and max int64"));

    // TODO(musikisomorphie): the dimension of points can be extended to n > 2.
    OP_REQUIRES(context, static_cast<int32>(input.dim_size(2)) == 2,
                errors::InvalidArgument("point dimension must be 2"));

    if (!context->status().ok()) return;

    Tensor *output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    typename TTypes<T, 3>::ConstTensor points_data(input.tensor<T, 3>());
    TTypes<float, 3>::Tensor hull_data(output->tensor<float, 3>());

    functor::ConvexHull<Device, T>()(context->eigen_device<Device>(),
                                     points_data, clockwise_, hull_data);
  }

 private:
  bool clockwise_;
};

namespace {
// This part follows the corresponding opencv implmentation
int64 sklansky(const std::vector<std::vector<float>> &pts_vec,
               std::vector<int64> &idx_vec, const int64 start, int64 end,
               const int64 pts_num, const int sign0, const int sign1) {
  if (start == end || pts_vec[start] == pts_vec[end]) {
    idx_vec[0] = start;
    return 1;
  }

  int incr = end > start ? 1 : -1;
  // Initialize three indices of points
  int64 pprev = start, pcur = pprev + incr, pnext = pcur + incr;
  // Initialize the number of convex hull points
  int64 idx_sz = 3;

  idx_vec[0] = pprev;
  idx_vec[1] = pcur;
  idx_vec[2] = pnext;

  end += incr;
  while (pnext != end) {
    // Check the angle of points pprev, pcur, pnext
    float by = pts_vec[pnext][1] - pts_vec[pcur][1];
    int by_sign = (by > 0) - (by < 0);

    if (by_sign != sign0) {
      float ax = pts_vec[pcur][0] - pts_vec[pprev][0];
      float bx = pts_vec[pnext][0] - pts_vec[pcur][0];
      float ay = pts_vec[pcur][1] - pts_vec[pprev][1];
      // Check if the angle is convex
      float convex = ay * bx - ax * by;
      int convex_sign = (convex > 0) - (convex < 0);

      if (convex_sign == sign1 && (ax != 0 || ay != 0)) {
        pprev = pcur;
        pcur = pnext;
        pnext += incr;
        idx_vec[idx_sz] = pnext;
        ++idx_sz;
      } else {
        if (pprev == start) {
          pcur = pnext;
          idx_vec[1] = pcur;
          pnext += incr;
          idx_vec[2] = pnext;
        } else {
          idx_vec[idx_sz - 2] = pnext;
          pcur = pprev;
          pprev = idx_vec[idx_sz - 4];
          --idx_sz;
        }
      }
    } else {
      pnext += incr;
      idx_vec[idx_sz - 1] = pnext;
    }
  }
  return --idx_sz;
}

void convex_hull(std::vector<std::vector<float>> &pts_vec,
                 std::vector<std::vector<float>> &out_vec, const bool clockwise,
                 const int64 pts_num) {
  auto cmp_x = [](const std::vector<float> &pt0,
                  const std::vector<float> &pt1) {
    return pt0[0] != pt1[0] ? pt0[0] < pt1[0] : pt0[1] < pt1[1];
  };
  // Sort points based on coordinate x
  std::sort(pts_vec.begin(), pts_vec.end(), cmp_x);

  auto cmp_y = [](const std::vector<float> &pt0,
                  const std::vector<float> &pt1) { return pt0[1] < pt1[1]; };

  // Find the indices of points with minimum and maximum y
  int64 min_idy =
      std::min_element(pts_vec.begin(), pts_vec.end(), cmp_y) - pts_vec.begin();
  int64 max_idy =
      std::max_element(pts_vec.begin(), pts_vec.end(), cmp_y) - pts_vec.begin();

  // Number of the points forming a convex hull
  int64 out_idx = 0;
  if (pts_vec[0] == pts_vec[pts_num - 1]) {
    out_vec[out_idx++] = pts_vec[0];
  } else {
    // Upper half of the convex hull
    int64 u_st = clockwise ? 0 : pts_num - 1;
    int u_sign = clockwise ? 1 : -1;

    std::vector<int64> u_vec0(pts_num + 2);
    int64 u_cnt0 =
        sklansky(pts_vec, u_vec0, u_st, max_idy, pts_num, -1, u_sign);

    std::vector<int64> u_vec1(pts_num + 2);
    int64 u_cnt1 = sklansky(pts_vec, u_vec1, pts_num - 1 - u_st, max_idy,
                            pts_num, -1, -u_sign);

    for (int64 i = 0; i < u_cnt0 - 1; ++i) {
      out_vec[out_idx++] = pts_vec[u_vec0[i]];
    }

    for (int64 i = u_cnt1 - 1; i > 0; --i) {
      out_vec[out_idx++] = pts_vec[u_vec1[i]];
    }

    int64 stop_idx =
        u_cnt1 > 2 ? u_vec1[1] : u_cnt0 > 2 ? u_vec0[u_cnt0 - 2] : -1;

    // Lower half of the convex hull
    int64 l_st = !clockwise ? 0 : pts_num - 1;
    int l_sign = !clockwise ? -1 : 1;

    std::vector<int64> l_vec0(pts_num + 2);
    int64 l_cnt0 = sklansky(pts_vec, l_vec0, l_st, min_idy, pts_num, 1, l_sign);

    std::vector<int64> l_vec1(pts_num + 2);
    int64 l_cnt1 = sklansky(pts_vec, l_vec1, pts_num - 1 - l_st, min_idy,
                            pts_num, 1, -l_sign);

    if (stop_idx >= 0) {
      int64 check_idx = l_cnt0 > 2
                            ? l_vec0[1]
                            : l_cnt0 + l_cnt1 > 2 ? l_vec1[2 - l_cnt0] : -1;
      if (check_idx == stop_idx ||
          (check_idx >= 0 && pts_vec[check_idx] == pts_vec[stop_idx])) {
        l_cnt0 = std::min(l_cnt0, static_cast<int64>(2));
        l_cnt1 = std::min(l_cnt1, static_cast<int64>(2));
      }
    }

    for (int64 i = 0; i < l_cnt0 - 1; ++i) {
      out_vec[out_idx++] = pts_vec[l_vec0[i]];
    }

    for (int64 i = l_cnt1 - 1; i > 0; --i) {
      out_vec[out_idx++] = pts_vec[l_vec1[i]];
    }
  }

  // Pad the last point of the convex hull to the output vector
  for (int64 i = out_idx; i < pts_num; ++i) {
    out_vec[i] = out_vec[out_idx - 1];
  }
}
}  // namespace

namespace functor {
template <typename T>
struct ConvexHull<CPUDevice, T> {
  void operator()(const CPUDevice &d, typename TTypes<T, 3>::ConstTensor points,
                  const bool clockwise,
                  typename TTypes<float, 3>::Tensor output) {
    const int batch_size = points.dimension(0);
    const int64 pts_num = points.dimension(1);
    const int pts_dim = points.dimension(2);

    const int64 out_num = output.dimension(1);
    const int out_dim = output.dimension(2);

    // Handle trivial convex hull efficiently
    if (pts_num <= 2) {
      output = points.template cast<float>();
      return;
    }

    const T *pts_flat = points.data();
    float *out_flat = output.data();

    for (int b = 0; b < batch_size; ++b) {
      // Copy tensor to vector container for sorting in convex_hull()
      std::vector<std::vector<float>> pts_vec(pts_num,
                                              std::vector<float>(pts_dim));
      std::vector<std::vector<float>> out_vec(pts_num,
                                              std::vector<float>(out_dim));
      for (int64 pn = 0; pn < pts_num; ++pn) {
        int64 idx = b * pts_num * pts_dim + pn * pts_dim;
        int64 idy = idx + 1;
        float ptx(pts_flat[idx]);
        float pty(pts_flat[idy]);
        pts_vec[pn] = {ptx, pty};
      }

      // Compute convex hull for each batch
      convex_hull(pts_vec, out_vec, clockwise, pts_num);

      for (int64 pn = 0; pn < out_num; ++pn) {
        int64 idx = b * pts_num * pts_dim + pn * out_dim;
        int64 idy = idx + 1;
        out_flat[idx] = out_vec[pn][0];
        out_flat[idy] = out_vec[pn][1];
      }
    }
  }
};
}  // namespace functor

#define REGISTER_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ConvexHull").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvexHullOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
}  // namespace tensorflow
