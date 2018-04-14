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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {
using errors::InvalidArgument;

class PmfToCdfOp : public OpKernel {
 public:
  explicit PmfToCdfOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES(
        context, 0 < precision_ && precision_ <= 16,
        InvalidArgument("`precision` must be in [1, 16]: ", precision_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& pmf_tensor = context->input(0);

    TensorShape shape = pmf_tensor.shape();
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(shape),
                InvalidArgument("`pmf` should be at least 1-D."));
    OP_REQUIRES(
        context, shape.dim_size(shape.dims() - 1) > 1,
        InvalidArgument("`pmf` size should be at least 2 in the last axis."));
    shape.set_dim(shape.dims() - 1, shape.dim_size(shape.dims() - 1) + 1);

    Tensor* cdf_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &cdf_tensor));

    auto pmf = pmf_tensor.flat_inner_dims<float, 2>();
    auto cdf = cdf_tensor->flat_inner_dims<int32, 2>();
    CHECK_EQ(pmf.dimension(0), cdf.dimension(0));
    CHECK_EQ(pmf.dimension(1) + 1, cdf.dimension(1));

    const double n = pmf.dimension(1);
    const int64 cost_per_unit = static_cast<int64>(50.0 * n * std::log2(n));
    thread::ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(
        pmf.dimension(0), cost_per_unit,
        [this, pmf, &cdf](int64 start, int64 limit) {
          const gtl::ArraySlice<float>::size_type pmf_size = pmf.dimension(1);
          for (int64 i = start; i < limit; ++i) {
            cdf(i, 0) = 0;
            PerShard({&pmf(i, 0), pmf_size}, {&cdf(i, 1), pmf_size});
          }
        });
  }

 private:
  struct Item {
    Item(int32* p, double mass) : pointer(p), mass(mass) {
      penalty = ComputeNextPenalty();
    }

    void Decrease() {
      CHECK_GT(*pointer, 1);
      --*pointer;
      penalty = ComputeNextPenalty();
    }

    friend bool operator<(const Item& lhs, const Item& rhs) {
      return lhs.penalty < rhs.penalty;
    }

    double ComputeNextPenalty() {
      if (*pointer <= 1) {
        return std::numeric_limits<double>::infinity();
      }
      return mass * (std::log2(*pointer) - std::log2(*pointer - 1));
    }

    int32* pointer;
    double mass;
    double penalty;
  };

  void PerShard(gtl::ArraySlice<float> pmf,
                gtl::MutableArraySlice<int32> cdf) const {
    CHECK_EQ(pmf.size(), cdf.size());

    const int32 normalizer = 1 << precision_;
    std::transform(pmf.begin(), pmf.end(), cdf.begin(),
                   [normalizer](float mass) {
                     int32 value = std::rint(mass * normalizer);
                     // NOTE: Consider checking if mass > 0.
                     value = std::max(value, 1);
                     return value;
                   });

    int32 sum = std::accumulate(cdf.begin(), cdf.end(), 0);
    if (sum > normalizer) {
      std::vector<Item> queue;
      queue.reserve(cdf.size());
      for (int i = 0; i < cdf.size(); ++i) {
        queue.emplace_back(&cdf[i], pmf[i]);
      }

      std::sort(queue.begin(), queue.end());
      while (sum-- > normalizer) {
        queue[0].Decrease();
        // Performs a linear search because this find_if is likely to return
        // iterator very close to the begin.
        auto iter =
            std::find_if(std::next(queue.begin()), queue.end(),
                         [&queue](const Item& rhs) { return queue[0] < rhs; });
        std::rotate(queue.begin(), std::next(queue.begin()), iter);
      }
    }
    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
  }

  int precision_;
};

REGISTER_KERNEL_BUILDER(Name("PmfToQuantizedCdf").Device(DEVICE_CPU),
                        PmfToCdfOp);
}  // namespace
}  // namespace tensorflow
