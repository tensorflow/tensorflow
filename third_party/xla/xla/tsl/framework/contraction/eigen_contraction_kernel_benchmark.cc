/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <memory>

#include "Eigen/ThreadPool"
#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/tsl/platform/test_benchmark.h"

namespace Eigen {
namespace {

using TensorIndex = Eigen::Index;
using Eigen::Tensor;

template <typename LhsType, typename RhsType, typename OutType>
static void BM_Contraction(benchmark::State& state, int M, int K, int N,
                           int num_threads) {
  std::unique_ptr<Eigen::ThreadPool> tp;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
  if (num_threads > 1) {
    tp = std::make_unique<Eigen::ThreadPool>(num_threads);
    device = std::make_unique<Eigen::ThreadPoolDevice>(tp.get(), num_threads);
  }

  Eigen::DSizes<TensorIndex, 2> lhs_dims(M, K);
  Eigen::DSizes<TensorIndex, 2> rhs_dims(K, N);
  Eigen::DSizes<TensorIndex, 2> out_dims(M, N);

  Tensor<LhsType, 2> lhs(lhs_dims);
  Tensor<RhsType, 2> rhs(rhs_dims);
  Tensor<OutType, 2> out(out_dims);

  lhs.setRandom();
  rhs.setRandom();

  out.setZero();

  using DimPair = typename Tensor<LhsType, 2>::DimensionPair;
  std::array<DimPair, 1> dims({DimPair(1, 0)});

  if (num_threads > 1 && device != nullptr) {
    for (auto s : state) {
      out.device(*device) = lhs.contract(rhs, dims);
    }
  } else {
    for (auto s : state) {
      out = lhs.contract(rhs, dims);
    }
  }

  state.SetItemsProcessed(static_cast<int64_t>(M) * K * N * 2 *
                          state.iterations());
}

#define BM_CONTRACTION(M, K, N, THREADS, LhsT, RhsT, OutT)                     \
  static void BM_Contraction_##LhsT##_##RhsT##_##M##_##K##_##N##_##THREADS##T( \
      benchmark::State& state) {                                               \
    BM_Contraction<LhsT, RhsT, OutT>(state, M, K, N, THREADS);                 \
    state.SetLabel(#LhsT "x" #RhsT "->" #OutT);                                \
  }                                                                            \
  BENCHMARK(BM_Contraction_##LhsT##_##RhsT##_##M##_##K##_##N##_##THREADS##T)   \
      ->UseRealTime()

// Float benchmarks
BM_CONTRACTION(512, 512, 512, 1, float, float, float);
BM_CONTRACTION(512, 512, 512, 4, float, float, float);
BM_CONTRACTION(128, 1024, 1024, 1, float, float, float);
BM_CONTRACTION(128, 1024, 1024, 4, float, float, float);
BM_CONTRACTION(1, 128, 128, 1, float, float, float);
BM_CONTRACTION(1, 13522, 80, 1, float, float, float);
BM_CONTRACTION(1, 13522, 80, 4, float, float, float);

}  // namespace
}  // namespace Eigen
