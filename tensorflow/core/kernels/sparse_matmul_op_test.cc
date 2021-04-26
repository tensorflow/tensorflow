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

#include "tensorflow/core/kernels/sparse_matmul_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
random::PhiloxRandom philox(1, 1);
random::SimplePhilox rnd(&philox);
using Eigen::operator==;

template <typename T>
void Sparsify(Tensor* t, float sparsity) {
  const int64 N = t->NumElements();
  CHECK_LE(sparsity, 1);
  auto flat = t->flat<T>();
  if (sparsity == 1) {
    flat.setZero();
    return;
  }
  static const uint32 K = 10000;
  for (int64 i = 0; i < N; ++i) {
    if (rnd.Uniform(K) < sparsity * K) {
      flat(i) = T(0);
    } else if (flat(i) == T(0)) {
      flat(i) = T(1);
    }
  }
}

Node* SparseMatMulNode(Graph* g, Node* in0, Node* in1, bool transpose_a,
                       bool transpose_b, bool a_sparse, bool b_sparse) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseMatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Attr("a_is_sparse", a_sparse)
                  .Attr("b_is_sparse", b_sparse)
                  .Finalize(g, &ret));
  return ret;
}

template <typename TA, typename TB>
static Graph* SparseMatMulHelper(Graph* g, int m, int n, int d,
                                 float sparsity_a, float sparsity_b,
                                 bool transpose_a, bool transpose_b) {
  bool a_sparse = (sparsity_a > 0);
  bool b_sparse = (sparsity_b > 0);

  auto left_shape = transpose_a ? TensorShape({d, m}) : TensorShape({m, d});
  Tensor left(DataTypeToEnum<TA>::value, left_shape);
  left.flat<TA>().setRandom();
  Sparsify<TA>(&left, sparsity_a);

  auto right_shape = transpose_b ? TensorShape({n, d}) : TensorShape({d, n});
  Tensor right(DataTypeToEnum<TB>::value, right_shape);
  right.flat<TB>().setRandom();
  Sparsify<TB>(&right, sparsity_b);

  SparseMatMulNode(g, test::graph::Constant(g, left),
                   test::graph::Constant(g, right), transpose_a, transpose_b,
                   a_sparse, b_sparse);
  return g;
}

template <typename TA, typename TB>
static Graph* SparseMatMul(int m, int n, int d, float sparsity_a,
                           float sparsity_b, bool transpose_a,
                           bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  return SparseMatMulHelper<TA, TB>(g, m, n, d, sparsity_a, sparsity_b,
                                    transpose_a, transpose_b);
}

static Graph* ReplicatedSparseMatMul(int m, int n, int d, float sparsity_1,
                                     float sparsity_2, int copies) {
  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < copies; ++i) {
    SparseMatMulHelper<float, float>(g, m, n, d, sparsity_1, sparsity_2, false,
                                     false);
  }
  return g;
}

#define BM_SPARSE(M, K, N, S1, S2, TRA, TRB, TA, TB)                           \
  static void                                                                  \
      BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TRA##_##TRB##_##TA##_##TB( \
          ::testing::benchmark::State& state) {                                \
    auto label = strings::Printf("tr_a: %d tr_b: %d sp_a: %0.2f sp_b: %0.2f",  \
                                 TRA, TRB, S1 / 100.0, S2 / 100.0);            \
    state.SetLabel(label);                                                     \
    auto g = SparseMatMul<TA, TB>(M, N, K, S1 / 100.0, S2 / 100.0, TRA, TRB);  \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TRA##_##TRB##_##TA##_##TB) \
      ->UseRealTime();

#define BM_SPARSE_REPLICATED(M, K, N, S1, S2, Copies)                          \
  static void BM_Sparse_replicated##_##M##_##K##_##N##_##S1##_##S2##_##Copies( \
      ::testing::benchmark::State& state) {                                    \
    auto label = strings::Printf("copies: %d sp_a: %0.2f sp_b: %0.2f",         \
                                 (Copies), S1 / 100.0, S2 / 100.0);            \
    state.SetLabel(label);                                                     \
    auto g =                                                                   \
        ReplicatedSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, (Copies));     \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
    state.SetItemsProcessed(state.iterations() * M * K * N * Copies * 2);      \
  }                                                                            \
  BENCHMARK(BM_Sparse_replicated##_##M##_##K##_##N##_##S1##_##S2##_##Copies)   \
      ->UseRealTime();

#define BM_SPARSE_FLOAT(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, float, float)
#define BM_SPARSE_BFLOAT16(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, bfloat16, bfloat16)
#define BM_SPARSE_FLOAT_BFLOAT16(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, float, bfloat16)
#define BM_SPARSE_BFLOAT16_FLOAT(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, bfloat16, float)

// Test sparse b
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 50, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 99, 0, false, false);
// Test sparse a
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 50, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, false, false);
// Test transposing
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, true, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, false, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, true, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, true, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, false, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, true, true);

// Test smaller sizes
BM_SPARSE_FLOAT(1024, 1024, 1024, 0, 0, false, false);
BM_SPARSE_FLOAT(1024, 1024, 1024, 1, 0, false, false);
BM_SPARSE_FLOAT(1024, 1024, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(256, 256, 256, 1, 0, false, false);
BM_SPARSE_FLOAT(512, 512, 512, 1, 0, false, false);
BM_SPARSE_FLOAT(2560, 400, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2560, 400, 1024, 85, 0, true, false);

BM_SPARSE_FLOAT(400, 800, 2560, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 2560, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 1024, 256, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 256, 1, 85, 0, false, false);

BM_SPARSE_REPLICATED(400, 800, 2560, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 2560, 1024, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 1024, 256, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 256, 1, 85, 0, 6);

BM_SPARSE_FLOAT(2048, 1792, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 1024, 768, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 768, 512, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 512, 256, 85, 0, false, false);

BM_SPARSE_FLOAT(2049, 1792, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 1024, 768, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 768, 512, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 512, 256, 85, 0, false, false);

BM_SPARSE_REPLICATED(2048, 1792, 1024, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 1024, 768, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 768, 512, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 512, 256, 85, 0, 6);

// Test bfloat16
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 99, 0, false, false);
BM_SPARSE_BFLOAT16_FLOAT(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_BFLOAT16_FLOAT(2048, 2048, 2048, 99, 0, false, false);
BM_SPARSE_FLOAT_BFLOAT16(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_FLOAT_BFLOAT16(2048, 2048, 2048, 99, 0, false, false);

static Graph* MultiSparseMatMul(int m, int n, int d, float sparsity_1,
                                float sparsity_2, int copies) {
  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < copies; ++i) {
    SparseMatMulHelper<float, float>(g, d, n, m, sparsity_1, sparsity_2, true,
                                     false);
    SparseMatMulHelper<float, float>(g, m, d, n, sparsity_2, 0, false, true);
  }
  return g;
}

// clang-format off
// NOLINTBEGIN
#define BM_SPARSE_MULTI(M, K, N, S1, S2, Copies)                              \
  static void BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2##_##Copies(::testing::benchmark::State& state) {                                              \
    auto label = strings::Printf("%d_%d_%d_%d_%0.2f_%0.2f", M, K, N, Copies,  \
                                 S1 / 100.0, S2 / 100.0);                     \
    state.SetLabel(label);                                                    \
    auto g = MultiSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, Copies);      \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);        \
    state.SetItemsProcessed(state.iterations() * M * K * N * 2 * 2 * Copies); \
  }                                                                           \
  BENCHMARK(BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2##_##Copies)       \
      ->UseRealTime();
// NOLINTEND
// clang-format on
BM_SPARSE_MULTI(1024, 2140, 4096, 0, 82, 1);
BM_SPARSE_MULTI(1024, 4096, 2048, 83, 83, 1);
BM_SPARSE_MULTI(400, 800, 2560, 85, 85, 1);
BM_SPARSE_MULTI(400, 2560, 1024, 85, 85, 1);
BM_SPARSE_MULTI(400, 1024, 256, 85, 85, 1);
BM_SPARSE_MULTI(400, 256, 1, 85, 85, 1);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 1);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 1);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 1);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 1);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 3);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 3);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 3);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 3);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 6);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 6);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 6);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 6);

}  // end namespace tensorflow

namespace Eigen {
namespace internal {

class SparseMatmulOpTest : public ::testing::Test {
 protected:
  SparseMatmulOpTest()
      : PacketSize(Eigen::internal::packet_traits<float>::size) {
    typedef typename NumTraits<float>::Real RealFloat;

    for (int i = 0; i < kMaxPacketSize; ++i) {
      data1[i] = internal::random<float>() / RealFloat(PacketSize);
      data2[i] = internal::random<float>() / RealFloat(PacketSize);
      data3[i] = internal::random<float>() / RealFloat(PacketSize);
    }
    for (int i = kMaxPacketSize; i < kMaxPacketSize * 2; ++i) {
      data3[i] = internal::random<float>() / RealFloat(PacketSize);
    }

    // zero out lower 16-bits of mantissa of data3 values
    // copy bfloat representation to data3_bfloat16
    for (int i = 0; i < kMaxPacketSize * 2; ++i) {
      uint16_t* data3_p = reinterpret_cast<uint16_t*>(&data3[i]);
      uint16_t* data3_bfloat16_p =
          reinterpret_cast<uint16_t*>(data3_bfloat16) + i;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      data3_p[1] = 0;
      data3_bfloat16_p[0] = data3_p[0];
#else
      data3_p[0] = 0;
      data3_bfloat16_p[0] = data3_p[1];
#endif
    }
  }

  bool areApprox(const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i) {
      if (a[i] != b[i] && !internal::isApprox(a[i], b[i])) {
        auto ma = Map<const Matrix<float, 1, Dynamic> >(a, size);
        auto mb = Map<const Matrix<float, 1, Dynamic> >(b, size);
        std::cout << "[" << ma << "]"
                  << " != [" << mb << "], differences: [" << (mb - ma) << "]\n";
        return false;
      }
    }
    return true;
  }

#ifdef EIGEN_VECTORIZE_AVX512
  static const int kMaxPacketSize = 16;
#elif defined EIGEN_VECTORIZE_AVX || defined EIGEN_VECTORIZE_AVX2
  static const int kMaxPacketSize = 8;
#else
  static constexpr int kMaxPacketSize = 4;
#endif
  typedef typename Eigen::internal::packet_traits<float>::type Packet;
  const int PacketSize;
  // float values
  EIGEN_ALIGN_MAX float data1[kMaxPacketSize];
  // output of intrinsics
  EIGEN_ALIGN_MAX float data2[kMaxPacketSize];
  // float values with only 7 mantissa bits (bfloat representable)
  EIGEN_ALIGN_MAX float data3[kMaxPacketSize * 2];
  // bfloat16 representation of data3
  EIGEN_ALIGN_MAX float data3_bfloat16[kMaxPacketSize];
  EIGEN_ALIGN_MAX float ref[kMaxPacketSize];
};

TEST_F(SparseMatmulOpTest, BroadcastPacketTest) {
  for (int i = 0; i < PacketSize; ++i) ref[i] = data1[0];
  internal::pstoreu(data2, internal::pbroadcast_first<Packet>(
                               internal::ploadu<Packet>(data1)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));
  if (PacketSize > 1) {
    for (int i = 0; i < PacketSize; ++i) ref[i] = data1[1];
    internal::pstoreu(data2, internal::pbroadcast_second<Packet>(
                                 internal::ploadu<Packet>(data1)));
    ASSERT_TRUE(areApprox(ref, data2, PacketSize));

    if (PacketSize > 2) {
      for (int i = 0; i < PacketSize; ++i) ref[i] = data1[2];
      internal::pstoreu(data2, internal::pbroadcast_third<Packet>(
                                   internal::ploadu<Packet>(data1)));
      ASSERT_TRUE(areApprox(ref, data2, PacketSize));

      if (PacketSize > 3) {
        for (int i = 0; i < PacketSize; ++i) ref[i] = data1[3];
        internal::pstoreu(data2, internal::pbroadcast_fourth<Packet>(
                                     internal::ploadu<Packet>(data1)));
        ASSERT_TRUE(areApprox(ref, data2, PacketSize));
      }
    }
  }
}

TEST_F(SparseMatmulOpTest, InterleavePacketTest) {
  if (PacketSize == 8) {  // AVX
    for (int i = 0; i < PacketSize / 4; ++i) ref[i] = data1[i];
    for (int i = PacketSize / 4; i < PacketSize / 2; ++i)
      ref[i] = data1[i + PacketSize / 4];
    for (int i = PacketSize / 2; i < 3 * PacketSize / 4; ++i)
      ref[i] = data1[i - PacketSize / 4];
    for (int i = 3 * PacketSize / 4; i < PacketSize; ++i) ref[i] = data1[i];
  } else {
    // No interleaving done for smaller packets
    for (int i = 0; i < PacketSize; ++i) ref[i] = data1[i];
  }

  internal::pstoreu(data2, internal::pinterleave4x64<Packet>(
                               internal::ploadu<Packet>(data1)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));
}

TEST_F(SparseMatmulOpTest, Bfloat16ExpandTest) {
  if (PacketSize == 8) {  // AVX
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i] = data3[i];
    }
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i + PacketSize / 2] = data3[i + PacketSize];
    }
  } else {
    for (int i = 0; i < PacketSize; ++i) {
      ref[i] = data3[i];
    }
  }
  internal::pstoreu(data2, internal::pexpand_bf16_l<Packet>(
                               internal::ploadu<Packet>(data3_bfloat16)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));

  if (PacketSize == 8) {  // AVX
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i] = data3[i + PacketSize / 2];
    }
    for (int i = 0; i < PacketSize / 2; ++i) {
      ref[i + PacketSize / 2] = data3[i + 3 * PacketSize / 2];
    }
  } else {
    for (int i = 0; i < PacketSize; ++i) {
      ref[i] = data3[i + PacketSize];
    }
  }

  internal::pstoreu(data2, internal::pexpand_bf16_u<Packet>(
                               internal::ploadu<Packet>(data3_bfloat16)));
  ASSERT_TRUE(areApprox(ref, data2, PacketSize));
}

TEST_F(SparseMatmulOpTest, Bfloat16LoadTest) {
  if (PacketSize >= 4) {
    for (int i = 0; i < 4; ++i) ref[i] = data3[i];
    internal::pstoreu(data2, internal::pload4bf16<Packet>(data3_bfloat16));
    ASSERT_TRUE(areApprox(ref, data2, 4));

    internal::pstoreu(data2, internal::pload2bf16<Packet>(data3_bfloat16));
    ASSERT_TRUE(areApprox(ref, data2, 2));
  }
}

}  // namespace internal
}  // namespace Eigen
