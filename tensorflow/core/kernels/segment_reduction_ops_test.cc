#include <functional>

#include "tensorflow/core/public/session_options.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"

namespace tensorflow {

template <typename Index>
static void BM_SegmentReduction(int iters, string reduction, Index num_rows,
                                Index num_cols, Index segment_size) {
  testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  // Create inputs
  gtl::InlinedVector<TensorValue, 4> reduction_inputs;
  TensorShape shape1({num_rows, num_cols});
  Tensor input1(DT_FLOAT, shape1);
  reduction_inputs.push_back({nullptr, &input1});

  TensorShape shape2({num_rows});
  Tensor input2(DataTypeToEnum<Index>::v(), shape2);
  test::FillFn<Index>(&input2, [&num_rows, &segment_size](Index i) -> Index {
    return std::min(i / segment_size, num_rows - 1);
  });
  reduction_inputs.push_back({nullptr, &input2});

  NodeDef reduction_node_def;
  TF_CHECK_OK(NodeDefBuilder(reduction, reduction)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DataTypeToEnum<Index>::v()))
                  .Finalize(&reduction_node_def));
  Status status;
  std::unique_ptr<OpKernel> reduction_op(CreateOpKernel(
      DEVICE_CPU, device.get(), cpu_allocator(), reduction_node_def, &status));
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &reduction_inputs;
  params.op_kernel = reduction_op.get();
  params.output_alloc_attr = [&device, &reduction_op, &params](int index) {
    AllocatorAttributes attr;
    const bool on_host =
        (reduction_op->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    return attr;
  };

  std::unique_ptr<OpKernelContext> reduction_context(
      new OpKernelContext(params));

  reduction_op->Compute(reduction_context.get());
  TF_CHECK_OK(reduction_context->status());
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete reduction_context->release_output(0).tensor;
    reduction_op->Compute(reduction_context.get());
  }
  int64 bytes_per_iter =
      static_cast<int64>(num_rows * num_cols * sizeof(float));
  testing::BytesProcessed(bytes_per_iter * iters);
}

#define BM_Reduce(O, R, C, S)                                      \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int32(int iters) { \
    BM_SegmentReduction<int32>(iters, #O, R, C, S);                \
  }                                                                \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int64(int iters) { \
    BM_SegmentReduction<int64>(iters, #O, R, C, S);                \
  }                                                                \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int32);              \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int64);

#define BM_Reduce_Arg(R, C, S)    \
  BM_Reduce(SegmentSum, R, C, S); \
  BM_Reduce(SegmentMean, R, C, S);

BM_Reduce_Arg(64, 32, 1);
BM_Reduce_Arg(4096, 128, 1);

BM_Reduce_Arg(16, 8, 2);
BM_Reduce_Arg(64, 32, 2);
BM_Reduce_Arg(4096, 32, 2);
BM_Reduce_Arg(4096, 128, 2);

static void SparseSegmentMeanGradHelper(int iters, float uniqueness, int size) {
  testing::StopTiming();
  RequireDefaultOps();
  Graph* g = new Graph(OpRegistry::Global());
  CHECK_LE(uniqueness, 1.0);
  CHECK_GT(uniqueness, 0.0);

  const int kNumIndices = size;
  Tensor indices(DT_INT32, TensorShape({kNumIndices}));
  auto indices_flat = indices.flat<int32>();
  Tensor segments(DT_INT32, TensorShape({kNumIndices}));
  auto segments_flat = segments.flat<int32>();

  int kUniqueIndices = uniqueness * kNumIndices;
  Tensor output_dim0(DT_INT32, TensorShape({}));
  output_dim0.scalar<int32>()() = kUniqueIndices;

  for (int i = 0; i < kNumIndices; ++i) {
    indices_flat(i) = (i * 31) % kUniqueIndices;
    segments_flat(i) = i * .8;
  }

  const int kDim1 = segments_flat(kNumIndices - 1) + 1;
  const int kDim2 = 128;
  Tensor input(DT_FLOAT, TensorShape({kDim1, kDim2}));
  input.flat<float>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseSegmentMeanGrad")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, indices))
                  .Input(test::graph::Constant(g, segments))
                  .Input(test::graph::Constant(g, output_dim0))
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &node));

  testing::UseRealTime();
  testing::BytesProcessed(static_cast<int64>(iters) * (kDim1 * kDim2) *
                          sizeof(float));
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
}

static void BM_SparseSegmentMeanGrad_Low(int iters, int size) {
  return SparseSegmentMeanGradHelper(iters, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High(int iters, int size) {
  return SparseSegmentMeanGradHelper(iters, 0.01, size);
}

BENCHMARK(BM_SparseSegmentMeanGrad_Low)->Arg(1000)->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High)->Arg(1000)->Arg(100000);

}  // namespace tensorflow
