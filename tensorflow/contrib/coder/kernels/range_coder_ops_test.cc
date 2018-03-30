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

#include <memory>
#include <vector>

#include "tensorflow/contrib/coder/kernels/range_coder.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
int LogUniform(random::SimplePhilox* gen, uint32 n) {
  CHECK_GT(n, 0);

  // Split [0, n) into {0}, [1, 2), [2, 4), [4, 8), ..., [2^(m-1), n).
  const int m = Log2Ceiling(n);

  int outcome;
  do {
    // Uniform() consumes at least 32 bits per call, therefore this is somewhat
    // wasteful implementation. Since this is used only for test, we do not
    // refine this implementation further.
    const int k = gen->Uniform(m + 1) - 1;
    // If k == -1, then sample from {0}.
    // If k == 0, then sample from [1, 2).
    // If k == 1, then sample from [2, 4), ... and so on.
    if (k < 1) {
      outcome = k + 1;
    } else {
      outcome = (1 << k) + gen->Uniform(1 << k);
    }
  } while (n <= outcome);
  return outcome;
}

std::vector<int64> ComputeStrides(const TensorShape& shape) {
  std::vector<int64> stride(shape.dims());
  int64 current = 1;
  for (int i = shape.dims() - 1; i >= 0; --i) {
    stride[i] = current;
    current *= shape.dim_size(i);
  }
  return stride;
}

class RangeCoderOpsTest : public OpsTestBase {
 protected:
  Status RunEncodeOp(int precision, gtl::ArraySlice<Tensor> input,
                     Tensor* output) {
    TF_RETURN_IF_ERROR(NodeDefBuilder("encode", "RangeEncode")
                           .Input(tensorflow::FakeInput(DT_INT16))
                           .Input(tensorflow::FakeInput(DT_INT32))
                           .Attr("precision", precision)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());

    inputs_.clear();
    std::vector<Tensor> copies(input.size());
    for (int i = 0; i < input.size(); ++i) {
      copies[i] = input[i];
      inputs_.emplace_back(&copies[i]);
    }

    TF_RETURN_IF_ERROR(RunOpKernel());

    *output = *GetOutput(0);
    inputs_.clear();

    return Status::OK();
  }

  Status RunDecodeOp(int precision, gtl::ArraySlice<Tensor> input,
                     Tensor* output) {
    TF_RETURN_IF_ERROR(NodeDefBuilder("decode", "RangeDecode")
                           .Input(tensorflow::FakeInput(DT_STRING))
                           .Input(tensorflow::FakeInput(DT_INT32))
                           .Input(tensorflow::FakeInput(DT_INT32))
                           .Attr("precision", precision)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());

    inputs_.clear();
    std::vector<Tensor> copies(input.size());
    for (int i = 0; i < input.size(); ++i) {
      copies[i] = input[i];
      inputs_.emplace_back(&copies[i]);
    }

    TF_RETURN_IF_ERROR(RunOpKernel());

    *output = *GetOutput(0);
    inputs_.clear();

    return Status::OK();
  }

  void TestEncodeAndDecode(int precision, const Tensor& data,
                           const Tensor& cdf) {
    Tensor encoded;
    TF_ASSERT_OK(RunEncodeOp(precision, {data, cdf}, &encoded));

    const TensorShape& data_shape = data.shape();
    Tensor shape{DT_INT32, {data_shape.dims()}};
    for (int i = 0; i < data_shape.dims(); ++i) {
      shape.flat<int32>()(i) = data_shape.dim_size(i);
    }

    Tensor decoded;
    TF_ASSERT_OK(RunDecodeOp(precision, {encoded, shape, cdf}, &decoded));

    EXPECT_EQ(decoded.dtype(), data.dtype());
    EXPECT_EQ(decoded.shape(), data.shape());
    EXPECT_EQ(decoded.tensor_data(), data.tensor_data());
  }

  void PopulateMaxValues(random::SimplePhilox* gen, Tensor* maxvalue_tensor,
                         int min_maxvalue, int max_maxvalue) {
    const int range = max_maxvalue - min_maxvalue;
    TTypes<int16>::Flat flat = maxvalue_tensor->flat<int16>();

    for (int64 i = 0; i < flat.size(); ++i) {
      flat(i) = min_maxvalue + gen->Uniform(range);
    }
  }

  void BuildCdf(random::SimplePhilox* gen, Tensor* data_tensor,
                Tensor* cdf_tensor, const Tensor& maxvalue_tensor) {
    CHECK(TensorShapeUtils::StartsWith(cdf_tensor->shape(),
                                       maxvalue_tensor.shape()));
    CHECK_EQ(cdf_tensor->dims(), maxvalue_tensor.dims() + 1);
    const int64 chip_size = cdf_tensor->dim_size(cdf_tensor->dims() - 1);

    std::vector<int64> data_stride = ComputeStrides(data_tensor->shape());
    std::vector<int64> cdf_stride = ComputeStrides(cdf_tensor->shape());

    for (int i = 0; i < cdf_tensor->dims(); ++i) {
      if (cdf_tensor->dim_size(i) == 1) {
        cdf_stride[i] = 0;
      }
    }

    Tensor histogram_tensor{DT_INT32, cdf_tensor->shape()};
    TTypes<int16>::Flat data = data_tensor->flat<int16>();
    TTypes<int32>::Flat histogram = histogram_tensor.flat<int32>();
    TTypes<int16>::ConstFlat maxvalue = maxvalue_tensor.flat<int16>();
    histogram.setZero();

    for (int64 index = 0; index < data.size(); ++index) {
      int64 temp = index;
      int64 offset = 0;
      for (int dim = 0; dim < data_stride.size(); ++dim) {
        const int64 coord = temp / data_stride[dim];
        offset += coord * cdf_stride[dim];
        temp -= coord * data_stride[dim];
      }
      ASSERT_EQ(temp, 0);

      const int64 maxvalue_offset = offset / chip_size;
      CHECK_EQ(maxvalue_offset * chip_size, offset);
      CHECK_LT(maxvalue(maxvalue_offset) + 1, chip_size);
      const int value = LogUniform(gen, maxvalue(maxvalue_offset));
      data(index) = value;
      histogram(offset + value + 1) += 1;
    }

    cdf_tensor->flat_inner_dims<int32, 2>() =
        histogram_tensor.flat_inner_dims<int32, 2>().cumsum(1);
  }
};

TEST_F(RangeCoderOpsTest, NoBroadcast) {
  constexpr int kPrecision = 14;
  constexpr int kMaxValue = 10;

  Tensor data{DT_INT16, {1, 32, 32, 16}};
  Tensor temp{DT_INT32, {1, 1, 1, 1, kMaxValue + 2}};
  Tensor maxvalue{DT_INT16, {1, 1, 1, 1}};
  maxvalue.flat<int16>()(0) = kMaxValue;

  ASSERT_LE(data.shape().num_elements(), 1 << kPrecision);

  random::PhiloxRandom philox(random::New64(), random::New64());
  random::SimplePhilox gen(&philox);
  BuildCdf(&gen, &data, &temp, maxvalue);

  const Eigen::array<int32, 5> broadcast = {1, 32, 32, 16, 1};

  Tensor cdf{DT_INT32, {1, 32, 32, 16, kMaxValue + 2}};
  cdf.tensor<int32, 5>() = temp.tensor<int32, 5>().broadcast(broadcast);

  TestEncodeAndDecode(kPrecision, data, cdf);
}

TEST_F(RangeCoderOpsTest, Broadcast1Axis) {
  constexpr int kPrecision = 9;
  constexpr int kDimensionSize = 1 << kPrecision;
  constexpr int kMinMaxValue = 10;
  constexpr int kMaxMaxValue = 64;

  random::PhiloxRandom philox(random::New64(), random::New64());
  random::SimplePhilox gen(&philox);
  Tensor data{DT_INT16, {1, kDimensionSize, kDimensionSize}};

  Tensor maxvalue{DT_INT16, {kDimensionSize}};
  PopulateMaxValues(&gen, &maxvalue, kMinMaxValue, kMaxMaxValue);

  {
    // Axis 1.
    Tensor maxvalue1;
    ASSERT_TRUE(maxvalue1.CopyFrom(maxvalue, {1, 1, kDimensionSize}));

    Tensor cdf{DT_INT32, {1, 1, kDimensionSize, kMaxMaxValue + 2}};
    BuildCdf(&gen, &data, &cdf, maxvalue1);
    TestEncodeAndDecode(kPrecision, data, cdf);
  }

  {
    // Axis 2.
    Tensor maxvalue2;
    ASSERT_TRUE(maxvalue2.CopyFrom(maxvalue, {1, kDimensionSize, 1}));

    Tensor cdf{DT_INT32, {1, kDimensionSize, 1, kMaxMaxValue + 2}};
    BuildCdf(&gen, &data, &cdf, maxvalue2);
    TestEncodeAndDecode(kPrecision, data, cdf);
  }
}

TEST_F(RangeCoderOpsTest, Broadcast2Axes) {
  constexpr int kPrecision = 13;
  constexpr int kDimensionSize1 = 1 << (kPrecision / 2);
  constexpr int kDimensionSize2 = 1 << (kPrecision - kPrecision / 2);
  constexpr int kMinMaxValue = 10;
  constexpr int kMaxMaxValue = 64;

  random::PhiloxRandom philox(random::New64(), random::New64());
  random::SimplePhilox gen(&philox);
  Tensor maxvalue{DT_INT16, {2, 1, 1, 7}};
  PopulateMaxValues(&gen, &maxvalue, kMinMaxValue, kMaxMaxValue);

  Tensor data{DT_INT16, {2, kDimensionSize1, kDimensionSize2, 7}};
  Tensor cdf{DT_INT32, {2, 1, 1, 7, kMaxMaxValue + 2}};
  BuildCdf(&gen, &data, &cdf, maxvalue);
  TestEncodeAndDecode(kPrecision, data, cdf);
}

TEST_F(RangeCoderOpsTest, InvalidCdfShape) {
  Tensor data{DT_INT16, {3, 3}};
  Tensor cdf{DT_INT32, {3, 3}};

  Tensor unused;
  {
    const Status status = RunEncodeOp(10, {data, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("`cdf` should have one more axis"),
              string::npos);
  }

  Tensor empty{DT_STRING, {}};
  Tensor shape{DT_INT32, {2}};
  shape.vec<int32>().setValues({3, 3});
  {
    const Status status = RunDecodeOp(10, {empty, shape, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("`cdf` should have one more axis"),
              string::npos);
  }

  cdf = Tensor{DT_INT32, {3, 3, 1}};
  {
    const Status status = RunEncodeOp(10, {data, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(
        status.error_message().find("last dimension of `cdf` should be > 1"),
        string::npos);
  }
  {
    const Status status = RunDecodeOp(10, {empty, shape, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(
        status.error_message().find("last dimension of `cdf` should be > 1"),
        string::npos);
  }
}

TEST_F(RangeCoderOpsTest, DecoderShapeFn) {
  Tensor encoded_tensor{DT_STRING, {}};
  Tensor shape_tensor{DT_INT32, {3}};
  Tensor cdf_tensor{DT_INT32, {4, 6, 8, 2}};

  shape_tensor.flat<int32>().setValues({4, 6, 8});

  Graph g{OpRegistry::Global()};
  Node* encoded = test::graph::Constant(&g, encoded_tensor);
  Node* shape = test::graph::Constant(&g, shape_tensor);
  Node* cdf = test::graph::Constant(&g, cdf_tensor);
  Node* decode;
  TF_ASSERT_OK(NodeBuilder("range_decode", "RangeDecode", g.op_registry())
                   .Input(encoded)
                   .Input(shape)
                   .Input(cdf)
                   .Attr("precision", 10)
                   .Finalize(&g, &decode));

  ShapeRefiner refiner{g.versions().producer(), g.op_registry()};
  TF_ASSERT_OK(refiner.AddNode(encoded));
  TF_ASSERT_OK(refiner.AddNode(shape));
  TF_ASSERT_OK(refiner.AddNode(cdf));
  TF_ASSERT_OK(refiner.AddNode(decode));

  auto* context = refiner.GetContext(decode);
  ASSERT_NE(context, nullptr);

  ASSERT_EQ(context->num_outputs(), 1);
  auto shape_handle = context->output(0);

  ASSERT_EQ(context->Rank(shape_handle), 3);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 0)), 4);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 1)), 6);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 2)), 8);
}

TEST_F(RangeCoderOpsTest, InvalidBroadcast) {
  Tensor data{DT_INT16, {3, 3}};
  Tensor cdf{DT_INT32, {3, 2, 2}};

  Tensor unused;
  {
    const Status status = RunEncodeOp(10, {data, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("Cannot broadcast shape"),
              string::npos);
  }

  data = Tensor{DT_INT16, {3, 1}};
  cdf = Tensor{DT_INT32, {3, 3, 2}};
  Tensor empty{DT_STRING, {}};
  Tensor shape{DT_INT32, {2}};
  shape.vec<int32>().setValues({3, 1});
  {
    const Status status = RunDecodeOp(10, {empty, shape, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("Cannot broadcast shape"),
              string::npos);
  }

  std::vector<int64> shape_vector = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  data = Tensor{DT_INT16, TensorShape{shape_vector}};
  cdf = Tensor{DT_INT32, {2, 1, 2, 1, 2, 1, 2, 1, 2, 2}};
  {
    const Status status = RunEncodeOp(10, {data, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("Irregular broadcast"), string::npos);
  }

  shape = Tensor{DT_INT32, {static_cast<int64>(shape_vector.size())}};
  for (int i = 0; i < shape_vector.size(); ++i) {
    shape.flat<int32>()(i) = shape_vector[i];
  }
  {
    const Status status = RunDecodeOp(10, {empty, shape, cdf}, &unused);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("Irregular broadcast"), string::npos);
  }
}

// Benchmark -------------------------------------------------------------

// This function creates RangeEncode graph with CDF built from a separate data
// sample.
Graph* CreateRangeEncodeFullBroadcastGraph(const TensorShape& shape,
                                           int precision) {
  CHECK_EQ(shape.dims(), 4);

  constexpr int kAlphabetSize = 70;

  Tensor histogram{DT_INT32, {kAlphabetSize + 1}};
  TTypes<int32>::Vec h = histogram.vec<int32>();
  h.setConstant(1);
  h(0) = 0;

  random::PhiloxRandom philox(random::New64(), random::New64());
  random::SimplePhilox gen(&philox);
  for (int i = 0; i < (1 << precision) - kAlphabetSize; ++i) {
    const int value = LogUniform(&gen, kAlphabetSize - 1);
    h(value + 1) += 1;
  }

  Tensor cdf{DT_INT32, {1, 1, 1, 1, kAlphabetSize + 1}};
  cdf.flat<int32>() = h.cumsum(0);

  Tensor data{DT_INT16, shape};
  TTypes<int16>::Flat d = data.flat<int16>();
  for (int64 i = 0; i < d.size(); ++i) {
    d(i) = LogUniform(&gen, kAlphabetSize - 1);
  }

  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(NodeBuilder("range_encode", "RangeEncode", g->op_registry())
                  .Input(test::graph::Constant(g, data))
                  .Input(test::graph::Constant(g, cdf))
                  .Attr("precision", precision)
                  .Finalize(g, nullptr));
  return g;
}

// This function creates RangeDecode graph with CDF built from a separate data
// sample.
Graph* CreateRangeDecodeFullBroadcastGraph(const TensorShape& shape,
                                           int precision) {
  CHECK_EQ(shape.dims(), 4);

  constexpr int kAlphabetSize = 200;
  const int64 num_elements = shape.num_elements();

  Tensor histogram{DT_INT32, {kAlphabetSize + 1}};
  TTypes<int32>::Vec h = histogram.vec<int32>();
  h.setConstant(1);
  h(0) = 0;

  random::PhiloxRandom philox(random::New64(), random::New64());
  random::SimplePhilox gen(&philox);
  for (int i = 0; i < (1 << precision) - kAlphabetSize; ++i) {
    const int value = LogUniform(&gen, kAlphabetSize - 1);
    h(value + 1) += 1;
  }

  Tensor cdf_tensor{DT_INT32, {1, 1, 1, 1, kAlphabetSize + 1}};
  TTypes<int32>::Flat cdf = cdf_tensor.flat<int32>();
  cdf = h.cumsum(0);

  Tensor string_tensor{DT_STRING, TensorShape{}};
  string& sink = string_tensor.scalar<string>()();

  RangeEncoder encoder{precision};
  for (int64 i = 0; i < num_elements; ++i) {
    const int value = LogUniform(&gen, kAlphabetSize - 1);
    encoder.Encode(cdf(value), cdf(value + 1), &sink);
  }
  encoder.Finalize(&sink);

  Tensor shape_tensor{DT_INT32, {shape.dims()}};
  for (int i = 0; i < shape.dims(); ++i) {
    shape_tensor.flat<int32>()(i) = shape.dim_size(i);
  }

  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(NodeBuilder("range_decode", "RangeDecode", g->op_registry())
                  .Input(test::graph::Constant(g, string_tensor))
                  .Input(test::graph::Constant(g, shape_tensor))
                  .Input(test::graph::Constant(g, cdf_tensor))
                  .Attr("precision", precision)
                  .Finalize(g, nullptr));
  return g;
}

void RunTensorFlowBenchmark(int iters, Graph* g, int64 num_elements) {
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(1);
  opts.config.set_inter_op_parallelism_threads(1);

  testing::UseRealTime();
  test::Benchmark("cpu", g, &opts).Run(iters);

  const int64 num_items = static_cast<int64>(iters) * num_elements;
  testing::ItemsProcessed(num_items);
}

void BM_RangeEncodeFullBroadcast(int iters, int code_size) {
  constexpr int kPrecision = 14;
  const TensorShape shape = {1, code_size, code_size, 256};
  Graph* g = CreateRangeEncodeFullBroadcastGraph(shape, kPrecision);
  RunTensorFlowBenchmark(iters, g, shape.num_elements());
}

BENCHMARK(BM_RangeEncodeFullBroadcast)->Arg(32)->Arg(64);

void BM_RangeDecodeFullBroadcast(int iters, int code_size) {
  constexpr int kPrecision = 14;
  const TensorShape shape = {1, code_size, code_size, 256};
  Graph* g = CreateRangeDecodeFullBroadcastGraph(shape, kPrecision);
  RunTensorFlowBenchmark(iters, g, shape.num_elements());
}

BENCHMARK(BM_RangeDecodeFullBroadcast)->Arg(32)->Arg(64);

}  // namespace
}  // namespace tensorflow
