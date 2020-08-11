/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow { 

static Graph* BM_ScalarSummaryOp(TensorShape shape, const char* tag, 
																 float value) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor tags(DT_STRING, shape);
  Tensor values(DT_FLOAT, shape);
  for (int i = 0; i < tags.NumElements(); ++i){ 
  	tags.flat<tstring>()(i) = tag; 
  	values.flat<float>()(i) = value; 
  } 
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("dummy"), "SummaryScalar")
                  .Input(test::graph::Constant(g, tags))
                  .Input(test::graph::Constant(g, values))
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &ret));
  return g;
}

// Macro used to parse initializer list for tensorshape 
#define DIMARGS(...) {__VA_ARGS__}
// Random parameters for testing
constexpr char longTagParam = "LONGTAG____________________________"; 
constexpr float largeValueParam = 2352352.2623433; 

#define BM_ScalarSummaryDev(device, dims, name, tag, value)       		\
	static void BM_ScalarSummary_##name##_##device(int iters) { 	      \
		TensorShape tensorshape(DIMARGS(dims)); 													\
		test::Benchmark(#device, BM_ScalarSummaryOp(											\
				tensorshape, #tag, value)).Run(iters); 												\
	}																																		\
	BENCHMARK(BM_ScalarSummary_##name##_##device); 

BM_ScalarSummaryDev(cpu, (5, 10, 100), Base, tag, 5.2);
// Benchmark for large shapes 
BM_ScalarSummaryDev(cpu, (500, 1000, 10000), Large_Shape, tag, 5.2);
// Benchmark for large tag tstring 
BM_ScalarSummaryDev(cpu, (5, 10, 100), Long_Tag, longTagParam, 5.2);
// Benchmark for large values 
BM_ScalarSummaryDev(cpu, (500, 1000, 10000), Large_Value, tag, largeValueParam);
} // namespace tensorflow