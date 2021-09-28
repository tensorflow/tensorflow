/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h"

namespace tensorflow {
namespace {

// Row reduction
BM_TFMlir2(RowReduceDynamicAll, f32, /* num_threads */ 0,
           MlirSpec("tf.Sum", "f32", {kDynamic, kDynamic},
                    /*dims_to_reduce=*/{0}));
BM_TFMlir2(RowReduceStaticRow, f32, /* num_threads */ 0,
           MlirSpec("tf.Sum", "f32", {kStatic, kDynamic},
                    /*dims_to_reduce=*/{0}));
BM_TFMlir2(RowReduceStaticCol, f32, /* num_threads */ 0,
           MlirSpec("tf.Sum", "f32", {kDynamic, kStatic},
                    /*dims_to_reduce=*/{0}));
BM_TFMlir2(RowReduceStaticAll, f32, /* num_threads */ 0,
           MlirSpec("tf.Sum", "f32", {kStatic, kStatic},
                    /*dims_to_reduce=*/{0}));
BM_Eigen2(RowReduce, f32, /* num_threads */ 0, /*output rank=*/1,
          EigenSpec({0}));

}  // namespace
}  // namespace tensorflow
