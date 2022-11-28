/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/jitrt_custom_calls.h"

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cholesky.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/collectives.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/custom_call.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/fft.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/gemm.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/io_feed.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/memcpy.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/memset.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/tracing.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/runtime/cublas_lt_matmul.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/graph_launch.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

using ::xla::runtime::CustomCallAttrEncodingSet;
using ::xla::runtime::Tagged;
using ::xla::runtime::TypeIDNameRegistry;

namespace se = ::stream_executor;

// Add custom call arguments and attributes encoding for custom HLO enums and
// structs, so that we can pass them to custom calls.
void PopulateLmhloToXlaAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  PopulateConvAttrEncoding(encoding);
  PopulateFftAttrEncoding(encoding);
  PopulateDotDimsAttrEncoding(encoding);

#if GOOGLE_CUDA
  PopulateCublasLtMatmulAttrEncoding(encoding);
#endif  // GOOGLE_CUDA
}

// Populate mapping from XLA (SE) enums/structs type id to symbol names.
void PopulateXlaGpuTypeIdNames(TypeIDNameRegistry& registry) {
#if GOOGLE_CUDA
  registry.Register<Tagged<se::cuda::BlasLt::Epilogue>>(
      "__type_id_se_cublas_lt_epilogue");
#endif  // GOOGLE_CUDA

  registry.Register<Tagged<se::dnn::ActivationMode>>(
      "__type_id_se_dnn_activation");
  registry.Register<Tagged<se::fft::Type>>("__type_id_se_fft_type");

  registry.Register<Tagged<DotDimensionNumbers>>(
      "__type_id_dot_dimension_numbers");
  registry.Register<Tagged<ConvDimensionNumbers>>(
      "__type_id_conv_dimension_numbers");
  registry.Register<Tagged<ConvBackendConfig>>("__type_id_conv_backend_config");

  RegisterTracingTypeIdNames(registry);
}

void PopulateXlaGpuCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  RegisterKernelLaunchCustomCalls(registry);
  RegisterTracingCustomCalls(registry);
  RegisterFftCustomCalls(registry);
  RegisterCholeskyCustomCalls(registry);
  RegisterCollectiveCustomCalls(registry);
  RegisterGemmCustomCalls(registry);
  RegisterConvCustomCalls(registry);
  RegisterMemcpyCustomCalls(registry);
  RegisterIoFeedCustomCalls(registry);
  RegisterMemsetCustomCalls(registry);

#if GOOGLE_CUDA
  // Graph launch kernels depend on Cuda Graph API.
  RegisterGraphLaunchCustomCalls(registry);
  RegisterMatmulCustomCalls(registry);
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  RegisterCustomCall(registry);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace gpu
}  // namespace xla
