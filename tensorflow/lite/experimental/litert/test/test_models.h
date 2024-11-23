// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_TEST_MODELS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_TEST_MODELS_H_

#include "absl/strings/string_view.h"

// Attention sub-module of a toy model.
static constexpr absl::string_view kAttentionModel = "attention.tflite";

// Attention vector einsum sub-module of a toy LLM.
static constexpr absl::string_view kAttnVecEinsumModel =
    "attn_vec_einsum.tflite";

//  Feed forward sub-module of a toy LLM.
static constexpr absl::string_view kFeedForwardModel = "ff.tflite";

// Key einsume sub-module of a toy LLM.
static constexpr absl::string_view kKeyEinsumModel = "k_einsum.tflite";

// Value einsum sub-module of a toy LLM.
static constexpr absl::string_view kValueEinsumModel = "v_einsum.tflite";

// Query einsum sub-module of a toy LLM.
static constexpr absl::string_view kQueryEinsumModel = "q_einsum.tflite";

// RMS Normalization sub-module of a toy LLM.
static constexpr absl::string_view kRMSNormModel = "norm.tflite";

// ROPE sub-module of a toy LLM.
static constexpr absl::string_view kROPEModel = "rope.tflite";

// Scale dot product attentionsub-module of a toy LLM.
static constexpr absl::string_view kSDPAModel = "sdpa.tflite";

// Transformer block sub-module of a toy LLM.
static constexpr absl::string_view kTransformerBlockModel =
    "transformer.tflite";

// Quantized model with a single mul op.
// Mul: <8x100x32x4xint16>, <8x100x32x4xint16> -> <8x100x32x4xint16>
static constexpr absl::string_view kQSimpleMul16x16Model = "mul_quant.tflite";

// Quantized model with a mul op and a add op.
// Mul: <8x100x32x4xint16>, <8x100x32x4xint16> -> <8x100x32x4xint16>
// Add: <8x100x32x4xint16>, <8x100x32x4xint16> -> <8x100x32x4xint16>
static constexpr absl::string_view kQMulAdd16x16Model =
    "simple_quantized_ops.tflite";

// Single add op i16 activations and i8 weights and dynamic shape.
static constexpr absl::string_view kQSingleDynAdd16x8Model =
    "single_add_default_a16w8_recipe_quantized.tflite";

// Single add op i8 activations and i8 weights and dynamic shape.
static constexpr absl::string_view kQSingleDynAdd8x8Model =
    "single_add_default_a8w8_recipe_quantized.tflite";

// Single mul op i16 activations and i8 weights and dynamic shape.
static constexpr absl::string_view kQSingleDynMul16x8Model =
    "single_mul_default_a16w8_recipe_quantized.tflite";

// Single mul op i8 activations and i8 weights and dynamic shape.
static constexpr absl::string_view kQSingleDynMul8x8Model =
    "single_mul_default_a8w8_recipe_quantized.tflite";

// Single rsqrt op i16 activations and i8 weights and dynamic shape.
static constexpr absl::string_view kQSingleDynRsqrt16x8Model =
    "single_rsqrt_default_a16w8_recipe_quantized.tflite";

// Single rsqrt op i8 activations and i8 weights and dynamic shape.
static constexpr absl::string_view kQSingleDynRsqrt8x8Model =
    "single_rsqrt_default_a8w8_recipe_quantized.tflite";

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_TEST_MODELS_H_
