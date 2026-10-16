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

#ifndef TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_CUSTOM_CALL_H_
#define TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_CUSTOM_CALL_H_
// Descriptors passed from the XLA op kernels (flash_attention_xla_op.cc) to the
// custom-call targets (flash_attention_custom_call.cc) through the custom
// call's opaque string. Both sides include this header so the layouts can't
// drift apart.
//
// seed and offset are not in the descriptors: they are runtime operands (u64
// scalars in device memory), so changing them doesn't trigger a recompile.

#include <cstdint>

namespace tensorflow {

enum FlashAttnDtype : int32_t {
  kFlashAttnF16 = 0,
  kFlashAttnBF16 = 1,
  kFlashAttnF32 = 2,
};

// "tf_flash_attention_fwd": operands [Q, K, V, seed, offset], outputs (Out,
// Stats).
// Q/K/V/Out are [B, S, H, D] and Stats is [B, H, Sq].
struct FlashAttnDesc {
  int32_t dtype;  // FlashAttnDtype
  int64_t B, H, Sq, Sk, Dqk, Dv;
  float dropout;
  float scale;
  bool causal;
};

// "tf_flash_attention_bwd": operands [Q, K, V, Out, Stats, dO, seed, offset],
// outputs (dQ, dK, dV, workspace). The workspace output is scratch memory for
// FlashAttentionGradFunctor (custom calls can't request temporaries) and is
// discarded.
struct FlashAttnBwdDesc {
  int32_t dtype;  // FlashAttnDtype
  int64_t B, H, Sq, Sk, Dqk, Dv;
  int64_t workspace_bytes;
  float dropout;
  float scale;
  bool causal;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_CUSTOM_CALL_H_
