/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_DATA_H_
#define XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_DATA_H_

namespace xla {
namespace gpu {

// The data below is obtained with
// xla/service/gpu/model:hlo_op_profiler_run

constexpr char kDeviceHloOpProfiles[] = R"pb(
  entries {
    key: "sm_90"  # "NVIDIA H100 80GB HBM3"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 356
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 364
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 297
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 71
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 685
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 253
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 300
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 304
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 126
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 71
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 629
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 253
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 201
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 997
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 102
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 217
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 182
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 245
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 993
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 502
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 451
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 43
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 526
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 178
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 978
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 79
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 190
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 166
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 229
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 75
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 958
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 75
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 467
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 431
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 19
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 510
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 586
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 558
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 376
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 712
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 815
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1259
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 277
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 554
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 332
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 431
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 930
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 526
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 2205
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 2415
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 641
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 2055
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 756
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 633
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 3148
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 2324
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 4344
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 2379
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 6462
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 498
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 79
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 5532
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 1750
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1342
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1275
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 2455
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2403
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 5500
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 1999
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 6636
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 4613
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 13131
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 2280
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 39
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 8363
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 15
      }
    }
  }

  entries {
    key: "sm_86"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 370
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 392
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 367
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 396
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 306
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 918
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 601
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 306
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 388
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 302
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 399
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 115
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 838
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 604
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 925
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 691
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 108
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 396
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 266
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 284
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 226
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 212
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 482
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 975
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 867
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 662
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 86
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 381
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 244
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 262
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 176
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 75
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 662
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 75
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 190
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 486
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 925
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 6339
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 1717
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 1652
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1900
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 608
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 2073
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 2412
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 698
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 1789
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 986
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 1609
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 3747
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 2016
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 5511
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 1360
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 1400
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 950
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 842
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 2383
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 3193
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 5353
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 687
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 3351
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 6613
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 4028
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 4161
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 7599
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 6962
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 11318
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 5878
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 15606
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 9939
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 97
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 39027
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 7941
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 270
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 18205
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 97
      }
    }
  }

  entries {
    key: "sm_80"  # "NVIDIA A100-SXM4-40GB"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 417
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 468
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 1094
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 420
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 417
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 391
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 454
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 908
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 744
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 1195
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 321
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 346
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 124
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 499
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 259
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 504
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 1221
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1638
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 572
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 699
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1223
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 329
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 597
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 397
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 733
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 1080
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 831
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 1861
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 1037
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 1029
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 6618
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 4131
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 2309
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 2371
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2405
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 3945
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 2284
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 5304
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 3618
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 13564
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 3037
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 6054
      }
    }
  }

  entries {
    key: "sm_70"  # "Tesla V100-SXM2-16GB"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 336
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 189
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 345
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 183
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 287
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 104
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 685
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 12
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 376
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 293
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 189
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 293
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 183
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 113
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 104
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 3
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 599
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 12
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 376
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 226
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 425
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 128
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 241
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 232
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 266
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 425
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 122
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 284
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 449
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 73
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 709
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 9
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 189
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 373
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 79
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 205
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 180
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 217
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 373
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 76
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 269
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 406
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 21
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 673
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 6
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 599
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 624
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 358
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 410
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 318
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 633
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 263
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 618
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 324
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 406
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 973
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 501
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 2099
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 780
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 722
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 703
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 758
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 654
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 3261
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 789
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 6282
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 1924
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 12
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 8151
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 480
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 42
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 8105
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 12
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 1808
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1487
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1334
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 1805
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1618
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 7261
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 2013
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 8237
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 6343
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 15
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 15355
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 2423
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 45
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 9810
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 15
      }
    }
  }

  entries {
    key: "sm_60"  # "Tesla P100-SXM2-16GB"
    value {
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 426
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S8 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 216
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 420
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S16 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 216
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 444
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S32 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S32 }
        }
        clock_cycles: 417
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: S64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 1018
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 82
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 1569
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U8 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 299
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U8 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 213
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U16 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 307
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U16 }
        }
        clock_cycles: 5
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 216
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U32 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 189
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U32 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U32 }
        }
        clock_cycles: 420
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: U64 }
        }
        clock_cycles: 2
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 888
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 79
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 1548
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 233
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 532
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 142
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 364
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 325
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 373
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 100
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 497
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 100
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 458
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F16 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 675
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 68
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F16 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 1012
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F16 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 213
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 494
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 109
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 337
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 284
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 328
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 71
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 473
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 71
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 426
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F32 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 663
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 35
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 988
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F32 }
        }
        clock_cycles: 11
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 645
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 1427
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 405
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 544
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 441
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 784
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 355
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 1640
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 417
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 473
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 1169
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 565
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 2682
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 1128
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 1021
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 991
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 1107
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 994
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 2158
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 1139
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 2934
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 1883
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C64 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 16282
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 760
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 65
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 8335
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C64 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 4302
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 3665
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 3656
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 2057
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1806
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 6135
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 4169
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 8595
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 5294
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 20
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 22278
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 3194
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 65
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 17893
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 20
      }
    }
  }

  entries {
    key: "sm_75"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 360
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S8 }
        }
        clock_cycles: 336
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 357
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S16 }
        }
        clock_cycles: 339
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 296
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 979
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 495
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 293
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 334
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 290
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 336
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 118
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 812
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 515
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 792
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 815
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 132
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 342
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 239
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 239
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 262
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 126
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 794
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 123
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 175
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F16 }
        }
        clock_cycles: 414
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 74
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 1120
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 783
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 737
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 83
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 319
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 201
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 218
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 181
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 74
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 717
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 74
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 167
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 414
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 1085
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 6494
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 1800
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 1630
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1929
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 596
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1774
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 2430
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 705
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 1805
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 984
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 1535
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: F64 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 3744
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 1915
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F64 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 5538
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: F64 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 1702
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 1503
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 1474
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 835
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 737
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 2232
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 1632
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 2989
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 2263
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 4847
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 3219
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 6474
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 4962
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 4037
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 7286
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 6848
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 10748
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 5391
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 15981
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 9653
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 95
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 38206
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 8040
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 273
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 18550
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 97
      }
    }
  }
)pb";

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_DATA_H_
