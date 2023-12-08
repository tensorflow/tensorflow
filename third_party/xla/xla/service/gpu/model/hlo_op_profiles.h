/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_H_
#define XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_H_

namespace xla {
namespace gpu {

// The data below is obtained with
// xla/service/gpu/model:hlo_op_profiler_run

constexpr char kDeviceHloOpProfiles[] = R"pb(
  entries {
    key: "sm_90"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 351
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
        clock_cycles: 115
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 375
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
        clock_cycles: 115
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 298
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
        clock_cycles: 66
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 698
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: S64 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 238
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 308
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
        clock_cycles: 115
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 301
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
        clock_cycles: 115
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U32 }
        }
        clock_cycles: 119
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
        clock_cycles: 66
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 621
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: U64 }
        }
        clock_cycles: 10
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 238
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 466
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 329
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 105
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
        clock_cycles: 94
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F16 }
        }
        clock_cycles: 333
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 98
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 200
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
        clock_cycles: 449
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F16 }
        }
        clock_cycles: 45
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
        clock_cycles: 491
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
        clock_cycles: 400
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 326
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 80
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 196
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 157
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 221
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 77
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 933
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F32 }
        }
        clock_cycles: 77
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 179
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
        clock_cycles: 428
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F32 }
        }
        clock_cycles: 24
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: F32 }
        }
        clock_cycles: 7
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 487
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
        clock_cycles: 1656
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 568
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 382
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 403
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 800
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1210
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
        clock_cycles: 561
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 333
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 393
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
        clock_cycles: 866
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 530
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
        clock_cycles: 2179
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
        clock_cycles: 579
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C64 }
        }
        clock_cycles: 635
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 631
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 807
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 614
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 2815
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 723
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 4113
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 2348
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
        clock_cycles: 6047
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 452
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C64 }
        }
        clock_cycles: 77
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 4706
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
        clock_cycles: 1779
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1333
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1288
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 2337
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2299
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 5036
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 1997
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 6181
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 4419
      }
      entries {
        instruction {
          opcode: "add"
          shape { element_type: C128 }
        }
        clock_cycles: 14
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 12453
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 2270
      }
      entries {
        instruction {
          opcode: "multiply"
          shape { element_type: C128 }
        }
        clock_cycles: 38
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 7339
      }
      entries {
        instruction {
          opcode: "subtract"
          shape { element_type: C128 }
        }
        clock_cycles: 14
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

  entries { key: "sm_80"
            value { entries {
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
                    })pb"
                                        R"pb(
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
    key: "sm_70"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 345
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
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 954
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 302
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U8 }
        }
        clock_cycles: 526
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U16 }
        }
        clock_cycles: 309
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U16 }
        }
        clock_cycles: 544
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 749
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 820
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 1227
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F16 }
        }
        clock_cycles: 865
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 137
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 544
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 354
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 388
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
        clock_cycles: 841
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 134
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 556
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 1279
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 1168
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F32 }
        }
        clock_cycles: 823
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 110
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 514
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 333
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 361
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 529
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 660
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 1214
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1392
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 673
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 474
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 676
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 618
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1061
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 290
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F64 }
        }
        clock_cycles: 667
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 391
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 709
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 1178
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 682
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 1679
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C64 }
        }
        clock_cycles: 1762
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 1450
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 1141
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C64 }
        }
        clock_cycles: 1787
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 3935
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 7025
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 948
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 4277
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 2386
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 1881
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 1875
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 2622
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2328
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 4531
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 2408
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 5388
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 3867
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 13794
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 3001
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 6046
      }
    }
  }

  entries {
    key: "sm_60"
    value {
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S8 }
        }
        clock_cycles: 438
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S16 }
        }
        clock_cycles: 479
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S32 }
        }
        clock_cycles: 758
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: S64 }
        }
        clock_cycles: 2037
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: S64 }
        }
        clock_cycles: 2937
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: U8 }
        }
        clock_cycles: 307
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
          opcode: "divide"
          shape { element_type: U64 }
        }
        clock_cycles: 1708
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: U64 }
        }
        clock_cycles: 2993
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F16 }
        }
        clock_cycles: 1661
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F16 }
        }
        clock_cycles: 213
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 778
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F16 }
        }
        clock_cycles: 598
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F16 }
        }
        clock_cycles: 538
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F16 }
        }
        clock_cycles: 402
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: F16 }
        }
        clock_cycles: 130
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F16 }
        }
        clock_cycles: 453
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F16 }
        }
        clock_cycles: 1717
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F32 }
        }
        clock_cycles: 1672
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F32 }
        }
        clock_cycles: 168
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 731
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F32 }
        }
        clock_cycles: 435
      }
      )pb"
                                        R"pb(
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F32 }
        }
        clock_cycles: 589
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F32 }
        }
        clock_cycles: 343
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: F32 }
        }
        clock_cycles: 1024
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F32 }
        }
        clock_cycles: 417
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F32 }
        }
        clock_cycles: 873
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F32 }
        }
        clock_cycles: 1779
      }
      entries {
        instruction {
          opcode: "cbrt"
          shape { element_type: F64 }
        }
        clock_cycles: 1649
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: F64 }
        }
        clock_cycles: 1175
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: F64 }
        }
        clock_cycles: 639
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 911
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: F64 }
        }
        clock_cycles: 935
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: F64 }
        }
        clock_cycles: 1421
      }
      entries {
        instruction {
          opcode: "logistic"
          shape { element_type: F64 }
        }
        clock_cycles: 1098
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
        clock_cycles: 1187
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: F64 }
        }
        clock_cycles: 645
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: F64 }
        }
        clock_cycles: 917
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: F64 }
        }
        clock_cycles: 1394
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: F64 }
        }
        clock_cycles: 959
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: F64 }
        }
        clock_cycles: 2667
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C64 }
        }
        clock_cycles: 1726
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C64 }
        }
        clock_cycles: 1518
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 4142
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C64 }
        }
        clock_cycles: 5069
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C64 }
        }
        clock_cycles: 4053
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C64 }
        }
        clock_cycles: 9469
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C64 }
        }
        clock_cycles: 1317
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C64 }
        }
        clock_cycles: 5617
      }
      entries {
        instruction {
          opcode: "cosine"
          shape { element_type: C128 }
        }
        clock_cycles: 3416
      }
      entries {
        instruction {
          opcode: "exponential"
          shape { element_type: C128 }
        }
        clock_cycles: 2730
      }
      entries {
        instruction {
          opcode: "exponential-minus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2765
      }
      entries {
        instruction {
          opcode: "log"
          shape { element_type: C128 }
        }
        clock_cycles: 3106
      }
      entries {
        instruction {
          opcode: "log-plus-one"
          shape { element_type: C128 }
        }
        clock_cycles: 2895
      }
      entries {
        instruction {
          opcode: "rsqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 5922
      }
      entries {
        instruction {
          opcode: "sine"
          shape { element_type: C128 }
        }
        clock_cycles: 3496
      }
      entries {
        instruction {
          opcode: "sqrt"
          shape { element_type: C128 }
        }
        clock_cycles: 7014
      }
      entries {
        instruction {
          opcode: "tanh"
          shape { element_type: C128 }
        }
        clock_cycles: 5400
      }
      entries {
        instruction {
          opcode: "atan2"
          shape { element_type: C128 }
        }
        clock_cycles: 21766
      }
      entries {
        instruction {
          opcode: "divide"
          shape { element_type: C128 }
        }
        clock_cycles: 4133
      }
      entries {
        instruction {
          opcode: "power"
          shape { element_type: C128 }
        }
        clock_cycles: 10458
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
      )pb"
                                        R"pb(
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

#endif  // XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_H_
