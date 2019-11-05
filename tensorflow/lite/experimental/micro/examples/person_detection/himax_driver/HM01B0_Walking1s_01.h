/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_WALKING1S_01_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_WALKING1S_01_H_

#include "HM01B0.h"

const hm_script_t sHM01b0TestModeScript_Walking1s[] = {
    {
        0x2100,
        0x00,
    },  // W 24 2100 00 2 1 ; AE
    {
        0x1000,
        0x00,
    },  // W 24 1000 00 2 1 ; BLC
    {
        0x1008,
        0x00,
    },  // W 24 1008 00 2 1 ; DPC
    {
        0x0205,
        0x00,
    },  // W 24 0205 00 2 1 ; AGain
    {
        0x020E,
        0x01,
    },  // W 24 020E 01 2 1 ; DGain
    {
        0x020F,
        0x00,
    },  // W 24 020F 00 2 1 ; DGain
    {
        0x0601,
        0x11,
    },  // W 24 0601 11 2 1 ; Test pattern
    {
        0x0104,
        0x01,
    },  // W 24 0104 01 2 1 ;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_WALKING1S_01_H_
