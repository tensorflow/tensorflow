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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_DEBUG_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_DEBUG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "HM01B0.h"

//*****************************************************************************
//
//! @brief Read one frame of data from HM01B0 scaled to 96x96 RGB.
//!
//! @param buffer       - Pointer to the frame buffer.
//! @param w            - Image width.
//! @param h            - Image height.
//! @param channels     - Number of channels per pixel.
//!
//! This function reads data of one frame from HM01B0. It trims the image to an
//! even power of two multiple of the requested width and height.  It down
//! samples the original image and duplicates the greyscale value for each color
//! channel.
//!
//! @return Error code.
//
//*****************************************************************************

void hm01b0_framebuffer_dump(uint8_t* frame, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_HIMAX_DRIVER_HM01B0_DEBUG_H_
