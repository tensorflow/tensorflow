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

#include "HM01B0.h"
#include "am_bsp.h" //NOLINT
#include "am_mcu_apollo.h" //NOLINT
#include "platform.h"      // TARGET specific implementation

// Image is down-sampled by applying a stride of 2 pixels in both the x and y
// directions.
static const int kStrideShift = 1;

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
//! even power of two mulitple of the requested width and height.  It down
//! samples the original image and duplicates the greyscale value for each color
//! channel.
//!
//! @return Error code.
//
//*****************************************************************************
uint32_t hm01b0_blocking_read_oneframe_scaled(
    hm01b0_cfg_t* psCfg, uint8_t* buffer, int w, int h, int channels) {
  hm01b0_single_frame_capture(psCfg);

  // Calculate the number of pixels to crop to get a centered image.
  const int offset_x = (HM01B0_PIXEL_X_NUM - (w * (1 << kStrideShift))) / 2;
  const int offset_y = (HM01B0_PIXEL_Y_NUM - (h * (1 << kStrideShift))) / 2;

  uint32_t hsync_count = 0;

  while ((hsync_count < HM01B0_PIXEL_Y_NUM)) {
    // Wait for horizontal sync.
    while (!read_hsync());

    // Get resulting image position.  When hsync_count < offset_y, this will
    // underflow resulting in an index out of bounds which we check later,
    // avoiding an unnecessary conditional.
    const uint32_t output_y = (hsync_count - offset_y) >> kStrideShift;
    uint32_t rowidx = 0;

    // Read one row. Hsync is held high for the duration of a row read.
    while (read_hsync()) {
      // Wait for pixel value to be ready.
      while (!read_pclk());

      // Read 8-bit value from camera.
      const uint8_t value = read_byte();
      const uint32_t output_x = (rowidx++ - offset_x) >> kStrideShift;
      if (output_x < w && output_y < h) {
        const int output_idx = (output_y * w + output_x) * channels;
        for (int i=0; i<channels; i++) {
          buffer[output_idx + i] = value;
        }
      }

      // Wait for next pixel clock.
      while (read_pclk());
    }

    hsync_count++;
  }
  return HM01B0_ERR_OK;
}
