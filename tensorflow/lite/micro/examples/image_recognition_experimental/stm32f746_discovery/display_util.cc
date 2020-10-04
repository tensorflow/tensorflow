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

#include "tensorflow/lite/micro/examples/image_recognition_experimental/stm32f746_discovery/display_util.h"

#include <stdint.h>

#include "LCD_DISCO_F746NG/LCD_DISCO_F746NG.h"

LCD_DISCO_F746NG lcd;

extern "C" {
// defined in stm32746g_discovery_camera.c
extern DCMI_HandleTypeDef hDcmiHandler;
void DCMI_IRQHandler(void) { HAL_DCMI_IRQHandler(&hDcmiHandler); }
void DMA2_Stream1_IRQHandler(void) {
  HAL_DMA_IRQHandler(hDcmiHandler.DMA_Handle);
}
}

static char lcd_output_string[50];

void init_lcd() { lcd.Clear(LCD_COLOR_WHITE); }

void display_image_rgb888(int x_dim, int y_dim, const uint8_t* image_data,
                          int x_loc, int y_loc) {
  for (int y = 0; y < y_dim; ++y) {
    for (int x = 0; x < x_dim; ++x, image_data += 3) {
      uint8_t a = 0xFF;
      auto r = image_data[0];
      auto g = image_data[1];
      auto b = image_data[2];
      int pixel = a << 24 | r << 16 | g << 8 | b;
      lcd.DrawPixel(x_loc + x, y_loc + y, pixel);
    }
  }
}

void display_image_rgb565(int x_dim, int y_dim, const uint8_t* image_data,
                          int x_loc, int y_loc) {
  for (int y = 0; y < y_dim; ++y) {
    for (int x = 0; x < x_dim; ++x, image_data += 2) {
      uint8_t a = 0xFF;
      uint8_t pix_lo = image_data[0];
      uint8_t pix_hi = image_data[1];
      uint8_t r = (0xF8 & pix_hi);
      uint8_t g = ((0x07 & pix_hi) << 5) | ((0xE0 & pix_lo) >> 3);
      uint8_t b = (0x1F & pix_lo) << 3;
      int pixel = a << 24 | r << 16 | g << 8 | b;
      // inverted image, so draw from bottom-right to top-left
      lcd.DrawPixel(x_loc + (x_dim - x), y_loc + (y_dim - y), pixel);
    }
  }
}

void print_prediction(const char* prediction) {
  // NOLINTNEXTLINE
  sprintf(lcd_output_string, "  Prediction: %s       ", prediction);
  lcd.DisplayStringAt(0, LINE(8), (uint8_t*)lcd_output_string, LEFT_MODE);
}

void print_confidence(uint8_t max_score) {
  // NOLINTNEXTLINE
  sprintf(lcd_output_string, "  Confidence: %.1f%%   ",
          (max_score / 255.0) * 100.0);
  lcd.DisplayStringAt(0, LINE(9), (uint8_t*)lcd_output_string, LEFT_MODE);
}
