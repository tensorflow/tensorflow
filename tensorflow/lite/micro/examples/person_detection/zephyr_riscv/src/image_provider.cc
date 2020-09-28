/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"
#include <device.h>
#include <drivers/gpio.h>
#include <drivers/i2c.h>
#include <drivers/spi.h>
#include <errno.h>
#include <kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <zephyr.h>
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/zephyr_riscv/src/ov2640_regs.h"

// JPEGDecoder library
#include "tensorflow/lite/micro/examples/person_detection/zephyr_riscv/JPEGDecoder/JPEGDecoder.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#define GPIO_OUT "gpio_out"
#define I2C "i2c0"
#define SPI "spi0"
#define OV2640_I2C_ADDR_WRITE \
  0x30  // 0x60 >> 1 because i2c driver does "<< 1" operation
#define OV2640_I2C_ADDR_READ \
  0xb0  // 0x61 >> 1 because i2c driver does "<< 1" operation

#define SELECT_BANK 0xff
#define BANK_0 0x00
#define BANK_1 0x01
#define COM7 0x12
#define SYSTEM_RESET 0x80
#define COM10 0x15
#define DEFAULT_COM10 0x00

#define TEST_REGISTER 0x00
#define ARDUCHIP_FRAMES 0x01
#define ARDUCHIP_FIFO 0x04  // FIFO and I2C control
#define FIFO_CLEAR_MASK 0x01
#define FIFO_START_MASK 0x02
#define SINGLE_FIFO_READ 0x3d
#define ARDUCHIP_TRIG 0x41  // Trigger source
#define CAP_DONE_MASK 0x08
#define RESET_CPLD 0x07
#define START_RESET_CPLD 0x80
#define STOP_RESET_CPLD 0x00
#define FIFO_SIZE1 0x42  // Camera write FIFO size[7:0] for burst to read
#define FIFO_SIZE2 0x43  // Camera write FIFO size[15:8]
#define FIFO_SIZE3 0x44  // Camera write FIFO size[18:16]
#define CS_PIN 0
#define MAX_JPEG_BYTES 8192
#define SPI_FREQUENCY 1000000U

struct device* gpio_out_dev;
struct device* i2c_dev;
struct device* spi_dev;
struct spi_config spi_conf;

uint8_t jpeg_buffer[MAX_JPEG_BYTES] = {0};
uint32_t jpeg_length;

void set_CS(int state) { gpio_pin_set(gpio_out_dev, CS_PIN, state); }

void set_registers(const uint16_t dev_addr, const struct sensor_reg* regs,
                   const int len) {
  for (int i = 0; i < len; i++) {
    while ((i2c_reg_write_byte(i2c_dev, dev_addr, regs[i].reg, regs[i].val)) !=
           0)
      ;
  }
}

void ov2640_initCamRegs(tflite::ErrorReporter* error_reporter) {
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "Setting up camera via I2C. It might take few minutes, please be "
      "patient...");

  struct sensor_reg reg[] = {{SELECT_BANK, BANK_1}, {COM7, SYSTEM_RESET}};
  set_registers(OV2640_I2C_ADDR_WRITE, reg, sizeof(reg) / sizeof(reg[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [1/8]");

  set_registers(OV2640_I2C_ADDR_WRITE, OV2640_JPEG_INIT,
                sizeof(OV2640_JPEG_INIT) / sizeof(OV2640_JPEG_INIT[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [2/8]");

  set_registers(OV2640_I2C_ADDR_WRITE, OV2640_YUV422,
                sizeof(OV2640_YUV422) / sizeof(OV2640_YUV422[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [3/8]");

  set_registers(OV2640_I2C_ADDR_WRITE, OV2640_JPEG,
                sizeof(OV2640_JPEG) / sizeof(OV2640_JPEG[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [4/8]");

  reg[1].reg = COM10;
  reg[1].val = DEFAULT_COM10;
  set_registers(OV2640_I2C_ADDR_WRITE, reg, sizeof(reg) / sizeof(reg[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [5/8]");

  set_registers(OV2640_I2C_ADDR_WRITE, OV2640_160x120_JPEG,
                sizeof(OV2640_160x120_JPEG) / sizeof(OV2640_160x120_JPEG[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [6/8]");

  // set automatic white balance mode to simple
  struct sensor_reg awb_simple[] = {{0xff, 0x00}, {0xc7, 0x10}};
  set_registers(OV2640_I2C_ADDR_WRITE, awb_simple,
                sizeof(awb_simple) / sizeof(awb_simple[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "Setting up sensor [7/8]");

  // set color sauration level to 0
  struct sensor_reg saturation_lvl_0[] = {{0xff, 0x00}, {0x7c, 0x00},
                                          {0x7d, 0x02}, {0x7c, 0x03},
                                          {0x7d, 0x48}, {0x7d, 0x48}};
  set_registers(OV2640_I2C_ADDR_WRITE, saturation_lvl_0,
                sizeof(saturation_lvl_0) / sizeof(saturation_lvl_0[0]));
  TF_LITE_REPORT_ERROR(error_reporter, "All registers set! [8/8]");
}

int ov2640_write_reg(uint8_t address, uint8_t value) {
  uint8_t txmsg[2] = {address | 0x80, value};
  struct spi_buf tx = {.buf = txmsg, .len = sizeof(txmsg)};
  const struct spi_buf_set tx_bufs = {.buffers = &tx, .count = 1};
  int ret = 0;

  set_CS(0);
  ret = spi_write(spi_dev, &spi_conf, &tx_bufs);
  set_CS(1);
  return ret;
}

int ov2640_read_reg(uint8_t address) {
  int val;
  uint8_t txmsg[2] = {address & 0x7f, 0};
  uint8_t rxmsg[2] = {0, 0};
  struct spi_buf tx = {.buf = txmsg, .len = sizeof(txmsg)};
  struct spi_buf rx = {.buf = rxmsg, .len = sizeof(rxmsg)};
  const struct spi_buf_set tx_bufs = {.buffers = &tx, .count = 1};
  const struct spi_buf_set rx_bufs = {.buffers = &rx, .count = 1};

  set_CS(0);
  spi_transceive(spi_dev, &spi_conf, &tx_bufs, &rx_bufs);
  set_CS(1);
  val = rxmsg[1];
  return val;
}

uint8_t ov2640_get_bit(uint8_t address, uint8_t bit) {
  uint8_t temp;
  temp = ov2640_read_reg(address);
  temp = temp & bit;
  return temp;
}

uint32_t ov2640_read_fifo_length() {
  uint32_t len1, len2, len3, length = 0;
  len1 = ov2640_read_reg(FIFO_SIZE1);
  len2 = ov2640_read_reg(FIFO_SIZE2);
  len3 = ov2640_read_reg(FIFO_SIZE3) & 0x7f;
  length = ((len3 << 16) | (len2 << 8) | len1) & 0x07fffff;
  return length;
}

TfLiteStatus InitCamera(tflite::ErrorReporter* error_reporter) {
  int ret;
  gpio_out_dev = device_get_binding(GPIO_OUT);
  i2c_dev = device_get_binding(I2C);
  spi_dev = device_get_binding(SPI);

  if (gpio_out_dev == NULL) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to bind gpio_out");
    return kTfLiteError;
  }
  if (spi_dev == NULL) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to bind spi");
    return kTfLiteError;
  }
  if (i2c_dev == NULL) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to bind i2c");
    return kTfLiteError;
  }
  uint32_t i2c_cfg = I2C_MODE_MASTER | I2C_SPEED_SET(I2C_SPEED_FAST);

  gpio_pin_configure(gpio_out_dev, 0, GPIO_OUTPUT_ACTIVE);
  gpio_pin_configure(gpio_out_dev, 1, GPIO_OUTPUT_ACTIVE);
  ret = i2c_configure(i2c_dev, i2c_cfg);
  if (ret == 0)
    TF_LITE_REPORT_ERROR(error_reporter, "I2C configured.");
  else {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to configure i2c device.");
    return kTfLiteError;
  }
  spi_conf.operation = SPI_WORD_SET(8) | SPI_OP_MODE_MASTER | SPI_TRANSFER_MSB;
  spi_conf.frequency = SPI_FREQUENCY;

  ov2640_initCamRegs(error_reporter);
  TF_LITE_REPORT_ERROR(error_reporter,
                       "Image sensor registers set successfully.");

  ov2640_write_reg(RESET_CPLD, START_RESET_CPLD);  // Start reset CPLD
  k_sleep(100);
  ov2640_write_reg(RESET_CPLD, STOP_RESET_CPLD);  // Stop reset CPLD
  k_sleep(100);
  ov2640_write_reg(TEST_REGISTER, 0x55);  // Write 0x55 to register 0x00
  if (ov2640_read_reg(TEST_REGISTER) != 0x55)
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to communicate with camera OV2640.");

  return kTfLiteOk;
}

TfLiteStatus PerformCapture(tflite::ErrorReporter* error_reporter) {
  TF_LITE_REPORT_ERROR(error_reporter, "Starting capture");
  ov2640_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);  // Flush FIFO
  ov2640_write_reg(ARDUCHIP_FIFO,
                   FIFO_CLEAR_MASK);     // Clear FIFO write done flag
  ov2640_write_reg(ARDUCHIP_FRAMES, 0);  // Set number of frames to 1 (0 + 1)
  ov2640_write_reg(ARDUCHIP_FIFO, FIFO_START_MASK);  // Start capture

  while (ov2640_get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK) != 0x08)
    ;  // Wait for camera write FIFO done flag
  TF_LITE_REPORT_ERROR(error_reporter, "Image captured.");
  ov2640_write_reg(ARDUCHIP_FIFO,
                   FIFO_CLEAR_MASK);  // Clear FIFO write done flag

  return kTfLiteOk;
}

TfLiteStatus ReadData(tflite::ErrorReporter* error_reporter) {
  jpeg_length = ov2640_read_fifo_length();
  TF_LITE_REPORT_ERROR(error_reporter, "Reading %d bytes from Arducam",
                       jpeg_length);
  // Ensure there's not too much data for our buffer
  if (jpeg_length > MAX_JPEG_BYTES) {
    TF_LITE_REPORT_ERROR(error_reporter, "Too many bytes in FIFO buffer (%d)",
                         MAX_JPEG_BYTES);
    return kTfLiteError;
  }
  if (jpeg_length == 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "No data in Arducam FIFO buffer");
    return kTfLiteError;
  }
  set_CS(0);
  uint8_t cur_byte = 0, last_byte = 0;
  int index = 0;
  bool is_header = false;
  do {
    last_byte = cur_byte;
    cur_byte = ov2640_read_reg(SINGLE_FIFO_READ);

    if (is_header == true) {
      jpeg_buffer[index++] = cur_byte;
    } else if (cur_byte == 0xd8 && last_byte == 0xff) {
      is_header = true;
      jpeg_buffer[0] = last_byte;
      jpeg_buffer[1] = cur_byte;
      index = 2;
    }
  } while ((cur_byte != 0xd9 || last_byte != 0xff) && index < jpeg_length);
  jpeg_length = index + 1;

  TF_LITE_REPORT_ERROR(error_reporter, "Finished reading");
  set_CS(1);
  return kTfLiteOk;
}

TfLiteStatus DecodeAndProcessImage(tflite::ErrorReporter* error_reporter,
                                   int image_width, int image_height,
                                   uint8_t* image_data) {
  TF_LITE_REPORT_ERROR(error_reporter,
                       "Decoding JPEG and converting to grayscale");
  // Parse the JPEG headers. The image will be decoded as a sequence of Minimum
  // Coded Units (MCUs), which are 16x8 blocks of pixels.
  JpegDec.decodeArray(jpeg_buffer, jpeg_length);

  // Crop the image by keeping a certain number of MCUs in each dimension
  const int keep_x_mcus = image_width / JpegDec.MCUWidth;
  const int keep_y_mcus = image_height / JpegDec.MCUHeight;

  // Calculate how many MCUs we will throw away on the x axis
  const int skip_x_mcus = JpegDec.MCUSPerRow - keep_x_mcus;
  // Roughly center the crop by skipping half the throwaway MCUs at the
  // beginning of each row
  const int skip_start_x_mcus = skip_x_mcus / 2;
  // Index where we will start throwing away MCUs after the data
  const int skip_end_x_mcu_index = skip_start_x_mcus + keep_x_mcus;
  // Same approach for the columns
  const int skip_y_mcus = JpegDec.MCUSPerCol - keep_y_mcus;
  const int skip_start_y_mcus = skip_y_mcus / 2;
  const int skip_end_y_mcu_index = skip_start_y_mcus + keep_y_mcus;

  // Pointer to the current pixel
  uint16_t* pImg;
  // Color of the current pixel
  uint16_t color;

  // Loop over the MCUs
  while (JpegDec.read()) {
    // Skip over the initial set of rows
    if (JpegDec.MCUy < skip_start_y_mcus) {
      continue;
    }
    // Skip if we're on a column that we don't want
    if (JpegDec.MCUx < skip_start_x_mcus ||
        JpegDec.MCUx >= skip_end_x_mcu_index) {
      continue;
    }
    // Skip if we've got all the rows we want
    if (JpegDec.MCUy >= skip_end_y_mcu_index) {
      continue;
    }
    // Pointer to the current pixel
    pImg = JpegDec.pImage;

    // The x and y indexes of the current MCU, ignoring the MCUs we skip
    int relative_mcu_x = JpegDec.MCUx - skip_start_x_mcus;
    int relative_mcu_y = JpegDec.MCUy - skip_start_y_mcus;

    // The coordinates of the top left of this MCU when applied to the output
    // image
    int x_origin = relative_mcu_x * JpegDec.MCUWidth;
    int y_origin = relative_mcu_y * JpegDec.MCUHeight;

    // Loop through the MCU's rows and columns
    for (int mcu_row = 0; mcu_row < JpegDec.MCUHeight; mcu_row++) {
      // The y coordinate of this pixel in the output index
      int current_y = y_origin + mcu_row;

      for (int mcu_col = 0; mcu_col < JpegDec.MCUWidth; mcu_col++) {
        // Read the color of the pixel as 16-bit integer
        color = *pImg++;
        // Extract the color values (5 red bits, 6 green, 5 blue)
        uint8_t r, g, b;
        r = ((color & 0xF800) >> 11) * 8;
        g = ((color & 0x07E0) >> 5) * 4;
        b = ((color & 0x001F) >> 0) * 8;
        // Convert to grayscale by calculating luminance
        // See https://en.wikipedia.org/wiki/Grayscale for magic numbers
        float gray_value = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);

        // The x coordinate of this pixel in the output image
        int current_x = x_origin + mcu_col;
        // The index of this pixel in our flat output buffer
        int index = (current_y * image_width) + current_x;
        image_data[index] = static_cast<uint8_t>(gray_value);
      }
    }
  }
  TF_LITE_REPORT_ERROR(error_reporter, "Image decoded and processed");
  return kTfLiteOk;
}

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, uint8_t* image_data) {
  static bool g_is_camera_initialized = false;
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "InitCamera failed");
      return init_status;
    }
    g_is_camera_initialized = true;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "\nGetting image...");
  TfLiteStatus capture_status = PerformCapture(error_reporter);
  if (capture_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "PerformCapture failed");
    return capture_status;
  }

  TfLiteStatus read_data_status = ReadData(error_reporter);
  if (read_data_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "ReadData failed");
    return read_data_status;
  }

  TfLiteStatus decode_status = DecodeAndProcessImage(
      error_reporter, image_width, image_height, image_data);
  if (decode_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "DecodeAndProcessImage failed");
    return decode_status;
  }

  return kTfLiteOk;
}
