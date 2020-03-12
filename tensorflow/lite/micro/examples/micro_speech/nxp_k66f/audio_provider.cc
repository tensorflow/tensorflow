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

// TensorFlow Headers
#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

// mbed and NXP FRDM-K66F Headers
#include "fsl_clock_config.h"  // NOLINT
#include "fsl_common.h"        // NOLINT
#include "fsl_dmamux.h"        // NOLINT
#include "fsl_edma.h"          // NOLINT
#include "fsl_gpio.h"          // NOLINT
#include "fsl_i2c.h"           // NOLINT
#include "fsl_lmem_cache.h"    // NOLINT
#include "fsl_port.h"          // NOLINT
#include "fsl_sai.h"           // NOLINT
#include "fsl_sai_edma.h"      // NOLINT
#include "mbed.h"              // NOLINT

// Compiler pragma for alignment of data to make efficient use of DMA
#if (defined(__ICCARM__))
#if ((!(defined(FSL_FEATURE_HAS_NO_NONCACHEABLE_SECTION) && \
        FSL_FEATURE_HAS_NO_NONCACHEABLE_SECTION)) &&        \
     defined(FSL_FEATURE_L1ICACHE_LINESIZE_BYTE))
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes) \
  SDK_PRAGMA(data_alignment = alignbytes) var @"NonCacheable"
#else
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes) \
  SDK_PRAGMA(data_alignment = alignbytes) var
#endif
#elif (defined(__CC_ARM) || defined(__ARMCC_VERSION))
#if ((!(defined(FSL_FEATURE_HAS_NO_NONCACHEABLE_SECTION) && \
        FSL_FEATURE_HAS_NO_NONCACHEABLE_SECTION)) &&        \
     defined(FSL_FEATURE_L1ICACHE_LINESIZE_BYTE))
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes) \
  __attribute__((section("NonCacheable"), zero_init))  \
      __attribute__((aligned(alignbytes))) var
#else
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes) \
  __attribute__((aligned(alignbytes))) var
#endif
#elif (defined(__GNUC__))
#if ((!(defined(FSL_FEATURE_HAS_NO_NONCACHEABLE_SECTION) && \
        FSL_FEATURE_HAS_NO_NONCACHEABLE_SECTION)) &&        \
     defined(FSL_FEATURE_L1ICACHE_LINESIZE_BYTE))
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes)          \
  __attribute__((section("NonCacheable,\"aw\",%nobits @"))) var \
      __attribute__((aligned(alignbytes)))
#else
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes) \
  var __attribute__((aligned(alignbytes)))
#endif
#else
#error Toolchain not supported.
#define AT_NONCACHEABLE_SECTION_ALIGN(var, alignbytes) var
#endif

namespace {

// Buffer configuration for receiving audio data
constexpr int kNoOfSamples = 512;
constexpr int kBufferSize = kNoOfSamples * 2;
constexpr int kNoOfBuffers = 4;
constexpr int kOverSampleRate = 384;

// Buffer management
AT_NONCACHEABLE_SECTION_ALIGN(
    static int16_t g_rx_buffer[kNoOfBuffers * kNoOfSamples], 4);
sai_edma_handle_t g_tx_sai_handle;
sai_edma_handle_t g_rx_sai_handle;
static volatile uint32_t g_tx_index = 0;
static volatile uint32_t g_rx_index = 0;
edma_handle_t g_tx_dma_handle = {0};
edma_handle_t g_rx_dma_handle = {0};
sai_transfer_t g_sai_transfer;

bool g_is_audio_initialized = false;
constexpr int kAudioCaptureBufferSize = kAudioSampleFrequency * 0.5;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;

// DA7212 configuration
constexpr int da7212ConfigurationSize = 48;
constexpr int da7212I2cAddress = 0x1A;
volatile uint8_t g_da7212_register_config[da7212ConfigurationSize][2] = {
    {0x21, 0x10},  // Set DIG_ROUTING_DAI to ADC right and ADC left
    {0x22, 0x05},  // Set Sampling rate to 16 KHz
    {0x23, 0x08},  // Enable master bias
    {0x24, 0x00},  // Clear PLL Fractional division top
    {0x25, 0x00},  // Clear PLL Fractional division bottom
    {0x26, 0x20},  // Set PLL Integer division to 32
    {0x27, 0x80},  // Set PLL input range to 2-10 MHz,system clock is PLL output
    {0x28, 0x01},  // 64  BCLK per WCLK and S
    {0x29, 0xC0},  // I2S 16-bit per channel, output is driven, DAI enable
    {0x2A, 0x32},  // One stream for left and another for right
    {0x45, 0x67},  // Set DAC Gain to 6 dB
    {0x46, 0x67},  // Set DAC Gain to 6 dB
    {0x47, 0xF1},  // Enable charge pump
    {0x4B, 0x08},  // DAC_L selected
    {0x4C, 0x08},  // DAC_R selected
    {0x69, 0xA0},  // Enable DAC_L
    {0x6A, 0xA0},  // Enable DAC_R
    {0x6B, 0xB8},  // Enable HP_L
    {0x6C, 0xB8},  // Enable HP_R
    {0x6E, 0x98},  // Enable MIXOUT_L
    {0x6F, 0x98},  // Enable MIXOUT_R
    {0x95, 0x32}, {0xE0, 0x00}, {0x32, 0x80},  // Enable MIC
    {0x33, 0x80},                              // Enable MIC
    {0x34, 0x03},                              // Add MXIN Gain
    {0x35, 0x03},                              // Add MXIN Gain
    {0x36, 0x78},                              // Add ADC Gain
    {0x37, 0x78},                              // Add ADC Gain
    {0x60, 0xB0}, {0x61, 0xB0}, {0x65, 0x88}, {0x66, 0x88}, {0x67, 0xA0},
    {0x68, 0xA0}, {0x62, 0xA9}, {0x50, 0xFE}, {0x51, 0xF7}, {0x93, 0x07},
    {0x3A, 0x04}, {0x64, 0x84}, {0x39, 0x01}, {0x63, 0x80}, {0x38, 0x88},
    {0x24, 0x00}, {0x25, 0x00}, {0x26, 0x20}, {0x20, 0x80}};

// Save audio samples into intermediate buffer
void CaptureSamples(const int16_t *sample_data) {
  const int sample_size = kNoOfSamples;
  const int32_t time_in_ms =
      g_latest_audio_timestamp + (sample_size / (kAudioSampleFrequency / 1000));

  const int32_t start_sample_offset =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < sample_size; ++i) {
    const int capture_index =
        (start_sample_offset + i) % kAudioCaptureBufferSize;
    g_audio_capture_buffer[capture_index] = sample_data[i];
  }
  // This is how we let the outside world know that new audio data has arrived.
  g_latest_audio_timestamp = time_in_ms;
}

// Callback function for SAI RX EDMA transfer complete
static void SaiRxCallback(I2S_Type *base, sai_edma_handle_t *handle,
                          status_t status, void *userData) {
  if (kStatus_SAI_RxError == status) {
    // Handle the error
  } else {
    // Save audio data into intermediate buffer
    CaptureSamples(
        reinterpret_cast<int16_t *>(g_rx_buffer + g_tx_index * kNoOfSamples));

    // Submit received audio buffer to SAI TX for audio loopback debug
    g_sai_transfer.data = (uint8_t *)(g_rx_buffer + g_tx_index * kNoOfSamples);
    g_sai_transfer.dataSize = kBufferSize;
    if (kStatus_Success ==
        SAI_TransferSendEDMA(I2S0, &g_tx_sai_handle, &g_sai_transfer)) {
      g_tx_index++;
    }
    if (g_tx_index == kNoOfBuffers) {
      g_tx_index = 0U;
    }

    // Submit buffer to SAI RX to receive audio data
    g_sai_transfer.data = (uint8_t *)(g_rx_buffer + g_rx_index * kNoOfSamples);
    g_sai_transfer.dataSize = kBufferSize;
    if (kStatus_Success ==
        SAI_TransferReceiveEDMA(I2S0, &g_rx_sai_handle, &g_sai_transfer)) {
      g_rx_index++;
    }
    if (g_rx_index == kNoOfBuffers) {
      g_rx_index = 0U;
    }
  }
}

// Callback function for TX Buffer transfer
static void SaiTxCallback(I2S_Type *base, sai_edma_handle_t *handle,
                          status_t status, void *userData) {
  if (kStatus_SAI_TxError == status) {
    // Handle the error
  }
  // Do nothing
}

// Initialize MCU pins
void McuInitializePins(void) {
  // Port B Clock Gate Control: Clock enabled
  CLOCK_EnableClock(kCLOCK_PortB);
  // Port C Clock Gate Control: Clock enabled
  CLOCK_EnableClock(kCLOCK_PortC);
  // Port E Clock Gate Control: Clock enabled
  CLOCK_EnableClock(kCLOCK_PortE);

  // PORTB16 (pin E10) is configured as UART0_RX
  PORT_SetPinMux(PORTB, 16U, kPORT_MuxAlt3);
  // PORTB17 (pin E9) is configured as UART0_TX
  PORT_SetPinMux(PORTB, 17U, kPORT_MuxAlt3);
  // PORTC1 (pin B11) is configured as I2S0_TXD0
  PORT_SetPinMux(PORTC, 1U, kPORT_MuxAlt6);

  // PORTC10 (pin C7) is configured as I2C1_SCL
  const port_pin_config_t portc10_pinC7_config = {
      kPORT_PullUp,          kPORT_FastSlewRate,     kPORT_PassiveFilterDisable,
      kPORT_OpenDrainEnable, kPORT_LowDriveStrength, kPORT_MuxAlt2,
      kPORT_UnlockRegister};
  PORT_SetPinConfig(PORTC, 10U, &portc10_pinC7_config);

  // PORTC11 (pin B7) is configured as I2C1_SDA
  const port_pin_config_t portc11_pinB7_config = {
      kPORT_PullUp,          kPORT_FastSlewRate,     kPORT_PassiveFilterDisable,
      kPORT_OpenDrainEnable, kPORT_LowDriveStrength, kPORT_MuxAlt2,
      kPORT_UnlockRegister};
  PORT_SetPinConfig(PORTC, 11U, &portc11_pinB7_config);

  // PORTC6 (pin C8) is configured as I2S0_MCLK
  PORT_SetPinMux(PORTC, 6U, kPORT_MuxAlt6);
  // PORTE11 (pin G4) is configured as I2S0_TX_FS
  PORT_SetPinMux(PORTE, 11U, kPORT_MuxAlt4);
  // PORTE12 (pin G3) is configured as I2S0_TX_BCLK
  PORT_SetPinMux(PORTE, 12U, kPORT_MuxAlt4);
  SIM->SOPT5 =
      ((SIM->SOPT5 & (~(SIM_SOPT5_UART0TXSRC_MASK))) | SIM_SOPT5_UART0TXSRC(0));
  // PORTE7 (pin F4) is configured as I2S0_RXD0
  PORT_SetPinMux(PORTE, 7U, kPORT_MuxAlt4);
  SIM->SOPT5 =
      ((SIM->SOPT5 & (~(SIM_SOPT5_UART0TXSRC_MASK))) | SIM_SOPT5_UART0TXSRC(0));
}

// Write DA7212 registers using I2C
status_t Da7212WriteRegister(uint8_t register_address, uint8_t register_data) {
  uint8_t data[1];
  data[0] = (uint8_t)register_data;
  i2c_master_transfer_t i2c_data;
  i2c_data.slaveAddress = da7212I2cAddress;
  i2c_data.direction = kI2C_Write;
  i2c_data.subaddress = register_address;
  i2c_data.subaddressSize = 1;
  i2c_data.data = (uint8_t * volatile) data;
  i2c_data.dataSize = 1;
  i2c_data.flags = kI2C_TransferDefaultFlag;
  return I2C_MasterTransferBlocking(I2C1, &i2c_data);
}

// Initialize DA7212
void Da7212Initialize(void) {
  for (uint32_t i = 0; i < da7212ConfigurationSize; i++) {
    Da7212WriteRegister(g_da7212_register_config[i][0],
                        g_da7212_register_config[i][1]);
  }
}

// Initialization for receiving audio data
TfLiteStatus InitAudioRecording(tflite::ErrorReporter *error_reporter) {
  edma_config_t dma_config = {0};
  sai_config_t sai_config;
  sai_transfer_format_t sai_format;
  volatile uint32_t delay_cycle = 500000;
  i2c_master_config_t i2c_config = {0};

  // Initialize FRDM-K66F pins
  McuInitializePins();

  // Set Clock to 180 MHz
  // BOARD_BootClockRUN();
  BOARD_BootClockHSRUN();

  // Enable Code Caching to improve performance
  LMEM_EnableCodeCache(LMEM, true);

  // Initialize I2C
  I2C_MasterGetDefaultConfig(&i2c_config);
  I2C_MasterInit(I2C1, &i2c_config, CLOCK_GetFreq(kCLOCK_BusClk));

  // Initialize SAI
  memset(&sai_format, 0U, sizeof(sai_transfer_format_t));
  SAI_TxGetDefaultConfig(&sai_config);
  SAI_TxInit(I2S0, &sai_config);
  SAI_RxGetDefaultConfig(&sai_config);
  SAI_RxInit(I2S0, &sai_config);
  sai_format.bitWidth = kSAI_WordWidth16bits;
  sai_format.channel = 0U;
  sai_format.sampleRate_Hz = kSAI_SampleRate16KHz;
  sai_format.masterClockHz = kOverSampleRate * sai_format.sampleRate_Hz;
  sai_format.protocol = sai_config.protocol;
  sai_format.stereo = kSAI_MonoRight;
  sai_format.watermark = FSL_FEATURE_SAI_FIFO_COUNT / 2U;

  // Initialize DA7212
  Da7212Initialize();

  // Initialize SAI EDMA
  EDMA_GetDefaultConfig(&dma_config);
  EDMA_Init(DMA0, &dma_config);
  EDMA_CreateHandle(&g_tx_dma_handle, DMA0, 0);
  EDMA_CreateHandle(&g_rx_dma_handle, DMA0, 1);

  // Initialize DMA MUX
  DMAMUX_Init(DMAMUX);
  DMAMUX_SetSource(DMAMUX, 0, (uint8_t)kDmaRequestMux0I2S0Tx);
  DMAMUX_EnableChannel(DMAMUX, 0);
  DMAMUX_SetSource(DMAMUX, 1, (uint8_t)kDmaRequestMux0I2S0Rx);
  DMAMUX_EnableChannel(DMAMUX, 1);

  // Wait few cycles for DA7212
  while (delay_cycle) {
    __ASM("nop");
    delay_cycle--;
  }

  // Setup SAI EDMA Callbacks
  SAI_TransferTxCreateHandleEDMA(I2S0, &g_tx_sai_handle, SaiTxCallback, NULL,
                                 &g_tx_dma_handle);
  SAI_TransferRxCreateHandleEDMA(I2S0, &g_rx_sai_handle, SaiRxCallback, NULL,
                                 &g_rx_dma_handle);
  SAI_TransferTxSetFormatEDMA(I2S0, &g_tx_sai_handle, &sai_format,
                              CLOCK_GetFreq(kCLOCK_CoreSysClk),
                              sai_format.masterClockHz);
  SAI_TransferRxSetFormatEDMA(I2S0, &g_rx_sai_handle, &sai_format,
                              CLOCK_GetFreq(kCLOCK_CoreSysClk),
                              sai_format.masterClockHz);

  // Submit buffers to SAI RX to start receiving audio
  g_sai_transfer.data = (uint8_t *)(g_rx_buffer + g_rx_index * kNoOfSamples);
  g_sai_transfer.dataSize = kBufferSize;
  if (kStatus_Success ==
      SAI_TransferReceiveEDMA(I2S0, &g_rx_sai_handle, &g_sai_transfer)) {
    g_rx_index++;
  }
  if (g_rx_index == kNoOfBuffers) {
    g_rx_index = 0U;
  }
  g_sai_transfer.data = (uint8_t *)(g_rx_buffer + g_rx_index * kNoOfSamples);
  g_sai_transfer.dataSize = kBufferSize;
  if (kStatus_Success ==
      SAI_TransferReceiveEDMA(I2S0, &g_rx_sai_handle, &g_sai_transfer)) {
    g_rx_index++;
  }
  if (g_rx_index == kNoOfBuffers) {
    g_rx_index = 0U;
  }
  return kTfLiteOk;
}

}  // namespace

// Main entry point for getting audio data.
TfLiteStatus GetAudioSamples(tflite::ErrorReporter *error_reporter,
                             int start_ms, int duration_ms,
                             int *audio_samples_size, int16_t **audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  // This should only be called when the main thread notices that the latest
  // audio sample data timestamp has changed, so that there's new data in the
  // capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < duration_sample_count; ++i) {
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
