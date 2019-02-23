/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank.h"

#include <string.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/bits.h"

void FilterbankConvertFftComplexToEnergy(struct FilterbankState* state,
                                         struct complex_int16_t* fft_output,
                                         int32_t* energy) {
  const int end_index = state->end_index;
  int i;
  energy += state->start_index;
  fft_output += state->start_index;
  for (i = state->start_index; i < end_index; ++i) {
    const int32_t real = fft_output->real;
    const int32_t imag = fft_output->imag;
    fft_output++;
    const uint32_t mag_squared = (real * real) + (imag * imag);
    *energy++ = mag_squared;
  }
}

void FilterbankAccumulateChannels(struct FilterbankState* state,
                                  const int32_t* energy) {
  uint64_t* work = state->work;
  uint64_t weight_accumulator = 0;
  uint64_t unweight_accumulator = 0;

  const int16_t* channel_frequency_starts = state->channel_frequency_starts;
  const int16_t* channel_weight_starts = state->channel_weight_starts;
  const int16_t* channel_widths = state->channel_widths;

  int num_channels_plus_1 = state->num_channels + 1;
  int i;
  for (i = 0; i < num_channels_plus_1; ++i) {
    const int32_t* magnitudes = energy + *channel_frequency_starts++;
    const int16_t* weights = state->weights + *channel_weight_starts;
    const int16_t* unweights = state->unweights + *channel_weight_starts++;
    const int width = *channel_widths++;
    int j;
    for (j = 0; j < width; ++j) {
      weight_accumulator += *weights++ * (static_cast<uint64_t>(*magnitudes));
      unweight_accumulator +=
          *unweights++ * (static_cast<uint64_t>(*magnitudes));
      ++magnitudes;
    }
    *work++ = weight_accumulator;
    weight_accumulator = unweight_accumulator;
    unweight_accumulator = 0;
  }
}

static uint16_t Sqrt32(uint32_t num) {
  if (num == 0) {
    return 0;
  }
  uint32_t res = 0;
  int max_bit_number = 32 - MostSignificantBit32(num);
  max_bit_number |= 1;
  uint32_t bit = 1U << (31 - max_bit_number);
  int iterations = (31 - max_bit_number) / 2 + 1;
  while (iterations--) {
    if (num >= res + bit) {
      num -= res + bit;
      res = (res >> 1U) + bit;
    } else {
      res >>= 1U;
    }
    bit >>= 2U;
  }
  // Do rounding - if we have the bits.
  if (num > res && res != 0xFFFF) {
    ++res;
  }
  return res;
}

static uint32_t Sqrt64(uint64_t num) {
  // Take a shortcut and just use 32 bit operations if the upper word is all
  // clear. This will cause a slight off by one issue for numbers close to 2^32,
  // but it probably isn't going to matter (and gives us a big performance win).
  if ((num >> 32) == 0) {
    return Sqrt32(static_cast<uint32_t>(num));
  }
  uint64_t res = 0;
  int max_bit_number = 64 - MostSignificantBit64(num);
  max_bit_number |= 1;
  uint64_t bit = 1ULL << (63 - max_bit_number);
  int iterations = (63 - max_bit_number) / 2 + 1;
  while (iterations--) {
    if (num >= res + bit) {
      num -= res + bit;
      res = (res >> 1U) + bit;
    } else {
      res >>= 1U;
    }
    bit >>= 2U;
  }
  // Do rounding - if we have the bits.
  if (num > res && res != 0xFFFFFFFFLL) {
    ++res;
  }
  return res;
}

uint32_t* FilterbankSqrt(struct FilterbankState* state, int scale_down_shift) {
  const int num_channels = state->num_channels;
  const int64_t* work = reinterpret_cast<int64_t*>(state->work + 1);
  // Reuse the work buffer since we're fine clobbering it at this point to hold
  // the output.
  uint32_t* output = reinterpret_cast<uint32_t*>(state->work);
  int i;
  for (i = 0; i < num_channels; ++i) {
    *output++ = Sqrt64(*work++) >> scale_down_shift;
  }
  return reinterpret_cast<uint32_t*>(state->work);
}

void FilterbankReset(struct FilterbankState* state) {
  memset(state->work, 0, (state->num_channels + 1) * sizeof(*state->work));
}
