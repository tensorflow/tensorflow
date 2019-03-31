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
#include "tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#define kFilterbankIndexAlignment 4
#define kFilterbankChannelBlockSize 4

void FilterbankFillConfigWithDefaults(struct FilterbankConfig* config) {
  config->num_channels = 32;
  config->lower_band_limit = 125.0f;
  config->upper_band_limit = 7500.0f;
  config->output_scale_shift = 7;
}

static float FreqToMel(float freq) { return 1127.0 * log1p(freq / 700.0); }

static void CalculateCenterFrequencies(const int num_channels,
                                       const float lower_frequency_limit,
                                       const float upper_frequency_limit,
                                       float* center_frequencies) {
  assert(lower_frequency_limit >= 0.0f);
  assert(upper_frequency_limit > lower_frequency_limit);

  const float mel_low = FreqToMel(lower_frequency_limit);
  const float mel_hi = FreqToMel(upper_frequency_limit);
  const float mel_span = mel_hi - mel_low;
  const float mel_spacing = mel_span / ((float)num_channels);
  int i;
  for (i = 0; i < num_channels; ++i) {
    center_frequencies[i] = mel_low + (mel_spacing * (i + 1));
  }
}

static void QuantizeFilterbankWeights(const float float_weight, int16_t* weight,
                                      int16_t* unweight) {
  *weight = floor(float_weight * (1 << kFilterbankBits) + 0.5);
  *unweight = floor((1.0 - float_weight) * (1 << kFilterbankBits) + 0.5);
}

int FilterbankPopulateState(const struct FilterbankConfig* config,
                            struct FilterbankState* state, int sample_rate,
                            int spectrum_size) {
  state->num_channels = config->num_channels;
  const int num_channels_plus_1 = config->num_channels + 1;

  // How should we align things to index counts given the byte alignment?
  const int index_alignment =
      (kFilterbankIndexAlignment < sizeof(int16_t)
           ? 1
           : kFilterbankIndexAlignment / sizeof(int16_t));

  state->channel_frequency_starts =
      malloc(num_channels_plus_1 * sizeof(*state->channel_frequency_starts));
  state->channel_weight_starts =
      malloc(num_channels_plus_1 * sizeof(*state->channel_weight_starts));
  state->channel_widths =
      malloc(num_channels_plus_1 * sizeof(*state->channel_widths));
  state->work = malloc(num_channels_plus_1 * sizeof(*state->work));

  float* center_mel_freqs =
      malloc(num_channels_plus_1 * sizeof(*center_mel_freqs));
  int16_t* actual_channel_starts =
      malloc(num_channels_plus_1 * sizeof(*actual_channel_starts));
  int16_t* actual_channel_widths =
      malloc(num_channels_plus_1 * sizeof(*actual_channel_widths));

  if (state->channel_frequency_starts == NULL ||
      state->channel_weight_starts == NULL || state->channel_widths == NULL ||
      center_mel_freqs == NULL || actual_channel_starts == NULL ||
      actual_channel_widths == NULL) {
    free(center_mel_freqs);
    free(actual_channel_starts);
    free(actual_channel_widths);
    fprintf(stderr, "Failed to allocate channel buffers\n");
    return 0;
  }

  CalculateCenterFrequencies(num_channels_plus_1, config->lower_band_limit,
                             config->upper_band_limit, center_mel_freqs);

  // Always exclude DC.
  const float hz_per_sbin = 0.5 * sample_rate / ((float)spectrum_size - 1);
  state->start_index = 1.5 + config->lower_band_limit / hz_per_sbin;
  state->end_index = 0;  // Initialized to zero here, but actually set below.

  // For each channel, we need to figure out what frequencies belong to it, and
  // how much padding we need to add so that we can efficiently multiply the
  // weights and unweights for accumulation. To simplify the multiplication
  // logic, all channels will have some multiplication to do (even if there are
  // no frequencies that accumulate to that channel) - they will be directed to
  // a set of zero weights.
  int chan_freq_index_start = state->start_index;
  int weight_index_start = 0;
  int needs_zeros = 0;

  int chan;
  for (chan = 0; chan < num_channels_plus_1; ++chan) {
    // Keep jumping frequencies until we overshoot the bound on this channel.
    int freq_index = chan_freq_index_start;
    while (FreqToMel((freq_index)*hz_per_sbin) <= center_mel_freqs[chan]) {
      ++freq_index;
    }

    const int width = freq_index - chan_freq_index_start;
    actual_channel_starts[chan] = chan_freq_index_start;
    actual_channel_widths[chan] = width;

    if (width == 0) {
      // This channel doesn't actually get anything from the frequencies, it's
      // always zero. We need then to insert some 'zero' weights into the
      // output, and just redirect this channel to do a single multiplication at
      // this point. For simplicity, the zeros are placed at the beginning of
      // the weights arrays, so we have to go and update all the other
      // weight_starts to reflect this shift (but only once).
      state->channel_frequency_starts[chan] = 0;
      state->channel_weight_starts[chan] = 0;
      state->channel_widths[chan] = kFilterbankChannelBlockSize;
      if (!needs_zeros) {
        needs_zeros = 1;
        int j;
        for (j = 0; j < chan; ++j) {
          state->channel_weight_starts[j] += kFilterbankChannelBlockSize;
        }
        weight_index_start += kFilterbankChannelBlockSize;
      }
    } else {
      // How far back do we need to go to ensure that we have the proper
      // alignment?
      const int aligned_start =
          (chan_freq_index_start / index_alignment) * index_alignment;
      const int aligned_width = (chan_freq_index_start - aligned_start + width);
      const int padded_width =
          (((aligned_width - 1) / kFilterbankChannelBlockSize) + 1) *
          kFilterbankChannelBlockSize;

      state->channel_frequency_starts[chan] = aligned_start;
      state->channel_weight_starts[chan] = weight_index_start;
      state->channel_widths[chan] = padded_width;
      weight_index_start += padded_width;
    }
    chan_freq_index_start = freq_index;
  }

  // Allocate the two arrays to store the weights - weight_index_start contains
  // the index of what would be the next set of weights that we would need to
  // add, so that's how many weights we need to allocate.
  state->weights = calloc(weight_index_start, sizeof(*state->weights));
  state->unweights = calloc(weight_index_start, sizeof(*state->unweights));

  // If the alloc failed, we also need to nuke the arrays.
  if (state->weights == NULL || state->unweights == NULL) {
    free(center_mel_freqs);
    free(actual_channel_starts);
    free(actual_channel_widths);
    fprintf(stderr, "Failed to allocate weights or unweights\n");
    return 0;
  }

  // Next pass, compute all the weights. Since everything has been memset to
  // zero, we only need to fill in the weights that correspond to some frequency
  // for a channel.
  const float mel_low = FreqToMel(config->lower_band_limit);
  for (chan = 0; chan < num_channels_plus_1; ++chan) {
    int frequency = actual_channel_starts[chan];
    const int num_frequencies = actual_channel_widths[chan];
    const int frequency_offset =
        frequency - state->channel_frequency_starts[chan];
    const int weight_start = state->channel_weight_starts[chan];
    const float denom_val = (chan == 0) ? mel_low : center_mel_freqs[chan - 1];

    int j;
    for (j = 0; j < num_frequencies; ++j, ++frequency) {
      const float weight =
          (center_mel_freqs[chan] - FreqToMel(frequency * hz_per_sbin)) /
          (center_mel_freqs[chan] - denom_val);

      // Make the float into an integer for the weights (and unweights).
      const int weight_index = weight_start + frequency_offset + j;
      QuantizeFilterbankWeights(weight, state->weights + weight_index,
                                state->unweights + weight_index);
    }
    if (frequency > state->end_index) {
      state->end_index = frequency;
    }
  }

  free(center_mel_freqs);
  free(actual_channel_starts);
  free(actual_channel_widths);
  if (state->end_index >= spectrum_size) {
    fprintf(stderr, "Filterbank end_index is above spectrum size.\n");
    return 0;
  }
  return 1;
}

void FilterbankFreeStateContents(struct FilterbankState* state) {
  free(state->channel_frequency_starts);
  free(state->channel_weight_starts);
  free(state->channel_widths);
  free(state->weights);
  free(state->unweights);
  free(state->work);
}
