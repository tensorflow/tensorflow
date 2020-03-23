


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
#include "tensorflow/lite/experimental/microfrontend/lib/fft.h"
#include <string.h>

#define USE_CEVA_FFT		1
#define T3T4_LOG2_FFT_LEN	9
#define T3T4_FRAME_LENGTH	512

#define FIXED_POINT 16
//#include "kiss_fft.h"
#include "../../../../../third_party/kissfft/kiss_fft.h"

#include "../../../../../third_party/kissfft/tools/kiss_fftr.h"


#if USE_CEVA_FFT
typedef int int32;
typedef short int16;
typedef char int8;
extern const int16 bitrev_16_1024[];
extern const int16 twi_table_16_rfft_512[];
extern const int16 CEVA_DSP_LIB_cos_sin_fft_16[];

extern "C" { void CEVA_DSP_LIB_INT16_FFT(int32 log2_buf_len,
        int16 *in_buf16,
        int16 *out_buf,
        int16 const *twi_table,
        int16 const *last_stage_twi_table,
        int16 const *bitrev_tbl,
        int16 *temp_buf,
        int8 *ScaleShift,
        int32 br);
}

short fft_tmp[2*T3T4_FRAME_LENGTH + 4*T3T4_LOG2_FFT_LEN];
char ScaleShift[T3T4_LOG2_FFT_LEN+1] ={1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

void run_fft(short *inp, short *out)
{
	int fft_len;
	int32 br;
	short *bitrev_tbl = (short*)bitrev_16_1024;
	short *ptr_twi_table_rfft = (short *)twi_table_16_rfft_512;
	fft_len = (int)T3T4_FRAME_LENGTH;

	br = 1;

	CEVA_DSP_LIB_INT16_FFT(T3T4_LOG2_FFT_LEN, inp, out, (int16 const*)CEVA_DSP_LIB_cos_sin_fft_16, (int16 const *)ptr_twi_table_rfft, (int16 const*)bitrev_tbl, (int16*)fft_tmp, ScaleShift, br);
}
#endif

void FftCompute(struct FftState* state, const int16_t* input,
                int input_scale_shift) {
  const size_t input_size = state->input_size;
  const size_t fft_size = state->fft_size;

  int16_t* fft_input = state->input;
  // First, scale the input by the given shift.
  int i;
  for (i = 0; i < input_size; ++i) {
    *fft_input++ = (*input++) << input_scale_shift;
  }
  // Zero out whatever else remains in the top part of the input.
  for (; i < fft_size; ++i) {
    *fft_input++ = 0;
  }
#if USE_CEVA_FFT
  run_fft((short*)state->input, (short*)state->output);
#else
  // Apply the FFT.
  kiss_fftr(reinterpret_cast<const kiss_fftr_cfg>(state->scratch), state->input,
            reinterpret_cast<kiss_fft_cpx*>(state->output));
#endif
}

void FftInit(struct FftState* state) {
  // All the initialization is done in FftPopulateState()
}

void FftReset(struct FftState* state) {
  memset(state->input, 0, state->fft_size * sizeof(*state->input));
  memset(state->output, 0, (state->fft_size / 2 + 1) * sizeof(*state->output));
}
