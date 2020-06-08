/***************************************************************************//**
 * # License
 * <b>Copyright 2020 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * SPDX-License-Identifier: Zlib
 *
 * The licensor of this software is Silicon Laboratories Inc.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 ******************************************************************************/
#include <cstring>
#include <cassert>
#include <stdio.h>

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"


#include "microphone.h"



#define AlignUp(m, n) ((((m) + (n) - 1) / (n)) * (n))



static void init_microphone(tflite::ErrorReporter* error_reporter);
static void microphone_buffer_callback(const void* buf, uint32_t buffer_length_bytes, void *arg);


// Buffers up to 2s of audio (aligned to kMaxAudioSampleSize)
constexpr int kAudioBufferSize = AlignUp(kAudioSampleFrequency * 2, kMaxAudioSampleSize);


static struct
{
    bool initialized;
    volatile int32_t latest_audio_timestamp_ms;
    int16_t audio_buffer[kAudioBufferSize];
    int16_t sample_buffer[kMaxAudioSampleSize];
} g_context;




/*************************************************************************************************/
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples)
{
    constexpr int kSamplePerMs = kAudioSampleFrequency / 1000;
    const int start_index = (start_ms * kSamplePerMs) % kAudioBufferSize;
    const int end_index = ((start_ms + duration_ms) * kSamplePerMs) % kAudioBufferSize;


    init_microphone(error_reporter);


    if(start_index < end_index)
    {
        const int length = (end_index - start_index);
        memcpy(g_context.sample_buffer, &g_context.audio_buffer[start_index], length * sizeof(int16_t));
    }
    else
    {
        const int length_to_end = (kAudioBufferSize - start_index);
        memcpy(g_context.sample_buffer, &g_context.audio_buffer[start_index], length_to_end * sizeof(int16_t));
        memcpy(&g_context.sample_buffer[length_to_end], g_context.audio_buffer, end_index * sizeof(int16_t));
    }

    *audio_samples_size = kMaxAudioSampleSize;
    *audio_samples = g_context.sample_buffer;

    return kTfLiteOk;
}

/*************************************************************************************************/
int32_t LatestAudioTimestamp()
{
    return g_context.latest_audio_timestamp_ms;
}


/*************************************************************************************************/
static void init_microphone(tflite::ErrorReporter* error_reporter)
{
    if(!g_context.initialized)
    {
        microphone_config_t config;
        config.sample_rate = kAudioSampleFrequency;
        config.sample_size = kMaxAudioSampleSize;
        config.resolution = MICROPHONE_RESOLUTION_16BITS;
        config.flags = MICROPHONE_FLAG_MONO_DROP_OTHER_CHANNEL;
        config.buffer_callback = microphone_buffer_callback;
        config.buffer = g_context.audio_buffer;
        config.buffer_length_bytes = kAudioBufferSize * sizeof(int16_t);

        if(microphone_init(&config) != 0)
        {
            error_reporter->Report("Failed to initialize microphone");
            return;
        }
        else if(microphone_start() != 0)
        {
            error_reporter->Report("Failed to start microphone");
            return;
        }

        g_context.initialized = true;

        error_reporter->Report("Microphone initialized");
    }
}

/*************************************************************************************************/
static void microphone_buffer_callback(const void* buf, uint32_t buffer_length_bytes, void *arg)
{
    g_context.latest_audio_timestamp_ms += (kMaxAudioSampleSize * 1000) / kAudioSampleFrequency;
}
