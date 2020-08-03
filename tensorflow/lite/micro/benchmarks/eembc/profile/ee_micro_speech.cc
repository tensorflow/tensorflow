
#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/benchmarks/eembc/monitor/th_api/th_lib.h"

extern const int     g_yes_1000ms_sample_data_size; //16000
extern const int16_t g_yes_1000ms_sample_data[]; //16000
// ee_main.c
extern bool          g_use_buffer;
extern void         *gp_buff;
extern size_t        g_buff_size;

int16_t g_dummy_audio_data[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;


TfLiteStatus
GetAudioSamples(
    tflite::ErrorReporter  *error_reporter,
    int                     start_ms,
    int                     duration_ms,
    int                    *audio_samples_size,
    int16_t               **audio_samples)
{
    static size_t wrap_at(0);
    static size_t wrap(0);
    int16_t *ptr;

    ptr     = g_use_buffer ? (int16_t*)gp_buff : (int16_t*)g_yes_1000ms_sample_data;
    wrap_at = g_use_buffer ? g_buff_size : g_yes_1000ms_sample_data_size;

    for (int i = 0; i < kMaxAudioSampleSize; ++i, ++wrap)
    {
        g_dummy_audio_data[i] = ptr[i];
        if (wrap >= wrap_at)
        {
            wrap = 0;
        }
    }
    *audio_samples_size = kMaxAudioSampleSize;
    *audio_samples = g_dummy_audio_data;
    return kTfLiteOk;
}

int32_t
LatestAudioTimestamp()
{
    g_latest_audio_timestamp += 100;
    return g_latest_audio_timestamp;
}

void
RespondToCommand(
    tflite::ErrorReporter *error_reporter,
    int32_t                ms,
    const char            *found_command,
    uint8_t                score,
    bool                   is_new_command)
{
    th_printf("m-found[%s]-ms[%d]-score[%d]\r\n", found_command, ms, score);
}
