
#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/detection_responder.h"
#include "tensorflow/lite/micro/benchmarks/eembc/monitor/th_api/th_lib.h"

extern const unsigned char  g_person_detect_model_data[];
extern bool                 g_use_buffer;
extern void                *gp_buff;

TfLiteStatus
GetImage(
    tflite::ErrorReporter *error_reporter,
    int                    image_width,
    int                    image_height,
    int                    channels,
    uint8_t*               image_data)
{
    const uint8_t *ptr;

    ptr = g_use_buffer ? (uint8_t *)gp_buff : g_person_detect_model_data;
    
    for (int i = 0; i < image_width * image_height * channels; ++i)
    {
        image_data[i] = ptr[i];
    }
    
    return kTfLiteOk;
}

void
RespondToDetection(
    tflite::ErrorReporter *error_reporter,
    uint8_t                person_score,
    uint8_t                no_person_score)
{
    th_printf(
        "m-[Person %03d, No Person %03d]\r\n",
        person_score,
        no_person_score);
}
