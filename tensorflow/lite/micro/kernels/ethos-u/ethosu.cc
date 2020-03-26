#include <ethosu_driver.h>

#include "tensorflow/lite/c/common.h"
#define FLATBUFFERS_LOCALE_INDEPENDENT 0
#include "tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/flatbuffers/flexbuffers.h"

namespace tflite {
namespace custom {
namespace ethosu {

constexpr uint8_t CO_TYPE_ETHOSU = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE(context, node->inputs->size > 0);
    TF_LITE_ENSURE(context, context->tensors);
    TF_LITE_ENSURE(context, node->custom_initial_data_size > 0);
    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    //Get base addresses
    TfLiteTensor* tensor;
    int num_base_addr = node->inputs->size + node->outputs->size;
    int i = 0;
    int num_tensors = 0;
    uint64_t base_addrs[num_base_addr];
    void* cms_data;
    int cms_data_size;
    uint8_t co_type;
    int result;

    const uint8_t *custom_data = static_cast<uint8_t const*>(node->custom_initial_data);
    auto root = flexbuffers::GetRoot(custom_data,
                                     node->custom_initial_data_size);
    co_type = root.AsInt8();
    if (co_type != CO_TYPE_ETHOSU) {
        context->ReportError(context, "CO_TYPE != ETHOSU");
        return kTfLiteError;
    }

    // Get command stream data address and size
    tensor = &(context->tensors[node->inputs->data[0]]);
    cms_data = reinterpret_cast<void*>(tensor->data.uint8);
    cms_data_size = tensor->bytes;

    // Get adresses to weights/scratch/input data
    for (i = 1; i < node->inputs->size; ++i) {
        tensor = &(context->tensors[node->inputs->data[i]]);
        base_addrs[num_tensors] =
            reinterpret_cast<uint64_t>(tensor->data.uint8);
        num_tensors++;
    }

    // Get adresses to output data
    for (i = 0; i < node->outputs->size; ++i) {
        tensor = &(context->tensors[node->outputs->data[i]]);
        base_addrs[num_tensors] =
            reinterpret_cast<uint64_t>(tensor->data.uint8);
        num_tensors++;
    }

    result = ethosu_invoke(cms_data, cms_data_size, base_addrs, num_tensors);
    if (-1 == result) {
        return kTfLiteError;
    } else {
        return kTfLiteOk;
    }
}

}  // namespace ethosu

TfLiteRegistration* Register_ETHOSU() {
    static TfLiteRegistration r =
        {ethosu::Init, ethosu::Free, ethosu::Prepare, ethosu::Eval};
    return &r;
}

const char* GetString_ETHOSU() {
    return "ethos-u";
}

}  // namespace custom
}  // namespace tflite
