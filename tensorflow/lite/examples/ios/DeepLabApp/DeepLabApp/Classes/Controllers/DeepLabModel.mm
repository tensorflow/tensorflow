// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "DeepLabModel.h"
#import <AssertMacros.h>
#include <iostream>

#include "DeepLabModel.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"

#define LOG(x) std::cerr

namespace {
    const int wanted_input_width = 257;
    const int wanted_input_height = 257;
    const int wanted_input_channels = 3;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    // Preprocess the input image and feed the TFLite interpreter buffer for a float model.
    void ProcessInputWithFloatModel(uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
        for (int y = 0; y < wanted_input_height; ++y) {
            float* out_row = buffer + (y * wanted_input_width * wanted_input_channels);
            for (int x = 0; x < wanted_input_width; ++x) {
                const int in_x = (y * image_width) / wanted_input_width;
                const int in_y = (x * image_height) / wanted_input_height;
                uint8_t* input_pixel =
                input + (in_y * image_width * image_channels) + (in_x * image_channels);
                float* out_pixel = out_row + (x * wanted_input_channels);
                for (int c = 0; c < wanted_input_channels; ++c) {
                    out_pixel[c] = (input_pixel[c] - input_mean) / input_std;
                }
            }
        }
    }
    
    // Preprocess the input image and feed the TFLite interpreter buffer for a quantized model.
    void ProcessInputWithQuantizedModel(uint8_t* input, uint8_t* output, int image_width, int image_height, int image_channels) {
        for (int y = 0; y < wanted_input_height; ++y) {
            uint8_t* out_row = output + (y * wanted_input_width * wanted_input_channels);
            for (int x = 0; x < wanted_input_width; ++x) {
                const int in_x = (y * image_width) / wanted_input_width;
                const int in_y = (x * image_height) / wanted_input_height;
                uint8_t* in_pixel = input + (in_y * image_width * image_channels) + (in_x * image_channels);
                uint8_t* out_pixel = out_row + (x * wanted_input_channels);
                for (int c = 0; c < wanted_input_channels; ++c) {
                    out_pixel[c] = in_pixel[c];
                }
            }
        }
    }
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
        << "' in bundle.";
    }
    return file_path;
}

@implementation DeepLabModel {
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate* delegate;
    unsigned char *result;
}

-(BOOL) loadModel {
    NSString *modelPath = FilePathForResourceName(@"deeplabv3_257_mv_gpu", @"tflite");
    NSFileManager *fileManager = [NSFileManager defaultManager];
    
    if ([fileManager fileExistsAtPath: [NSString stringWithFormat:@"file://%@", modelPath]] == YES) {
        return NO;
    }
    
    model = tflite::FlatBufferModel::BuildFromFile([modelPath UTF8String]);
    if (!model) {
        LOG(FATAL) << "Failed to mmap model " << modelPath;
    }
    
    LOG(INFO) << "Loaded model " << modelPath;
    model->error_reporter();
    LOG(INFO) << "resolved reporter";
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    GpuDelegateOptions options;
    options.allow_precision_loss = true;
    options.wait_type = GpuDelegateOptions::WaitType::kActive;
    delegate = NewGpuDelegate(&options);
    interpreter->ModifyGraphWithDelegate(delegate);
    
    // Explicitly resize the input tensor.
    {
        int input = interpreter->inputs()[0];
        std::vector<int> sizes = {1, 257, 257, 3};
        interpreter->ResizeInputTensor(input, sizes);
    }
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter";
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    result = new unsigned char [257 * 257 * 4];
    return YES;
}

- (unsigned char *) process:(CVPixelBufferRef) pixelBuffer {
    assert(pixelBuffer != NULL);
    
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB || sourcePixelFormat == kCVPixelFormatType_32BGRA);
    const int image_channels = 4;
    
    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
    
    int image_height;
    unsigned char* sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    
    assert(image_channels >= wanted_input_channels);
    uint8_t* in = sourceStartAddr;
    
    int input = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input);
    
    bool is_quantized;
    switch (input_tensor->type) {
        case kTfLiteFloat32:
            is_quantized = false;
            break;
        case kTfLiteUInt8:
            is_quantized = true;
            break;
        default:
            NSLog(@"Input data type is not supported by this demo app.");
            return nil;
    }
    
    if (is_quantized) {
        uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
        ProcessInputWithQuantizedModel(in, out, image_width, image_height, image_channels);
    } else {
        float* out = interpreter->typed_tensor<float>(input);
        ProcessInputWithFloatModel(in, out, image_width, image_height, image_channels);
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    
    float* output = interpreter->typed_output_tensor<float>(0);
    int class_count = 21;
    static unsigned int colors[21] = {  0000000000, 0x804caf50, 0x80e7d32, 0x8064ffda, 0x80004d40,
                                        0x800277bd, 0x8001579b, 0x8003a9f4, 0x80795548, 0x806200ea,
                                        0x80d50000, 0x80ff8a80, 0x80ff8a80, 0x8033691e, 0x80827717,
                                        1077952640, 0x80aeea00, 0x80ff9100, 0x80ff5722, 0x80795548,
                                        0x80546e7a};
    
    for (int index = 0; index<257*257; index++) {
        int classID = 0;
        int classValue = INT_MIN;
        
        for (int classIndex = 0; classIndex < class_count; classIndex++) {
            int class_value_index = (index * class_count) + classIndex;
            float value = output[class_value_index];
            if (classValue < value) {
                classValue = value;
                classID = classIndex;
            }
        }

        unsigned int color = colors[classID];
        memcpy(&result[index * 4 + 0], &color, sizeof(unsigned int));

    }
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

    return result;
}


@end
