// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#import "RunModelViewController.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#include "ios_image_load.h"

#define LOG(x) std::cerr
#define CHECK(x)                  \
  if (!(x)) {                     \
    LOG(ERROR) << #x << "failed"; \
    exit(1);                      \
  }

static void GetTopN(const float* prediction,
                    const int prediction_size,
                    const int num_results, const float threshold,
                    std::vector<std::pair<float, int> >* top_results);

@interface RunModelViewController ()<UINavigationControllerDelegate, UIImagePickerControllerDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UITextView *textView;

@end

@implementation RunModelViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.imageView.image = [UIImage imageNamed:@"grace_hopper.jpg"];
}

- (IBAction)runModel:(UIButton *)sender {
    self.textView.text = [self runInferenceOnImage:self.imageView.image];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info; {
    [picker dismissViewControllerAnimated:YES completion:nil];
    self.imageView.image = [self fixrotation:info[UIImagePickerControllerOriginalImage]];
}

- (IBAction)openPhoto:(UIBarButtonItem *)sender {
    if ([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypePhotoLibrary]) {
        UIImagePickerController *picker = [[UIImagePickerController alloc] init];
        picker.delegate = self;
        picker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
        [self presentViewController:picker animated:YES completion:nil];
    }
}

- (UIImage *)fixrotation:(UIImage *)image{
    if (image.imageOrientation == UIImageOrientationUp) return image;
    CGAffineTransform transform = CGAffineTransformIdentity;
    
    switch (image.imageOrientation) {
        case UIImageOrientationDown:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.width, image.size.height);
            transform = CGAffineTransformRotate(transform, (CGFloat) M_PI);
            break;
            
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.width, 0);
            transform = CGAffineTransformRotate(transform, (CGFloat) M_PI_2);
            break;
            
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, 0, image.size.height);
            transform = CGAffineTransformRotate(transform, (CGFloat) -M_PI_2);
            break;
        case UIImageOrientationUp:
        case UIImageOrientationUpMirrored:
            break;
    }
    
    switch (image.imageOrientation) {
        case UIImageOrientationUpMirrored:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.width, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
            
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, image.size.height, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
        case UIImageOrientationUp:
        case UIImageOrientationDown:
        case UIImageOrientationLeft:
        case UIImageOrientationRight:
            break;
    }
    
    // Now we draw the underlying CGImage into a new context, applying the transform
    // calculated above.
    CGContextRef ctx = CGBitmapContextCreate(NULL, image.size.width, image.size.height,
                                             CGImageGetBitsPerComponent(image.CGImage), 0,
                                             CGImageGetColorSpace(image.CGImage),
                                             CGImageGetBitmapInfo(image.CGImage));
    CGContextConcatCTM(ctx, transform);
    CGRect rect;
    switch (image.imageOrientation) {
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            rect = CGRectMake(0,0,image.size.height,image.size.width);
            break;
        default:
            rect = CGRectMake(0,0,image.size.width,image.size.height);
            break;
    }
    CGContextDrawImage(ctx, rect, image.CGImage);
    
    // And now we just create a new UIImage from the drawing context
    CGImageRef cgimg = CGBitmapContextCreateImage(ctx);
    UIImage *img = [UIImage imageWithCGImage:cgimg];
    CGContextRelease(ctx);
    CGImageRelease(cgimg);
    return img;
}

- (NSString*)filePathForResourceName:(NSString*)name withExtension:(NSString*)extension {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (!file_path) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}

- (NSString*)runInferenceOnImage:(UIImage *)image {
    const int num_threads = 1;
    std::string input_layer_type = "float";
    std::vector<int> sizes = {1, 224, 224, 3};
    
    NSString* graph_path = [self filePathForResourceName:@"mobilenet_v1_1.0_224" withExtension: @"tflite"];
    
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile([graph_path fileSystemRepresentation]));
    NSAssert(model, @"Failed to mmap model %@", graph_path);

    LOG(INFO) << "Loaded model " << graph_path.UTF8String;
    model->error_reporter();
    LOG(INFO) << "resolved reporter";
#ifdef TFLITE_CUSTOM_OPS_HEADER
    tflite::MutableOpResolver resolver;
    RegisterSelectedOps(&resolver);
#else
    tflite::ops::builtin::BuiltinOpResolver resolver;
#endif
    
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter";
    }
    
    if (num_threads != -1) {
        interpreter->SetNumThreads(num_threads);
    }
    
    int input = interpreter->inputs()[0];
    
    if (input_layer_type != "string") {
        interpreter->ResizeInputTensor(input, sizes);
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }
    
    // Read the label list
    NSString* labels_path = [self filePathForResourceName:@"labels" withExtension:@"txt"];
    std::vector<std::string> label_strings;
    std::ifstream t;
    t.open([labels_path fileSystemRepresentation]);
    std::string line;
    while(t){
        std::getline(t, line);
        label_strings.push_back(line);
    }
    t.close();
    
    NSAssert(image, @"Image is nil");
    
    const int image_width = image.size.width;
    const int image_height = image.size.height;
    const int image_channels = 4;
    std::vector<uint8_t> image_data = LoadImageFromUIImage(image);
    
    const int wanted_width = 224;
    const int wanted_height = 224;
    const int wanted_channels = 3;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    assert(image_channels >= wanted_channels);
    uint8_t* in = image_data.data();
    float* out = interpreter->typed_tensor<float>(input);
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        uint8_t* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            uint8_t* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    
    float* output = interpreter->typed_output_tensor<float>(0);
    const int output_size = 1000;
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    std::vector<std::pair<float, int> > top_results;
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    
    std::stringstream ss;
    ss.precision(3);
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        
        ss << index << " " << confidence << "  ";
        
        // Write out the result as a string
        if (index < label_strings.size()) {
            // just for safety: theoretically, the output is under 1000 unless there
            // is some numerical issues leading to a wrong prediction.
            ss << label_strings[index];
        } else {
            ss << "Prediction: " << index;
        }
        
        ss << "\n";
    }
    
    LOG(INFO) << "Predictions: " << ss.str();
    
    std::string predictions = ss.str();
    NSString* result = @"";
    result = [NSString stringWithFormat: @"%@ - %s", result,
              predictions.c_str()];
    
    return result;
}

@end

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(const float* prediction,
                    const int prediction_size,
                    const int num_results, const float threshold,
                    std::vector<std::pair<float, int> >* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>,
    std::vector<std::pair<float, int> >,
    std::greater<std::pair<float, int> > > top_result_pq;
    
    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        const float value = prediction[i];
        
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}
