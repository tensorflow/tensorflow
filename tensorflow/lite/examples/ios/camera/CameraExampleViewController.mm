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

#import "CameraExampleViewController.h"
#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#if TFLITE_USE_CONTRIB_LITE
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/op_resolver.h"
#include "tensorflow/contrib/lite/string_util.h"
#else
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"
#if TFLITE_USE_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
#endif

#define LOG(x) std::cerr

namespace {

// If you have your own model, modify this to the file name, and make sure
// you've added the file to your app resources too.
#if TFLITE_USE_GPU_DELEGATE
// GPU Delegate only supports float model now.
NSString* model_file_name = @"mobilenet_v1_1.0_224";
#else
NSString* model_file_name = @"mobilenet_quant_v1_224";
#endif
NSString* model_file_type = @"tflite";
// If you have your own model, point this to the labels file.
NSString* labels_file_name = @"labels";
NSString* labels_file_type = @"txt";

// These dimensions need to match those the model was trained with.
const int wanted_input_width = 224;
const int wanted_input_height = 224;
const int wanted_input_channels = 3;
const float input_mean = 127.5f;
const float input_std = 127.5f;
const std::string input_layer_name = "input";
const std::string output_layer_name = "softmax1";

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
               << "' in bundle.";
  }
  return file_path;
}

void LoadLabels(NSString* file_name, NSString* file_type, std::vector<std::string>* label_strings) {
  NSString* labels_path = FilePathForResourceName(file_name, file_type);
  if (!labels_path) {
    LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
               << [file_type UTF8String];
  }
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings->push_back(line);
  }
  t.close();
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
void GetTopN(
    const float* prediction, const int prediction_size, const int num_results,
    const float threshold, std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;

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

// Preprocess the input image and feed the TFLite interpreter buffer for a float model.
void ProcessInputWithFloatModel(
    uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
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
void ProcessInputWithQuantizedModel(
    uint8_t* input, uint8_t* output, int image_width, int image_height, int image_channels) {
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

}  // namespace

@interface CameraExampleViewController (InternalMethods)
- (void)setupAVCapture;
- (void)teardownAVCapture;
@end

@implementation CameraExampleViewController {
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  TfLiteDelegate* delegate;
}

- (void)setupAVCapture {
  NSError* error = nil;

  session = [AVCaptureSession new];
  if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone)
    [session setSessionPreset:AVCaptureSessionPreset640x480];
  else
    [session setSessionPreset:AVCaptureSessionPresetPhoto];

  AVCaptureDevice* device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
  AVCaptureDeviceInput* deviceInput =
      [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];

  if (error != nil) {
    NSLog(@"Failed to initialize AVCaptureDeviceInput. Note: This app doesn't work with simulator");
    assert(NO);
  }

  if ([session canAddInput:deviceInput]) [session addInput:deviceInput];

  videoDataOutput = [AVCaptureVideoDataOutput new];

  NSDictionary* rgbOutputSettings =
      [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                                  forKey:(id)kCVPixelBufferPixelFormatTypeKey];
  [videoDataOutput setVideoSettings:rgbOutputSettings];
  [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
  videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
  [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];

  if ([session canAddOutput:videoDataOutput]) [session addOutput:videoDataOutput];
  [[videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];

  previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
  [previewLayer setBackgroundColor:[[UIColor blackColor] CGColor]];
  [previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];
  CALayer* rootLayer = [previewView layer];
  [rootLayer setMasksToBounds:YES];
  [previewLayer setFrame:[rootLayer bounds]];
  [rootLayer addSublayer:previewLayer];
  [session startRunning];

  if (error) {
    NSString* title = [NSString stringWithFormat:@"Failed with error %d", (int)[error code]];
    UIAlertController* alertController =
        [UIAlertController alertControllerWithTitle:title
                                            message:[error localizedDescription]
                                     preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction* dismiss =
        [UIAlertAction actionWithTitle:@"Dismiss" style:UIAlertActionStyleDefault handler:nil];
    [alertController addAction:dismiss];
    [self presentViewController:alertController animated:YES completion:nil];
    [self teardownAVCapture];
  }
}

- (void)teardownAVCapture {
  [previewLayer removeFromSuperlayer];
}

- (AVCaptureVideoOrientation)avOrientationForDeviceOrientation:
    (UIDeviceOrientation)deviceOrientation {
  AVCaptureVideoOrientation result = (AVCaptureVideoOrientation)(deviceOrientation);
  if (deviceOrientation == UIDeviceOrientationLandscapeLeft)
    result = AVCaptureVideoOrientationLandscapeRight;
  else if (deviceOrientation == UIDeviceOrientationLandscapeRight)
    result = AVCaptureVideoOrientationLandscapeLeft;
  return result;
}

- (IBAction)takePicture:(id)sender {
  if ([session isRunning]) {
    [session stopRunning];
    [sender setTitle:@"Continue" forState:UIControlStateNormal];

    flashView = [[UIView alloc] initWithFrame:[previewView frame]];
    [flashView setBackgroundColor:[UIColor whiteColor]];
    [flashView setAlpha:0.f];
    [[[self view] window] addSubview:flashView];

    [UIView animateWithDuration:.2f
        animations:^{
          [flashView setAlpha:1.f];
        }
        completion:^(BOOL finished) {
          [UIView animateWithDuration:.2f
              animations:^{
                [flashView setAlpha:0.f];
              }
              completion:^(BOOL finished) {
                [flashView removeFromSuperview];
                flashView = nil;
              }];
        }];

  } else {
    [session startRunning];
    [sender setTitle:@"Freeze Frame" forState:UIControlStateNormal];
  }
}

- (void)captureOutput:(AVCaptureOutput*)captureOutput
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  CFRetain(pixelBuffer);
  [self runModelOnFrame:pixelBuffer];
  CFRelease(pixelBuffer);
}

- (void)runModelOnFrame:(CVPixelBufferRef)pixelBuffer {
  assert(pixelBuffer != NULL);

  OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
  assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
         sourcePixelFormat == kCVPixelFormatType_32BGRA);

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
  const int image_channels = 4;
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
    return;
  }

  if (is_quantized) {
    uint8_t* out = interpreter->typed_tensor<uint8_t>(input);
    ProcessInputWithQuantizedModel(in, out, image_width, image_height, image_channels);
  } else {
    float* out = interpreter->typed_tensor<float>(input);
    ProcessInputWithFloatModel(in, out, image_width, image_height, image_channels);
  }

  double start = [[NSDate new] timeIntervalSince1970];
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke!";
  }
  double end = [[NSDate new] timeIntervalSince1970];
  total_latency += (end - start);
  total_count += 1;
  NSLog(@"Time: %.4lf, avg: %.4lf, count: %d", end - start, total_latency / total_count,
        total_count);

  const int output_size = 1000;
  const int kNumResults = 5;
  const float kThreshold = 0.1f;

  std::vector<std::pair<float, int> > top_results;

  if (is_quantized) {
    uint8_t* quantized_output = interpreter->typed_output_tensor<uint8_t>(0);
    int32_t zero_point = input_tensor->params.zero_point;
    float scale = input_tensor->params.scale;
    float output[output_size];
    for (int i = 0; i < output_size; ++i) {
      output[i] = (quantized_output[i] - zero_point) * scale;
    }
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
  } else {
    float* output = interpreter->typed_output_tensor<float>(0);
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
  }

  NSMutableDictionary* newValues = [NSMutableDictionary dictionary];
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    NSString* labelObject = [NSString stringWithUTF8String:labels[index].c_str()];
    NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
    [newValues setObject:valueObject forKey:labelObject];
  }
  dispatch_async(dispatch_get_main_queue(), ^(void) {
    [self setPredictionValues:newValues];
  });

  CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)dealloc {
#if TFLITE_USE_GPU_DELEGATE
  if (delegate) {
    DeleteGpuDelegate(delegate);
  }
#endif
  [self teardownAVCapture];
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
}

- (void)viewDidLoad {
  [super viewDidLoad];
  labelLayers = [[NSMutableArray alloc] init];
  oldPredictionValues = [[NSMutableDictionary alloc] init];

  NSString* graph_path = FilePathForResourceName(model_file_name, model_file_type);
  model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << graph_path;
  }
  LOG(INFO) << "Loaded model " << graph_path;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

  tflite::ops::builtin::BuiltinOpResolver resolver;
  LoadLabels(labels_file_name, labels_file_type, &labels);

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

#if TFLITE_USE_GPU_DELEGATE
  GpuDelegateOptions options;
  options.allow_precision_loss = true;
  options.wait_type = GpuDelegateOptions::WaitType::kActive;
  delegate = NewGpuDelegate(&options);
  interpreter->ModifyGraphWithDelegate(delegate);
#endif

  // Explicitly resize the input tensor.
  {
    int input = interpreter->inputs()[0];
    std::vector<int> sizes = {1, 224, 224, 3};
    interpreter->ResizeInputTensor(input, sizes);
  }
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter";
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  [self setupAVCapture];
}

- (void)viewDidUnload {
  [super viewDidUnload];
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];
}

- (void)viewDidDisappear:(BOOL)animated {
  [super viewDidDisappear:animated];
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
  return (interfaceOrientation == UIInterfaceOrientationPortrait);
}

- (BOOL)prefersStatusBarHidden {
  return YES;
}

- (void)setPredictionValues:(NSDictionary*)newValues {
  const float decayValue = 0.75f;
  const float updateValue = 0.25f;
  const float minimumThreshold = 0.01f;

  NSMutableDictionary* decayedPredictionValues = [[NSMutableDictionary alloc] init];
  for (NSString* label in oldPredictionValues) {
    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
    const float oldPredictionValue = [oldPredictionValueObject floatValue];
    const float decayedPredictionValue = (oldPredictionValue * decayValue);
    if (decayedPredictionValue > minimumThreshold) {
      NSNumber* decayedPredictionValueObject = [NSNumber numberWithFloat:decayedPredictionValue];
      [decayedPredictionValues setObject:decayedPredictionValueObject forKey:label];
    }
  }
  oldPredictionValues = decayedPredictionValues;

  for (NSString* label in newValues) {
    NSNumber* newPredictionValueObject = [newValues objectForKey:label];
    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
    if (!oldPredictionValueObject) {
      oldPredictionValueObject = [NSNumber numberWithFloat:0.0f];
    }
    const float newPredictionValue = [newPredictionValueObject floatValue];
    const float oldPredictionValue = [oldPredictionValueObject floatValue];
    const float updatedPredictionValue = (oldPredictionValue + (newPredictionValue * updateValue));
    NSNumber* updatedPredictionValueObject = [NSNumber numberWithFloat:updatedPredictionValue];
    [oldPredictionValues setObject:updatedPredictionValueObject forKey:label];
  }
  NSArray* candidateLabels = [NSMutableArray array];
  for (NSString* label in oldPredictionValues) {
    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
    const float oldPredictionValue = [oldPredictionValueObject floatValue];
    if (oldPredictionValue > 0.05f) {
      NSDictionary* entry = @{@"label" : label, @"value" : oldPredictionValueObject};
      candidateLabels = [candidateLabels arrayByAddingObject:entry];
    }
  }
  NSSortDescriptor* sort = [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
  NSArray* sortedLabels =
      [candidateLabels sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];

  const float leftMargin = 10.0f;
  const float topMargin = 10.0f;

  const float valueWidth = 48.0f;
  const float valueHeight = 18.0f;

  const float labelWidth = 246.0f;
  const float labelHeight = 18.0f;

  const float labelMarginX = 5.0f;
  const float labelMarginY = 5.0f;

  [self removeAllLabelLayers];

  int labelCount = 0;
  for (NSDictionary* entry in sortedLabels) {
    NSString* label = [entry objectForKey:@"label"];
    NSNumber* valueObject = [entry objectForKey:@"value"];
    const float value = [valueObject floatValue];
    const float originY = topMargin + ((labelHeight + labelMarginY) * labelCount);
    const int valuePercentage = (int)roundf(value * 100.0f);

    const float valueOriginX = leftMargin;
    NSString* valueText = [NSString stringWithFormat:@"%d%%", valuePercentage];

    [self addLabelLayerWithText:valueText
                        originX:valueOriginX
                        originY:originY
                          width:valueWidth
                         height:valueHeight
                      alignment:kCAAlignmentRight];

    const float labelOriginX = (leftMargin + valueWidth + labelMarginX);

    [self addLabelLayerWithText:[label capitalizedString]
                        originX:labelOriginX
                        originY:originY
                          width:labelWidth
                         height:labelHeight
                      alignment:kCAAlignmentLeft];

    labelCount += 1;
    if (labelCount > 4) {
      break;
    }
  }
}

- (void)removeAllLabelLayers {
  for (CATextLayer* layer in labelLayers) {
    [layer removeFromSuperlayer];
  }
  [labelLayers removeAllObjects];
}

- (void)addLabelLayerWithText:(NSString*)text
                      originX:(float)originX
                      originY:(float)originY
                        width:(float)width
                       height:(float)height
                    alignment:(NSString*)alignment {
  CFTypeRef font = (CFTypeRef) @"Menlo-Regular";
  const float fontSize = 12.0;
  const float marginSizeX = 5.0f;
  const float marginSizeY = 2.0f;

  const CGRect backgroundBounds = CGRectMake(originX, originY, width, height);
  const CGRect textBounds = CGRectMake((originX + marginSizeX), (originY + marginSizeY),
                                       (width - (marginSizeX * 2)), (height - (marginSizeY * 2)));

  CATextLayer* background = [CATextLayer layer];
  [background setBackgroundColor:[UIColor blackColor].CGColor];
  [background setOpacity:0.5f];
  [background setFrame:backgroundBounds];
  background.cornerRadius = 5.0f;

  [[self.view layer] addSublayer:background];
  [labelLayers addObject:background];

  CATextLayer* layer = [CATextLayer layer];
  [layer setForegroundColor:[UIColor whiteColor].CGColor];
  [layer setFrame:textBounds];
  [layer setAlignmentMode:alignment];
  [layer setWrapped:YES];
  [layer setFont:font];
  [layer setFontSize:fontSize];
  layer.contentsScale = [[UIScreen mainScreen] scale];
  [layer setString:text];

  [[self.view layer] addSublayer:layer];
  [labelLayers addObject:layer];
}

@end
