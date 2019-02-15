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

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

#include <vector>

// TensorFlow Lite was migrated out of `contrib/` directory. The change
// wasn't reflected in newest CocoaPod release yet (1.12.0).
// Change this to 0 when using a TFLite version which is newer than 1.12.0.
// TODO(ycling): Remove the macro when we release the next version.
#ifndef TFLITE_USE_CONTRIB_LITE
#define TFLITE_USE_CONTRIB_LITE 1
#endif

// Set TFLITE_USE_GPU_DELEGATE to 1 to use TFLite GPU Delegate.
// Note: TFLite GPU Delegate binary isn't releast yet, and we're working
// on it.
#ifndef TFLITE_USE_GPU_DELEGATE
#define TFLITE_USE_GPU_DELEGATE 0
#endif

#if TFLITE_USE_GPU_DELEGATE && TFLITE_USE_CONTRIB_LITE
// Sanity check.
#error "GPU Delegate only works with newer TFLite " \
    "after migrating out of contrib"
#endif

@interface CameraExampleViewController
    : UIViewController<UIGestureRecognizerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate> {
  IBOutlet UIView* previewView;
  AVCaptureVideoPreviewLayer* previewLayer;
  AVCaptureVideoDataOutput* videoDataOutput;
  dispatch_queue_t videoDataOutputQueue;
  UIView* flashView;
  BOOL isUsingFrontFacingCamera;
  NSMutableDictionary* oldPredictionValues;
  NSMutableArray* labelLayers;
  AVCaptureSession* session;

  std::vector<std::string> labels;
  double total_latency;
  int total_count;
}
@property(strong, nonatomic) CATextLayer* predictionTextLayer;

- (IBAction)takePicture:(id)sender;
- (IBAction)switchCameras:(id)sender;

@end
