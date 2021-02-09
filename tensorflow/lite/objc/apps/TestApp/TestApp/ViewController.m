// Copyright 2019 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "ViewController.h"

#if COCOAPODS
@import TFLTensorFlowLite;
#else
#import "TFLTensorFlowLite.h"
#endif

NS_ASSUME_NONNULL_BEGIN

/**
 * Safely dispatches the given `block` on the main thread. If already on the main thread, the given
 * block is executed immediately; otherwise, dispatches the block asynchronously on the main thread.
 *
 * @param block The block to dispatch on the main thread.
 */
void TLTSafeDispatchOnMain(dispatch_block_t block) {
  if (block == nil) return;
  if (NSThread.isMainThread) {
    block();
  } else {
    dispatch_async(dispatch_get_main_queue(), block);
  }
}

/**
 * Name of a float model that performs two add operations on one input tensor and returns the result
 * in one output tensor.
 */
static NSString *const kModelNameAdd = @"add";

/**
 * Name of a quantized model that performs two add operations on one input tensor and returns the
 * result in one output tensor.
 */
static NSString *const kModelNameAddQuantized = @"add_quantized";

/**
 * Name of a float model that performs three add operations on four input tensors and returns the
 * results in 2 output tensors.
 */
static NSString *const kModelNameMultiAdd = @"multi_add";

/** Model resource type. */
static NSString *const kModelType = @"bin";

/** The label for the serial queue for synchronizing interpreter calls. */
static const char *kInterpreterSerialQueueLabel = "com.tensorflow.lite.objc.testapp.interpreter";

static NSString *const kNilInterpreterError =
    @"Failed to invoke the interpreter because the interpreter was nil.";
static NSString *const kInvokeInterpreterError = @"Failed to invoke interpreter due to error: %@.";

/** Model paths. */
static NSArray<NSString *> *gModelPaths;

@interface ViewController ()

/** Serial queue for synchronizing interpreter calls. */
@property(nonatomic) dispatch_queue_t interpreterSerialQueue;

/** TensorFlow Lite interpreter for the currently selected model. */
@property(nonatomic) TFLInterpreter *interpreter;

@property(weak, nonatomic) IBOutlet UISegmentedControl *modelControl;
@property(weak, nonatomic) IBOutlet UIBarButtonItem *invokeButton;
@property(weak, nonatomic) IBOutlet UITextView *resultsTextView;

@end

@implementation ViewController

#pragma mark - NSObject

+ (void)initialize {
  if (self == [ViewController self]) {
    gModelPaths = @[
      [NSBundle.mainBundle pathForResource:kModelNameAdd ofType:kModelType],
      [NSBundle.mainBundle pathForResource:kModelNameAddQuantized ofType:kModelType],
      [NSBundle.mainBundle pathForResource:kModelNameMultiAdd ofType:kModelType],
    ];
  }
}

#pragma mark - UIViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  self.interpreterSerialQueue =
      dispatch_queue_create(kInterpreterSerialQueueLabel, DISPATCH_QUEUE_SERIAL);
  self.invokeButton.enabled = NO;
  [self updateResultsText:[NSString stringWithFormat:@"Using TensorFlow Lite runtime version %@.",
                                                     TFLVersion]];
  [self loadModel];
}

#pragma mark - IBActions

- (IBAction)modelChanged:(id)sender {
  self.invokeButton.enabled = NO;
  NSString *results = [NSString
      stringWithFormat:@"Switched to the %@ model.",
                       [self.modelControl
                           titleForSegmentAtIndex:self.modelControl.selectedSegmentIndex]];
  [self updateResultsText:results];
  [self loadModel];
}

- (IBAction)invokeInterpreter:(id)sender {
  switch (self.modelControl.selectedSegmentIndex) {
    case 0:
      [self invokeAdd];
      break;
    case 1:
      [self invokeAddQuantized];
      break;
    case 2:
      [self invokeMultiAdd];
  }
}

#pragma mark - Private

/** Path of the currently selected model. */
- (nullable NSString *)currentModelPath {
  return self.modelControl.selectedSegmentIndex == UISegmentedControlNoSegment
             ? nil
             : gModelPaths[self.modelControl.selectedSegmentIndex];
}

- (void)loadModel {
  NSString *modelPath = [self currentModelPath];
  if (modelPath.length == 0) {
    [self updateResultsText:@"No model is selected."];
    return;
  }

  __weak typeof(self) weakSelf = self;
  dispatch_async(self.interpreterSerialQueue, ^{
    TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
    options.numberOfThreads = 2;

    NSError *error;
    weakSelf.interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                             options:options
                                                           delegates:@[]
                                                               error:&error];
    if (weakSelf.interpreter == nil || error != nil) {
      NSString *results =
          [NSString stringWithFormat:@"Failed to create the interpreter due to error:%@",
                                     error.localizedDescription];
      [weakSelf updateResultsText:results];
    } else {
      TLTSafeDispatchOnMain(^{
        weakSelf.invokeButton.enabled = YES;
      });
    }
  });
}

- (void)invokeAdd {
  __weak typeof(self) weakSelf = self;
  dispatch_async(self.interpreterSerialQueue, ^{
    if (weakSelf.interpreter == nil) {
      [weakSelf updateResultsText:kNilInterpreterError];
      return;
    }

    NSArray<NSNumber *> *shape = @[@2];
    NSError *error;

    if (![weakSelf.interpreter resizeInputTensorAtIndex:0 toShape:shape error:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    if (![weakSelf.interpreter allocateTensorsWithError:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    TFLTensor *inputTensor = [weakSelf.interpreter inputTensorAtIndex:0 error:&error];
    if (inputTensor == nil || error != nil) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
    float one = 1.f;
    float three = 3.f;
    [inputData appendBytes:&one length:sizeof(float)];
    [inputData appendBytes:&three length:sizeof(float)];
    if (![inputTensor copyData:inputData error:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    if (![weakSelf.interpreter invokeWithError:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    TFLTensor *outputTensor = [weakSelf.interpreter outputTensorAtIndex:0 error:&error];
    if (outputTensor == nil || error != nil) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    NSData *outputData = [outputTensor dataWithError:&error];
    if (outputData == nil || error != nil) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }
    float output[2U];
    [outputData getBytes:output length:(sizeof(float) * 2U)];

    [weakSelf
        updateResultsText:[NSString stringWithFormat:@"Performing 2 add operations:\n\nInput = "
                                                     @"[%.1f, %.1f]\n\nOutput = [%.1f, %.1f]",
                                                     one, three, output[0], output[1]]];
  });
}

- (void)invokeAddQuantized {
  __weak typeof(self) weakSelf = self;
  dispatch_async(self.interpreterSerialQueue, ^{
    if (weakSelf.interpreter == nil) {
      [weakSelf updateResultsText:kNilInterpreterError];
      return;
    }

    NSArray<NSNumber *> *shape = @[@2];
    NSError *error;

    if (![weakSelf.interpreter resizeInputTensorAtIndex:0 toShape:shape error:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    if (![weakSelf.interpreter allocateTensorsWithError:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    TFLTensor *inputTensor = [weakSelf.interpreter inputTensorAtIndex:0 error:&error];
    if (inputTensor == nil || error != nil) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
    uint8_t one = 1U;
    uint8_t three = 3U;
    [inputData appendBytes:&one length:sizeof(uint8_t)];
    [inputData appendBytes:&three length:sizeof(uint8_t)];
    if (![inputTensor copyData:inputData error:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    if (![weakSelf.interpreter invokeWithError:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    TFLTensor *outputTensor = [weakSelf.interpreter outputTensorAtIndex:0 error:&error];
    if (outputTensor == nil || error != nil) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    TFLQuantizationParameters *params = outputTensor.quantizationParameters;
    if (params == nil) {
      [weakSelf updateResultsText:
                    [NSString stringWithFormat:kInvokeInterpreterError,
                                               @"Missing qualitization parameters in the output"]];
      return;
    }

    NSData *outputData = [outputTensor dataWithError:&error];
    if (outputData == nil || error != nil) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }
    uint8_t output[2U];
    [outputData getBytes:output length:(sizeof(uint8_t) * 2U)];
    float dequantized[2U];
    dequantized[0] = params.scale * (output[0] - params.zeroPoint);
    dequantized[1] = params.scale * (output[1] - params.zeroPoint);

    [weakSelf updateResultsText:
                  [NSString stringWithFormat:@"Performing 2 add operations on quantized input:\n\n"
                                             @"Input = [%d, %d]\n\nQuantized Output = [%d, %d]\n\n"
                                             @"Dequantized Output = [%f, %f]",
                                             one, three, output[0], output[1], dequantized[0],
                                             dequantized[1]]];
  });
}

- (void)invokeMultiAdd {
  __weak typeof(self) weakSelf = self;
  dispatch_async(self.interpreterSerialQueue, ^{
    if (weakSelf.interpreter == nil) {
      [weakSelf updateResultsText:kNilInterpreterError];
      return;
    }

    NSArray<NSNumber *> *shape = @[@2];
    NSError *error;

    for (int i = 0; i < weakSelf.interpreter.inputTensorCount; ++i) {
      if (![weakSelf.interpreter resizeInputTensorAtIndex:i toShape:shape error:&error]) {
        [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                               error.localizedDescription]];
        return;
      }
    }

    if (![weakSelf.interpreter allocateTensorsWithError:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    NSString *inputs = @"";
    for (int i = 0; i < weakSelf.interpreter.inputTensorCount; ++i) {
      TFLTensor *inputTensor = [weakSelf.interpreter inputTensorAtIndex:i error:&error];
      if (inputTensor == nil || error != nil) {
        [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                               error.localizedDescription]];
        return;
      }

      NSMutableData *inputData = [NSMutableData dataWithCapacity:0];
      float input1 = (float)(i + 1);
      float input2 = (float)(i + 2);
      inputs = [NSString stringWithFormat:@"%@%@[%.1f, %.1f]", inputs,
                                          (inputs.length == 0 ? @"[" : @", "), input1, input2];

      [inputData appendBytes:&input1 length:sizeof(float)];
      [inputData appendBytes:&input2 length:sizeof(float)];
      if (![inputTensor copyData:inputData error:&error]) {
        [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                               error.localizedDescription]];
        return;
      }
    }
    inputs = [NSString stringWithFormat:@"%@]", inputs];

    if (![weakSelf.interpreter invokeWithError:&error]) {
      [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                             error.localizedDescription]];
      return;
    }

    NSString *outputs = @"";
    for (int i = 0; i < weakSelf.interpreter.outputTensorCount; ++i) {
      TFLTensor *outputTensor = [weakSelf.interpreter outputTensorAtIndex:i error:&error];
      if (outputTensor == nil || error != nil) {
        [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                               error.localizedDescription]];
        return;
      }

      NSData *outputData = [outputTensor dataWithError:&error];
      if (outputData == nil || error != nil) {
        [weakSelf updateResultsText:[NSString stringWithFormat:kInvokeInterpreterError,
                                                               error.localizedDescription]];
        return;
      }
      float output[2U];
      [outputData getBytes:output length:(sizeof(float) * 2U)];
      outputs =
          [NSString stringWithFormat:@"%@%@[%.1f, %.1f]", outputs,
                                     (outputs.length == 0 ? @"[" : @", "), output[0], output[1]];
    }
    outputs = [NSString stringWithFormat:@"%@]", outputs];

    [weakSelf
        updateResultsText:
            [NSString
                stringWithFormat:@"Performing 3 add operations:\n\nInputs = %@\n\nOutputs = %@",
                                 inputs, outputs]];
  });
}

- (void)updateResultsText:(NSString *)text {
  __weak typeof(self) weakSelf = self;
  TLTSafeDispatchOnMain(^{
    weakSelf.resultsTextView.text = text;
  });
}

@end

NS_ASSUME_NONNULL_END
