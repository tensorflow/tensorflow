// Copyright 2018 Google Inc. All rights reserved.
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

#import "tensorflow/lite/experimental/objc/apis/TFLQuantizationParameters.h"

#import <XCTest/XCTest.h>

#import "tensorflow/lite/experimental/objc/sources/TFLQuantizationParameters+Internal.h"

NS_ASSUME_NONNULL_BEGIN

/** Test scale of quantization parameters. */
static const float kTestScale = 2.0;

/** Test zero point of quantization parameters. */
static const int32_t kTestZeroPoint = 128;

/**
 * Unit tests for TFLQuantizationParameters.
 */
@interface TFLQuantizationParametersTests : XCTestCase
@end

@implementation TFLQuantizationParametersTests

#pragma mark - Tests

- (void)testInitWithScaleAndZeroPoint {
  TFLQuantizationParameters *params =
      [[TFLQuantizationParameters alloc] initWithScale:kTestScale zeroPoint:kTestZeroPoint];
  XCTAssertEqual(params.scale, kTestScale);
  XCTAssertEqual(params.zeroPoint, kTestZeroPoint);
}

@end

NS_ASSUME_NONNULL_END
