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

#import "tensorflow/lite/objc/apis/TFLInterpreterOptions.h"

#import <XCTest/XCTest.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Unit tests for TFLInterpreterOptions.
 */
@interface TFLInterpreterOptionsTests : XCTestCase
@end

@implementation TFLInterpreterOptionsTests

#pragma mark - Tests

- (void)testInit {
  TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
  XCTAssertNotNil(options);
  XCTAssertEqual(options.numberOfThreads, 0);
  XCTAssertFalse(options.useXNNPACK);
}

- (void)testSetNumberOfThread {
  TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
  options.numberOfThreads = 2;
  XCTAssertEqual(options.numberOfThreads, 2);
  options.numberOfThreads = 0;
  XCTAssertEqual(options.numberOfThreads, 0);
  options.numberOfThreads = 3;
  XCTAssertEqual(options.numberOfThreads, 3);
}

- (void)testUseXNNPACK {
  TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
  options.useXNNPACK = YES;
  XCTAssertTrue(options.useXNNPACK);
  options.useXNNPACK = NO;
  XCTAssertFalse(options.useXNNPACK);
}

@end

NS_ASSUME_NONNULL_END
