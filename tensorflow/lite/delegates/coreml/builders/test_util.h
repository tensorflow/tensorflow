/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_TEST_UTIL_H_

#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#include "tensorflow/lite/kernels/test_util.h"

#import <XCTest/XCTest.h>

namespace tflite {
namespace delegates {
namespace coreml {
class SingleOpModelWithCoreMlDelegate : public tflite::SingleOpModel {
 public:
  static const char kDelegateName[];

  SingleOpModelWithCoreMlDelegate();
  tflite::Interpreter* interpreter() { return interpreter_.get(); }

 protected:
  using SingleOpModel::builder_;

 private:
  tflite::Interpreter::TfLiteDelegatePtr delegate_;
  TfLiteCoreMlDelegateOptions params_ = {
      .enabled_devices = TfLiteCoreMlDelegateAllDevices,
      .min_nodes_per_partition = 1,
  };
};

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

@interface BaseOpTest : XCTestCase
@property tflite::delegates::coreml::SingleOpModelWithCoreMlDelegate* model;
- (void)validateInterpreter:(tflite::Interpreter*)interpreter;
- (void)checkInterpreterNotDelegated:(tflite::Interpreter*)interpreter;
- (void)invokeAndValidate;
- (void)invokeAndCheckNotDelegated;
@end

#endif  // TENSORFLOW_LITE_DELEGATES_COREML_BUILDERS_TEST_UTIL_H_
