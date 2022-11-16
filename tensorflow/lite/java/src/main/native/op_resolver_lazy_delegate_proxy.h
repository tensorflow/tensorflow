/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_OP_RESOLVER_LAZY_DELEGATE_PROXY_H_
#define TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_OP_RESOLVER_LAZY_DELEGATE_PROXY_H_

#include <memory>
#include <utility>

#include "tensorflow/lite/op_resolver.h"

namespace tflite {
namespace jni {

class OpResolverLazyDelegateProxy : public OpResolver {
 public:
  OpResolverLazyDelegateProxy(std::unique_ptr<tflite::OpResolver>&& op_resolver,
                              bool use_xnnpack)
      : op_resolver_(std::move(op_resolver)), use_xnnpack_(use_xnnpack) {}

  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
  const TfLiteRegistration* FindOp(const char* op, int version) const override;

  OpResolver::TfLiteDelegateCreators GetDelegateCreators() const override;
  OpResolver::TfLiteOpaqueDelegateCreators GetOpaqueDelegateCreators()
      const override;

 private:
  bool MayContainUserDefinedOps() const override;

  static std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
  createXNNPackDelegate(TfLiteContext* context);

  static OpResolver::TfLiteOpaqueDelegatePtr createXNNPackOpaqueDelegate(
      int num_threads);

  std::unique_ptr<tflite::OpResolver> op_resolver_;
  bool use_xnnpack_ = false;
};

}  // namespace jni
}  // namespace tflite

#endif  // TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_OP_RESOLVER_LAZY_DELEGATE_PROXY_H_
