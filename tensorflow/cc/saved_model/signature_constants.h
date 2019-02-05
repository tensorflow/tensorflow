/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_SAVED_MODEL_SIGNATURE_CONSTANTS_H_
#define TENSORFLOW_CC_SAVED_MODEL_SIGNATURE_CONSTANTS_H_

namespace tensorflow {

/// Key in the signature def map for `default` serving signatures. The default
/// signature is used in inference requests where a specific signature was not
/// specified.
static constexpr char kDefaultServingSignatureDefKey[] = "serving_default";

////////////////////////////////////////////////////////////////////////////////
/// Classification API constants.

/// Classification inputs.
static constexpr char kClassifyInputs[] = "inputs";

/// Classification method name used in a SignatureDef.
static constexpr char kClassifyMethodName[] = "tensorflow/serving/classify";

/// Classification classes output.
static constexpr char kClassifyOutputClasses[] = "classes";

/// Classification scores output.
static constexpr char kClassifyOutputScores[] = "scores";

////////////////////////////////////////////////////////////////////////////////
/// Predict API constants.

/// Predict inputs.
static constexpr char kPredictInputs[] = "inputs";

/// Predict method name used in a SignatureDef.
static constexpr char kPredictMethodName[] = "tensorflow/serving/predict";

/// Predict outputs.
static constexpr char kPredictOutputs[] = "outputs";

////////////////////////////////////////////////////////////////////////////////
/// Regression API constants.

/// Regression inputs.
static constexpr char kRegressInputs[] = "inputs";

/// Regression method name used in a SignatureDef.
static constexpr char kRegressMethodName[] = "tensorflow/serving/regress";

/// Regression outputs.
static constexpr char kRegressOutputs[] = "outputs";

////////////////////////////////////////////////////////////////////////////////

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_SIGNATURE_CONSTANTS_H_
