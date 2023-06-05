// Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#import "EvaluationViewController.h"
#import <algorithm>
#include <fstream>
#import <sstream>
#import <string>
#import <vector>

#import <TensorFlowLiteInferenceDiffC/TensorFlowLiteInferenceDiffC.h>

namespace {

NSString* const kDocumentsPrefix = @"/Documents/";
NSString* const kModelFileKey = @"model_file";
NSString* const kOutputFilePathKey = @"output_file_path";

NSString* FilePathForResourceName(NSString* filename) {
  NSString* name = [filename stringByDeletingPathExtension];
  NSString* extension = [filename pathExtension];
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    TFLITE_LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
                      << "' in bundle.";
  }
  return file_path;
}

NSDictionary* ParseEvaluationParamsJson() {
  NSString* params_json_path = FilePathForResourceName(@"evaluation_params.json");
  NSData* data = [NSData dataWithContentsOfFile:params_json_path];
  NSDictionary* dict = [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:nil];
  return dict;
}

std::string FormatCommandLineParam(NSString* key, NSString* value) {
  std::ostringstream stream;
  stream << "--" << [key UTF8String] << "=" << [value UTF8String];
  return stream.str();
}

// Reads the |evaluation_params.json| to read command line parameters and returns them as a vector
// of strings.
// Returns the evaluation parameters as key-value pairs.
void ReadCommandLineParameters(std::vector<std::string>* params) {
  NSDictionary* param_dict = ParseEvaluationParamsJson();
  for (NSString* key in param_dict) {
    NSString* value = param_dict[key];
    if ([key isEqualToString:kModelFileKey]) {
      value = FilePathForResourceName(value);
    }
    if ([key isEqualToString:kOutputFilePathKey]) {
      if (![value hasPrefix:kDocumentsPrefix]) {
        TFLITE_LOG(FATAL) << "Output file must be under the Document directory";
      }
      // Replace the prefix "/Documents/" with the actual documents path on the device.
      NSString* documents = [NSSearchPathForDirectoriesInDomains(
          NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0];
      NSString* relpath = [value substringFromIndex:[kDocumentsPrefix length]];
      value = [documents stringByAppendingPathComponent:relpath];

      // Create the output directory if necessary.
      NSString* path = value.stringByDeletingLastPathComponent;
      if (![[NSFileManager defaultManager] createDirectoryAtPath:path
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil]) {
        TFLITE_LOG(FATAL) << "Cannot create output directory: " << [path UTF8String];
      }

      // Create the output file.
      std::string output_file_path_ = std::string([value UTF8String]);
      std::unique_ptr<std::ofstream> output_file_ =
          std::make_unique<std::ofstream>(output_file_path_, std::ios::out | std::ios::binary);
      if (!output_file_->is_open()) {
        TFLITE_LOG(ERROR) << "Cannot open output file: " << output_file_path_;
      } else {
        TFLITE_LOG(INFO) << "Create output file: " << output_file_path_;
      }
    }
    params->push_back(FormatCommandLineParam(key, value));
  }
}

std::vector<char*> StringVecToCharPtrVec(const std::vector<std::string>& str_vec) {
  std::vector<char*> charptr_vec;
  std::transform(str_vec.begin(), str_vec.end(), std::back_inserter(charptr_vec),
                 [](const std::string& s) -> char* { return const_cast<char*>(s.c_str()); });
  return charptr_vec;
}

std::string EvaluationMetricsToString(TfLiteEvaluationMetrics* metrics) {
  std::ostringstream stream;
  stream << "Num evaluation runs: " << TfLiteEvaluationMetricsGetNumRuns(metrics);
  TfLiteEvaluationMetricsLatency ref_latency = TfLiteEvaluationMetricsGetReferenceLatency(metrics);
  stream << "\nReference run latency: avg=" << ref_latency.avg_us
         << "(us), std_dev=" << ref_latency.std_deviation_us << "(us)";
  TfLiteEvaluationMetricsLatency test_latency = TfLiteEvaluationMetricsGetTestLatency(metrics);
  stream << "\nTest run latency: avg=" << test_latency.avg_us
         << "(us), std_dev=" << test_latency.std_deviation_us << "(us)";
  for (int i = 0; i < TfLiteEvaluationMetricsGetOutputErrorCount(metrics); ++i) {
    TfLiteEvaluationMetricsAccuracy error = TfLiteEvaluationMetricsGetOutputError(metrics, i);
    stream << "\nOutputDiff[" << i << "]: avg_error=" << error.avg_value
           << ", std_dev=" << error.std_deviation;
  }

  return stream.str();
}

TfLiteEvaluationMetrics* RunEvaluation() {
  std::vector<std::string> command_line_params;
  ReadCommandLineParameters(&command_line_params);
  std::vector<char*> argv = StringVecToCharPtrVec(command_line_params);
  int argc = static_cast<int>(argv.size());
  TfLiteEvaluationTask* task = TfLiteEvaluationTaskCreate();
  return TfLiteEvaluationTaskRunWithArgs(task, argc, argv.data());
}
}  // namespace

@interface EvaluationViewController ()
@end

@implementation EvaluationViewController
- (IBAction)onEvaluateModel:(UIButton*)sender {
  TfLiteEvaluationMetrics* metrics = RunEvaluation();
  [_resultsView setText:[NSString stringWithUTF8String:EvaluationMetricsToString(metrics).c_str()]];
}
@end
