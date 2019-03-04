// Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#import "BenchmarkViewController.h"
#import <algorithm>
#import <sstream>
#import <string>
#import <vector>
#import "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#import "tensorflow/lite/tools/benchmark/logging.h"

namespace {
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

NSDictionary* ParseJson() {
  NSString* params_json_path = FilePathForResourceName(@"benchmark_params.json");
  NSData* data = [NSData dataWithContentsOfFile:params_json_path];
  return [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:nil];
}

std::string FormatCommandLineParam(NSString* key, NSString* value) {
  std::ostringstream stream;
  stream << "--" << [key UTF8String] << "=" << [value UTF8String];
  return stream.str();
}

// Reads the |benchmark_params.json| to read command line parameters and returns them as a vector of
// strings.
void ReadCommandLineParameters(std::vector<std::string>* params) {
  NSDictionary* param_dict = ParseJson();
  for (NSString* key in param_dict) {
    NSString* value = param_dict[key];
    if ([key isEqualToString:@"graph"]) {
      value = FilePathForResourceName(value);
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

class ResultsListener : public tflite::benchmark::BenchmarkListener {
 public:
  void OnBenchmarkEnd(const tflite::benchmark::BenchmarkResults& results) override;
  std::string Results() { return results_; }

 private:
  std::string results_;
};

void OutputMicrosecondsStatToStream(const tensorflow::Stat<int64_t>& time_us,
                                    const std::string& prefix, std::ostringstream* stream) {
  *stream << prefix << "Num runs: " << time_us.count() << "\n";

  *stream << prefix << "Average: " << time_us.avg() / 1e3 << " ms\n";
  *stream << prefix << "Min: " << time_us.min() / 1e3 << " ms \n";
  *stream << prefix << "Max: " << time_us.max() / 1e3 << " ms \n";
  *stream << prefix << "Std deviation: " << time_us.std_deviation() / 1e3 << " ms\n";
}

void ResultsListener::OnBenchmarkEnd(const tflite::benchmark::BenchmarkResults& results) {
  std::ostringstream stream;
  const std::string prefix = " - ";
  stream << "Startup latency: ";
  stream << results.startup_latency_us() / 1e3 << " ms\n";
  stream << "\nInference:\n";
  OutputMicrosecondsStatToStream(results.inference_time_us(), prefix, &stream);
  stream << "\nWarmup:\n";
  OutputMicrosecondsStatToStream(results.warmup_time_us(), prefix, &stream);

  results_ = stream.str();
}

std::string RunBenchmark() {
  ResultsListener listener;
  tflite::benchmark::BenchmarkTfLiteModel benchmark;
  benchmark.AddListener(&listener);
  // TODO(shashishekhar): Passing arguments like this is brittle, refactor the BenchmarkParams
  // so that it contains arguments for BenchmarkTfLiteModel and set parameters using BenchmarkParams
  std::vector<std::string> command_line_params;
  // Benchmark model expects first arg to be program name.
  // push a string for name of program.
  command_line_params.push_back("benchmark_tflite_model");
  ReadCommandLineParameters(&command_line_params);
  std::vector<char*> argv = StringVecToCharPtrVec(command_line_params);
  int argc = static_cast<int>(argv.size());
  benchmark.Run(argc, argv.data());
  return listener.Results();
}
}  // namespace

@interface BenchmarkViewController ()
@end

@implementation BenchmarkViewController
- (IBAction)onBenchmarkModel:(UIButton*)sender {
  std::string results = RunBenchmark();
  [_resultsView setText:[NSString stringWithUTF8String:results.c_str()]];
}
@end
