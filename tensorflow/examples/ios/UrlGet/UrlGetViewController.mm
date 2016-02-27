// Copyright 2015 Google Inc. All rights reserved.
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

#import "UrlGetViewController.h"

#include <fstream>
#include <sstream>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"

#import "tensorflow/examples/ios/UrlGet/test.pb.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "ios_image_load.h"

#include <iostream>

NSString* TestFunction();

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace

@interface UrlGetViewController ()
@end

@implementation UrlGetViewController {
}

- (IBAction)getUrl:(id)sender {
  test::KeyVal keyval;
  std::cerr << keyval.DebugString() << std::endl;

  NSString* networkPath = TestFunction();

  self.urlContentTextView.text = networkPath;
}

@end

#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
//#include "tensorflow/examples/android/jni/jni_utils.h"


#define JNIEXPORT
#define JNICALL
#define TENSORFLOW_METHOD(x) x
#define JNI_FALSE false
#define JNI_TRUE true
typedef int jint;
typedef std::string jstring;
typedef void* jobject;
typedef int JNIEnv;
typedef bool jboolean;
typedef std::vector<int> jintArray;


// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> session;

static std::vector<std::string> g_label_strings;
static bool g_compute_graph_initialized = false;
//static mutex g_compute_graph_mutex(base::LINKER_INITIALIZED);

static int g_tensorflow_input_size;  // The image size for the mognet input.
static int g_image_mean;  // The image mean.

using namespace tensorflow;

inline static int64 CurrentThreadTimeUs() {
  struct timeval tv;
  //gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initializeTensorflow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager,
    jstring model, jstring labels,
    jint num_classes, jint mognet_input_size, jint image_mean) {
  //MutexLock input_lock(&g_compute_graph_mutex);
  if (g_compute_graph_initialized) {
    LOG(INFO) << "Compute graph already loaded. skipping.";
    return 0;
  }

  const char* const model_cstr = "";//env->GetStringUTFChars(model, NULL);
  const char* const labels_cstr = "";//env->GetStringUTFChars(labels, NULL);

  g_tensorflow_input_size = mognet_input_size;
  g_image_mean = image_mean;

  LOG(INFO) << "Loading Tensorflow.";

  LOG(INFO) << "Making new SessionOptions.";
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session.reset(tensorflow::NewSession(options));
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  //AAssetManager* const asset_manager =
  //    AAssetManager_fromJava(env, java_asset_manager);
  //LOG(INFO) << "Acquired AssetManager.";

  //LOG(INFO) << "Reading file to proto: " << model_cstr;
  //ReadFileToProto(asset_manager, model_cstr, &tensorflow_graph);

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return -1;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph loaded from: " << model_cstr;

  // Read the label list
  //ReadFileToVector(asset_manager, labels_cstr, &g_label_strings);
  LOG(INFO) << g_label_strings.size() << " label strings loaded from: "
            << labels_cstr;
  g_compute_graph_initialized = true;

  return 0;
}

namespace {
typedef struct {
  uint8 red;
  uint8 green;
  uint8 blue;
  uint8 alpha;
} RGBA;
}  // namespace

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                           Eigen::Aligned>& prediction,
    const int num_results, const float threshold,
    std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>,
      std::vector<std::pair<float, int> >,
      std::greater<std::pair<float, int> > > top_result_pq;

  const int count = prediction.size();
  for (int i = 0; i < count; ++i) {
    const float value = prediction(i);

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

static std::string ClassifyImage(const RGBA* const bitmap_src,
                                 const int in_stride,
                                 const int width, const int height) {
  // Very basic benchmarking functionality.
  static int num_runs = 0;
  static int64 timing_total_us = 0;
  ++num_runs;

  // Create input tensor
  tensorflow::Tensor input_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({
          1, g_tensorflow_input_size, g_tensorflow_input_size, 3}));

  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  LOG(INFO) << "Tensorflow: Copying Data.";
  for (int i = 0; i < g_tensorflow_input_size; ++i) {
    const RGBA* src = bitmap_src + i * g_tensorflow_input_size;
    for (int j = 0; j < g_tensorflow_input_size; ++j) {
       // Copy 3 values
      input_tensor_mapped(0, i, j, 0) =
          static_cast<float>(src->red) - g_image_mean;
      input_tensor_mapped(0, i, j, 1) =
          static_cast<float>(src->green) - g_image_mean;
      input_tensor_mapped(0, i, j, 2) =
          static_cast<float>(src->blue) - g_image_mean;
      ++src;
    }
  }

  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors(
      {{"input:0", input_tensor}});

  VLOG(0) << "Start computing.";
  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<std::string> output_names({"output:0"});

  const int64 start_time = CurrentThreadTimeUs();
  tensorflow::Status s =
      session->Run(input_tensors, output_names, {}, &output_tensors);
  const int64 end_time = CurrentThreadTimeUs();

  const int64 elapsed_time_inf = end_time - start_time;
  timing_total_us += elapsed_time_inf;
  VLOG(0) << "End computing. Ran in " << elapsed_time_inf / 1000 << "ms ("
          << (timing_total_us / num_runs / 1000) << "ms avg over " << num_runs
          << " runs)";

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return "";
  }

  VLOG(0) << "Reading from layer " << output_names[0];
  tensorflow::Tensor* output = &output_tensors[0];
  const int kNumResults = 5;
  const float kThreshold = 0.1f;
  std::vector<std::pair<float, int> > top_results;
  GetTopN(output->flat<float>(), kNumResults, kThreshold, &top_results);

  std::stringstream ss;
  ss.precision(3);
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;

    ss << index << " " << confidence << " ";

    // Write out the result as a string
    if (index < g_label_strings.size()) {
      // just for safety: theoretically, the output is under 1000 unless there
      // is some numerical issues leading to a wrong prediction.
      ss << g_label_strings[index];
    } else {
      ss << "Prediction: " << index;
    }

    ss << "\n";
  }

  LOG(INFO) << "Predictions: " << ss.str();
  return ss.str();
}

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageRgb)(
    JNIEnv* env, jobject thiz, jintArray image, jint width, jint height) {
  // Copy image into currFrame.
  jboolean iCopied = JNI_FALSE;
  jint* pixels = nullptr;//env->GetIntArrayElements(image, &iCopied);

  std::string result = ClassifyImage(
      reinterpret_cast<const RGBA*>(pixels), width * 4, width, height);

  //env->ReleaseIntArrayElements(image, pixels, JNI_ABORT);

  return "";//env->NewStringUTF(result.c_str());
}

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageBmp)(
    JNIEnv* env, jobject thiz, jobject bitmap) {
  return "";
#if 0
  // Obtains the bitmap information.
  AndroidBitmapInfo info;
  CHECK_EQ(AndroidBitmap_getInfo(env, bitmap, &info),
           ANDROID_BITMAP_RESULT_SUCCESS);
  void* pixels;
  CHECK_EQ(AndroidBitmap_lockPixels(env, bitmap, &pixels),
           ANDROID_BITMAP_RESULT_SUCCESS);
  LOG(INFO) << "Height: " << info.height;
  LOG(INFO) << "Width: " << info.width;
  LOG(INFO) << "Stride: " << info.stride;
  // TODO(jiayq): deal with other formats if necessary.
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    return env->NewStringUTF(
        "Error: Android system is not using RGBA_8888 in default.");
  }

  std::string result = ClassifyImage(
      static_cast<const RGBA*>(pixels), info.stride, info.width, info.height);

  // Finally, unlock the pixels
  CHECK_EQ(AndroidBitmap_unlockPixels(env, bitmap),
           ANDROID_BITMAP_RESULT_SUCCESS);

  return env->NewStringUTF(result.c_str());
#endif
}

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively. 
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

NSString* TestFunction() {
//    initializeTensorflow(nullptr, nullptr, nullptr, "model", "labels", 0, 0, 0);

  tensorflow::CreateDirectSessionFactory();

  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices.";

  Session* session_pointer = nullptr;
  Status session_status = tensorflow::NewSession(options, &session_pointer);
  if (!session_status.ok()) {
    std::string status_string = session_status.ToString();
    return [NSString stringWithFormat: @"Session create failed - %s",
	status_string.c_str()];
  }
  session.reset(tensorflow::NewSession(options));
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  NSString* networkPath = [[NSBundle mainBundle] pathForResource:@"tensorflow_inception_graph" ofType:@"pb"];
  if (networkPath == NULL) {
    fprintf(stderr, "Couldn't find the neural network parameters file - did you add it as a resource to your application?\n");
    assert(false);
  }
  PortableReadFileToProto([networkPath UTF8String], &tensorflow_graph);

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return @"";
  }

  // Read the label list
  NSString* labelsPath = [[NSBundle mainBundle] pathForResource:@"imagenet_comp_graph_label_strings" ofType:@"txt"];
  if (labelsPath == NULL) {
    fprintf(stderr, "Couldn't find the network labels file - did you add it as a resource to your application?\n");
    assert(false);
  }

  std::vector<std::string> label_strings;
  std::ifstream t;
  t.open([labelsPath UTF8String]);
  std::string line;
  while(t){
    std::getline(t, line);
    label_strings.push_back(line);
  }
  t.close();

  // Read the Grace Hopper image.
  NSString* imagePath = [[NSBundle mainBundle] pathForResource:@"grace_hopper" ofType:@"jpg"];
  if (imagePath == NULL) {
    fprintf(stderr, "Couldn't find the image file - did you add it as a resource to your application?\n");
    assert(false);
  }

  int image_width;
  int image_height;
  int image_channels;
  std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
	[imagePath UTF8String], &image_width, &image_height, &image_channels);
  tensorflow::Tensor image_tensor(
      tensorflow::DT_UINT8,
      tensorflow::TensorShape({
          1, image_height, image_width, image_channels}));
  auto image_tensor_mapped = image_tensor.tensor<tensorflow::uint8, 4>();
  memcpy(image_tensor_mapped.data(), image_data.data(),
	(image_height * image_width * image_channels));

  NSString* result = [networkPath stringByAppendingString: @" - loaded!"];
  result = [NSString stringWithFormat: @"%@ - %d, %s - %dx%d", result,
	label_strings.size(), label_strings[0].c_str(), image_width, image_height];

  fprintf(stderr, "a\n");

  std::string input_layer = "Cast";
  std::string output_layer = "softmax";
  std::vector<Tensor> outputs;
  fprintf(stderr, "b\n");
  Status run_status = session->Run({{input_layer, image_tensor}},
                                   {output_layer}, {}, &outputs);
  fprintf(stderr, "c\n");
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
  }
  string status_string = run_status.ToString();
  result = [NSString stringWithFormat: @"%@ - %s", result,
	status_string.c_str()];

  fprintf(stderr, "d\n");

  return result;
}

#include <sys/syslog.h>

tensorflow::Status TPDValidateOpName(const tensorflow::string& op_name);

__attribute__((constructor))
static void SomeFactoryInitFunc() {
  syslog(LOG_ERR, "SomeFactoryInitFunc was called from UrlGetViewController!");
  volatile int foo = 0;
  if (foo == 1) {
    //TPDValidateOpName("foo");
  }
}

