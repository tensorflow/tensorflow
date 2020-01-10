#include "tensorflow/examples/android/jni/tensorflow_jni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <jni.h>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/examples/android/jni/jni_utils.h"

// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> session;

static std::vector<std::string> g_label_strings;
static bool g_compute_graph_initialized = false;
//static mutex g_compute_graph_mutex(base::LINKER_INITIALIZED);

static int g_tensorflow_input_size;  // The image size for the mognet input.
static int g_image_mean;  // The image mean.

using namespace tensorflow;

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

  const char* const model_cstr = env->GetStringUTFChars(model, NULL);
  const char* const labels_cstr = env->GetStringUTFChars(labels, NULL);

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

  AAssetManager* const asset_manager =
      AAssetManager_fromJava(env, java_asset_manager);
  LOG(INFO) << "Acquired AssetManager.";

  LOG(INFO) << "Reading file to proto: " << model_cstr;
  ReadFileToProto(asset_manager, model_cstr, &tensorflow_graph);

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
  ReadFileToVector(asset_manager, labels_cstr, &g_label_strings);
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

  tensorflow::Status s =
      session->Run(input_tensors, output_names, {}, &output_tensors);
  VLOG(0) << "End computing.";

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
  jint* pixels = env->GetIntArrayElements(image, &iCopied);

  std::string result = ClassifyImage(
      reinterpret_cast<const RGBA*>(pixels), width * 4, width, height);

  env->ReleaseIntArrayElements(image, pixels, JNI_ABORT);

  return env->NewStringUTF(result.c_str());
}

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageBmp)(
    JNIEnv* env, jobject thiz, jobject bitmap) {
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
}
