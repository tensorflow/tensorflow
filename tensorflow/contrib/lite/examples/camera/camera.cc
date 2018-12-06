/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include <libv4l2.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fstream>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "tensorflow/contrib/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/contrib/lite/examples/label_image/get_top_n.h"

#define LOG(x) std::cerr

#if TfLite_RPI_APPS_DEBUG
#define DEBUG(INFO) LOG(INFO)
#endif

namespace tflite {
namespace label_image {

std::vector<uint8_t> in;
uint8_t* frame_data;

// Used to store the memory-mapped buffers we use for capture.
struct CameraBuffer {
  void* start;
  size_t length;
};

TfLiteStatus SendCameraCommand(int fh, int request, void* arg) {
  int r;
  do {
    r = v4l2_ioctl(fh, request, arg);
  } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));
  if (r == -1) {
    LOG(ERROR) << "SendCameraCommand error " << errno << " (" << strerror(errno)
               << ")";
    //    return errors::Unknown("SendCameraCommand error ", errno,
    //                                       strerror(errno));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus OpenCamera(int* camera_handle) {
  const char* dev_name = "/dev/video0";
  int fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
  if (fd < 0) {
    LOG(ERROR) << "Cannot open camera device";
    //    return tensorflow::errors::NotFound("V4L2 camera device not found");
    return kTfLiteError;
  }
  *camera_handle = fd;
  return kTfLiteOk;
}

TfLiteStatus CloseCamera(int camera_handle) {
  v4l2_close(camera_handle);
  return kTfLiteOk;
}

TfLiteStatus SetCameraFormat(int camera_handle, int wanted_width,
                             int wanted_height) {
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = wanted_width;
  fmt.fmt.pix.height = wanted_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  TfLiteStatus set_format_status =
      SendCameraCommand(camera_handle, VIDIOC_S_FMT, &fmt);
  if (kTfLiteError == set_format_status) {
    LOG(ERROR) << "Setting format failed with " << set_format_status;
    return kTfLiteError;
  }
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
    LOG(ERROR) << "Libv4l didn't accept RGB24 format. Can't proceed.";
    //    return tensorflow::errors::Unknown("Libv4l didn't accept RGB24
    //    format");
    return kTfLiteError;
  }
  if ((fmt.fmt.pix.width != wanted_width) ||
      (fmt.fmt.pix.height != wanted_height)) {
    LOG(WARNING) << "Warning: driver is sending image at " << fmt.fmt.pix.width
                 << "x" << fmt.fmt.pix.height;
  }
  return kTfLiteOk;
}

TfLiteStatus StartCameraCapture(int camera_handle, int buffer_count,
                                CameraBuffer** buffers) {
  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));
  req.count = buffer_count;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  TfLiteStatus request_buffers_status =
      SendCameraCommand(camera_handle, VIDIOC_REQBUFS, &req);
  if (kTfLiteError == request_buffers_status) {
    LOG(ERROR) << "Request buffers failed with " << kTfLiteError;
    return kTfLiteError;
  }

  *buffers = (CameraBuffer*)(calloc(buffer_count, sizeof(*buffers)));
  for (int n_buffers = 0; n_buffers < buffer_count; ++n_buffers) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = n_buffers;
    TfLiteStatus query_buffer_status =
        SendCameraCommand(camera_handle, VIDIOC_QUERYBUF, &buf);
    if (kTfLiteError == query_buffer_status) {
      LOG(ERROR) << "Query buffer failed with " << kTfLiteError;
      return query_buffer_status;
    }
    (*buffers)[n_buffers].length = buf.length;
    (*buffers)[n_buffers].start =
        v4l2_mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                  camera_handle, buf.m.offset);

    if (MAP_FAILED == (*buffers)[n_buffers].start) {
      LOG(ERROR) << "Memory-mapping buffer failed";
      //      return tensorflow::errors::Unknown("Memory-mapping buffer
      //      failed");
      return kTfLiteError;
    }
  }

  for (int i = 0; i < buffer_count; ++i) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    TfLiteStatus set_buffer_status =
        SendCameraCommand(camera_handle, VIDIOC_QBUF, &buf);
    if (kTfLiteError == set_buffer_status) {
      LOG(ERROR) << "Set buffer failed with " << kTfLiteError;
      return kTfLiteError;
    }
  }

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  TfLiteStatus stream_on_status =
      SendCameraCommand(camera_handle, VIDIOC_STREAMON, &type);
  if (kTfLiteError == stream_on_status) {
    LOG(ERROR) << "Turning stream on failed with " << kTfLiteError;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EndCameraCapture(int camera_handle, CameraBuffer* buffers,
                              int buffer_count) {
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  TfLiteStatus stream_off_status =
      SendCameraCommand(camera_handle, VIDIOC_STREAMOFF, &type);
  if (kTfLiteOk != stream_off_status) {
    LOG(ERROR) << "Turning stream off failed with " << stream_off_status;
    return stream_off_status;
  }
  for (int i = 0; i < buffer_count; ++i)
    v4l2_munmap(buffers[i].start, buffers[i].length);
  return kTfLiteOk;
}

TfLiteStatus CaptureNextFrame(int camera_handle, CameraBuffer* buffers,
                              uint8_t** frame, int* frame_size,
                              v4l2_buffer* buf) {
  int r;
  do {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(camera_handle, &fds);
    struct timeval tv;
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    r = select(camera_handle + 1, &fds, NULL, NULL, &tv);
  } while ((r == -1 && (errno = EINTR)));
  if (r == -1) {
    LOG(ERROR) << "select() failed while waiting for the camera with " << errno;
    //    return tensorflow::errors::Unknown(
    //        "CaptureCameraFrame: select() failed with", errno);
    return kTfLiteError;
  }

  memset(buf, 0, sizeof(*buf));
  buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf->memory = V4L2_MEMORY_MMAP;
  TfLiteStatus get_buffer_status =
      SendCameraCommand(camera_handle, VIDIOC_DQBUF, buf);
  if (kTfLiteError == get_buffer_status) {
    LOG(ERROR) << "Get buffer failed with " << get_buffer_status;
    return get_buffer_status;
  }

  *frame = static_cast<uint8_t*>(buffers[buf->index].start);
  *frame_size = buf->bytesused;

  return kTfLiteOk;
}

TfLiteStatus ReleaseFrame(int camera_handle, v4l2_buffer* buf) {
  TfLiteStatus release_buffer_status =
      SendCameraCommand(camera_handle, VIDIOC_QBUF, buf);
  if (kTfLiteOk != release_buffer_status) {
    LOG(ERROR) << "Release buffer failed with " << release_buffer_status;
    return kTfLiteError;
  }
  return kTfLiteOk;
}

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Node " << std::setw(3) << std::setprecision(3) << op_index
            << ", OpCode " << std::setw(3) << std::setprecision(3)
            << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}

void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "no model file name\n";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << s->model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(s->accel);

  if (s->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  // Start to handel camera.

  int video_width = 640;
  int video_height = 480;
  int video_channels = 3;
  int input_width = 224;
  int input_height = 224;
  int input_mean = 128;
  int input_std = 128;

  int camera_handle;

  TfLiteStatus open_status = OpenCamera(&camera_handle);
  if (kTfLiteError == open_status) {
    LOG(ERROR) << "OpenCamera failed with " << open_status;
    exit(-1);
  }

  TfLiteStatus format_status =
      SetCameraFormat(camera_handle, video_width, video_height);
  if (kTfLiteError == format_status) {
    LOG(ERROR) << "SetCameraFormat failed with " << format_status;
    exit(-1);
  }

  const int how_many_buffers = 2;
  CameraBuffer* buffers;
  TfLiteStatus start_capture_status =
      StartCameraCapture(camera_handle, how_many_buffers, &buffers);
  if (kTfLiteError == start_capture_status) {
    LOG(ERROR) << "StartCameraCapture failed with " << start_capture_status;
    exit(-1);
  }

  for (int i = 0; i < 100; i++) {
    LOG(INFO) << "The " << i << "th frame ! \n";
    // Captur a frame.
    //{
    //      uint8_t* frame_data;
    int frame_data_size;
    v4l2_buffer buf;
    TfLiteStatus capture_next_status = CaptureNextFrame(
        camera_handle, buffers, &frame_data, &frame_data_size, &buf);
    if (kTfLiteError == capture_next_status) {
      LOG(ERROR) << "CaptureNextFrame failed with " << capture_next_status;
      exit(-1);
    }

    TfLiteStatus release_frame_status = ReleaseFrame(camera_handle, &buf);
    if (kTfLiteOk != release_frame_status) {
      LOG(ERROR) << "ReleaseFrame failed with " << release_frame_status;
      exit(-1);
    }

#if TfLite_RPI_APPS_DEBUG
    DEBUG(INFO) << "Going to tensor processing -- \n";
#endif

    int input = interpreter->inputs()[0];
    if (s->verbose) LOG(INFO) << "input: " << input << "\n";
    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    if (s->verbose) {
      LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
      LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      LOG(FATAL) << "Failed to allocate tensors!";
    }

    if (s->verbose) PrintInterpreterState(interpreter.get());

    // get input dimension from the input tensor metadata
    // assuming one input only
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

    switch (interpreter->tensor(input)->type) {
      case kTfLiteFloat32:
        s->input_floating = true;
        resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                      video_height, video_width, video_channels, wanted_height,
                      wanted_width, wanted_channels, s);
        break;
      case kTfLiteUInt8:

#if TfLite_RPI_APPS_DEBUG
        DEBUG(INFO) << "Entering kTfLiteUInt8:" << wanted_height << " ,"
                    << wanted_width << " ," << wanted_channels << " \n";
#endif

        resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), frame_data,
                        video_height, video_width, video_channels,
                        wanted_height, wanted_width, wanted_channels, s);

#if TfLite_RPI_APPS_DEBUG
        DEBUG(INFO) << "resize is done \n";
#endif

        break;
      default:
        LOG(FATAL) << "cannot handle input type "
                   << interpreter->tensor(input)->type << " yet";
        exit(-1);
    }

    profiling::Profiler* profiler = new profiling::Profiler();
    interpreter->SetProfiler(profiler);

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
      if (interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke tflite!\n";
      }
    }
    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: "
              << (get_us(stop_time) - get_us(start_time)) /
                     (s->loop_count * 1000)
              << " ms \n";
    if (s->profiling) {
      profiler->StopProfiling();
      auto profile_events = profiler->GetProfileEvents();
      for (int i = 0; i < profile_events.size(); i++) {
        auto op_index = profile_events[i]->event_metadata;
        const auto node_and_registration =
            interpreter->node_and_registration(op_index);
        const TfLiteRegistration registration = node_and_registration->second;
        PrintProfilingInfo(profile_events[i], op_index, registration);
      }
    }

    const float threshold = 0.001f;

    std::vector<std::pair<float, int>> top_results;

    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    switch (interpreter->tensor(output)->type) {
      case kTfLiteFloat32:
        get_top_n<float>(interpreter->typed_output_tensor<float>(0),
                         output_size, s->number_of_results, threshold,
                         &top_results, true);
        break;
      case kTfLiteUInt8:
        get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                           output_size, s->number_of_results, threshold,
                           &top_results, false);
        break;
      default:
        LOG(FATAL) << "cannot handle output type "
                   << interpreter->tensor(input)->type << " yet";
        exit(-1);
    }

    std::vector<string> labels;
    size_t label_count;

    if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)
      exit(-1);

    for (const auto& result : top_results) {
      const float confidence = result.first;
      const int index = result.second;
      LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
    }
  }  // End: 1 frame processing.

  TfLiteStatus end_capture_status =
      EndCameraCapture(camera_handle, buffers, how_many_buffers);
  if (kTfLiteOk != end_capture_status) {
    LOG(ERROR) << "EndCameraCapture failed with " << end_capture_status;
    exit(-1);
  }

  TfLiteStatus close_status = CloseCamera(camera_handle);
  if (kTfLiteError == close_status) {
    LOG(ERROR) << "CloseCamera failed with " << open_status;
    exit(-1);
  }
}

void display_usage() {
  LOG(INFO) << "label_image\n"
            << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
            << "--count, -c: loop interpreter->Invoke() for certain times\n"
            << "--input_mean, -b: input mean\n"
            << "--input_std, -s: input standard deviation\n"
            << "--image, -i: image_name.bmp\n"
            << "--labels, -l: labels for the model\n"
            << "--tflite_model, -m: model_name.tflite\n"
            << "--profiling, -p: [0|1], profiling or not\n"
            << "--num_results, -r: number of results to show\n"
            << "--threads, -t: number of threads\n"
            << "--verbose, -v: [0|1] print more information\n"
            << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"count", required_argument, nullptr, 'c'},
        {"verbose", required_argument, nullptr, 'v'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:f:i:l:m:p:r:s:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image::Main(argc, argv);
}
