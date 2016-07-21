/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Full build instructions are at tensorflow/contrib/pi_examples/README.md.

#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <libv4l2.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Used to store the memory-mapped buffers we use for capture.
struct CameraBuffer {
  void* start;
  size_t length;
};

// Wrapper around camera command sending.
Status SendCameraCommand(int fh, int request, void* arg) {
  int r;
  do {
    r = v4l2_ioctl(fh, request, arg);
  } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));
  if (r == -1) {
    LOG(ERROR) << "SendCameraCommand error " << errno << " (" << strerror(errno)
               << ")";
    return tensorflow::errors::Unknown("SendCameraCommand error ", errno,
                                       strerror(errno));
  }
  return Status::OK();
}

Status OpenCamera(int* camera_handle) {
  const char* dev_name = "/dev/video0";
  int fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
  if (fd < 0) {
    LOG(ERROR) << "Cannot open camera device";
    return tensorflow::errors::NotFound("V4L2 camera device not found");
  }
  *camera_handle = fd;
  return Status::OK();
}

Status CloseCamera(int camera_handle) {
  v4l2_close(camera_handle);
  return Status::OK();
}

Status SetCameraFormat(int camera_handle, int wanted_width, int wanted_height) {
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = wanted_width;
  fmt.fmt.pix.height = wanted_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  Status set_format_status =
      SendCameraCommand(camera_handle, VIDIOC_S_FMT, &fmt);
  if (!set_format_status.ok()) {
    LOG(ERROR) << "Setting format failed with " << set_format_status;
    return set_format_status;
  }
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
    LOG(ERROR) << "Libv4l didn't accept RGB24 format. Can't proceed.";
    return tensorflow::errors::Unknown("Libv4l didn't accept RGB24 format");
  }
  if ((fmt.fmt.pix.width != wanted_width) ||
      (fmt.fmt.pix.height != wanted_height)) {
    LOG(WARNING) << "Warning: driver is sending image at " << fmt.fmt.pix.width
                 << "x" << fmt.fmt.pix.height;
  }
  return Status::OK();
}

Status StartCameraCapture(int camera_handle, int buffer_count,
                          CameraBuffer** buffers) {
  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));
  req.count = buffer_count;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  Status request_buffers_status =
      SendCameraCommand(camera_handle, VIDIOC_REQBUFS, &req);
  if (!request_buffers_status.ok()) {
    LOG(ERROR) << "Request buffers failed with " << request_buffers_status;
    return request_buffers_status;
  }

  *buffers = (CameraBuffer*)(calloc(buffer_count, sizeof(*buffers)));
  for (int n_buffers = 0; n_buffers < buffer_count; ++n_buffers) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = n_buffers;
    Status query_buffer_status =
        SendCameraCommand(camera_handle, VIDIOC_QUERYBUF, &buf);
    if (!query_buffer_status.ok()) {
      LOG(ERROR) << "Query buffer failed with " << query_buffer_status;
      return query_buffer_status;
    }
    (*buffers)[n_buffers].length = buf.length;
    (*buffers)[n_buffers].start =
        v4l2_mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                  camera_handle, buf.m.offset);

    if (MAP_FAILED == (*buffers)[n_buffers].start) {
      LOG(ERROR) << "Memory-mapping buffer failed";
      return tensorflow::errors::Unknown("Memory-mapping buffer failed");
    }
  }

  for (int i = 0; i < buffer_count; ++i) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    Status set_buffer_status =
        SendCameraCommand(camera_handle, VIDIOC_QBUF, &buf);
    if (!set_buffer_status.ok()) {
      LOG(ERROR) << "Set buffer failed with " << set_buffer_status;
      return set_buffer_status;
    }
  }

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  Status stream_on_status =
      SendCameraCommand(camera_handle, VIDIOC_STREAMON, &type);
  if (!stream_on_status.ok()) {
    LOG(ERROR) << "Turning stream on failed with " << stream_on_status;
    return stream_on_status;
  }
  return Status::OK();
}

Status EndCameraCapture(int camera_handle, CameraBuffer* buffers,
                        int buffer_count) {
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  Status stream_off_status =
      SendCameraCommand(camera_handle, VIDIOC_STREAMOFF, &type);
  if (!stream_off_status.ok()) {
    LOG(ERROR) << "Turning stream off failed with " << stream_off_status;
    return stream_off_status;
  }
  for (int i = 0; i < buffer_count; ++i)
    v4l2_munmap(buffers[i].start, buffers[i].length);
  return Status::OK();
}

Status CaptureNextFrame(int camera_handle, CameraBuffer* buffers,
                        uint8_t** frame_data, int* frame_data_size,
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
    return tensorflow::errors::Unknown(
        "CaptureCameraFrame: select() failed with", errno);
  }

  memset(buf, 0, sizeof(*buf));
  buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf->memory = V4L2_MEMORY_MMAP;
  Status get_buffer_status =
      SendCameraCommand(camera_handle, VIDIOC_DQBUF, buf);
  if (!get_buffer_status.ok()) {
    LOG(ERROR) << "Get buffer failed with " << get_buffer_status;
    return get_buffer_status;
  }

  *frame_data = static_cast<uint8_t*>(buffers[buf->index].start);
  *frame_data_size = buf->bytesused;

  return Status::OK();
}

Status ReleaseFrame(int camera_handle, v4l2_buffer* buf) {
  Status release_buffer_status =
      SendCameraCommand(camera_handle, VIDIOC_QBUF, buf);
  if (!release_buffer_status.ok()) {
    LOG(ERROR) << "Release buffer failed with " << release_buffer_status;
    return release_buffer_status;
  }
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* out_indices, Tensor* out_scores) {
  const Tensor& unsorted_scores_tensor = outputs[0];
  auto unsorted_scores_flat = unsorted_scores_tensor.flat<float>();
  std::vector<std::pair<int, float>> scores;
  for (int i = 0; i < unsorted_scores_flat.size(); ++i) {
    scores.push_back(std::pair<int, float>({i, unsorted_scores_flat(i)}));
  }
  std::sort(scores.begin(), scores.end(),
            [](const std::pair<int, float>& left,
               const std::pair<int, float>& right) {
              return left.second > right.second;
            });
  scores.resize(how_many_labels);
  Tensor sorted_indices(tensorflow::DT_INT32, {scores.size()});
  Tensor sorted_scores(tensorflow::DT_FLOAT, {scores.size()});
  for (int i = 0; i < scores.size(); ++i) {
    sorted_indices.flat<int>()(i) = scores[i].first;
    sorted_scores.flat<float>()(i) = scores[i].second;
  }
  *out_indices = sorted_indices;
  *out_scores = sorted_scores;
  return Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(string file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
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
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const std::vector<string>& labels, int label_count,
                      float print_threshold) {
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
    // Print the top label to stdout if it's above a threshold.
    if ((pos == 0) && (score > print_threshold)) {
      std::cout << labels[label_index] << std::endl;
    }
  }
  return Status::OK();
}

// Given an image buffer, resize it to the requested size, and then scale the
// values as desired.
Status TensorFromFrame(uint8_t* image_data, int image_width, int image_height,
                       int image_channels, const int wanted_height,
                       const int wanted_width, const float input_mean,
                       const float input_std,
                       std::vector<Tensor>* out_tensors) {
  const int wanted_channels = 3;
  if (image_channels < wanted_channels) {
    return tensorflow::errors::FailedPrecondition(
        "Image needs to have at least ", wanted_channels, " but only has ",
        image_channels);
  }
  // In these loops, we convert the eight-bit data in the image into float,
  // resize it using bilinear filtering, and scale it numerically to the float
  // range that the model expects (given by input_mean and input_std).
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape(
          {1, wanted_height, wanted_width, wanted_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8* in = image_data;
  float* out = image_tensor_mapped.data();
  const size_t image_rowlen = image_width * image_channels;
  const float width_scale = static_cast<float>(image_width) / wanted_width;
  const float height_scale = static_cast<float>(image_height) / wanted_height;
  for (int y = 0; y < wanted_height; ++y) {
    const float in_y = y * height_scale;
    const int top_y_index = static_cast<int>(floorf(in_y));
    const int bottom_y_index =
        std::min(static_cast<int>(ceilf(in_y)), (image_height - 1));
    const float y_lerp = in_y - top_y_index;
    tensorflow::uint8* in_top_row = in + (top_y_index * image_rowlen);
    tensorflow::uint8* in_bottom_row = in + (bottom_y_index * image_rowlen);
    float* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const float in_x = x * width_scale;
      const int left_x_index = static_cast<int>(floorf(in_x));
      const int right_x_index =
          std::min(static_cast<int>(ceilf(in_x)), (image_width - 1));
      tensorflow::uint8* in_top_left_pixel =
          in_top_row + (left_x_index * wanted_channels);
      tensorflow::uint8* in_top_right_pixel =
          in_top_row + (right_x_index * wanted_channels);
      tensorflow::uint8* in_bottom_left_pixel =
          in_bottom_row + (left_x_index * wanted_channels);
      tensorflow::uint8* in_bottom_right_pixel =
          in_bottom_row + (right_x_index * wanted_channels);
      const float x_lerp = in_x - left_x_index;
      float* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        const float top_left((in_top_left_pixel[c] - input_mean) / input_std);
        const float top_right((in_top_right_pixel[c] - input_mean) / input_std);
        const float bottom_left((in_bottom_left_pixel[c] - input_mean) /
                                input_std);
        const float bottom_right((in_bottom_right_pixel[c] - input_mean) /
                                 input_std);
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom =
            bottom_left + (bottom_right - bottom_left) * x_lerp;
        out_pixel[c] = top + (bottom - top) * y_lerp;
      }
    }
  }

  out_tensors->push_back(image_tensor);
  return Status::OK();
}

int main(int argc, char** argv) {
  string graph =
      "tensorflow/contrib/pi_examples/label_image/data/"
      "tensorflow_inception_stripped.pb";
  string labels_file_name =
      "tensorflow/contrib/pi_examples/label_image/data/"
      "imagenet_comp_graph_label_strings.txt";
  int32 input_width = 299;
  int32 input_height = 299;
  int32 input_mean = 128;
  int32 input_std = 128;
  string input_layer = "Mul";
  string output_layer = "softmax";
  int32 video_width = 640;
  int32 video_height = 480;
  int print_threshold = 50;
  string root_dir = "";
  const bool parse_result = tensorflow::ParseFlags(
      &argc, argv, {Flag("graph", &graph),                      //
                    Flag("labels", &labels_file_name),          //
                    Flag("input_width", &input_width),          //
                    Flag("input_height", &input_height),        //
                    Flag("input_mean", &input_mean),            //
                    Flag("input_std", &input_std),              //
                    Flag("input_layer", &input_layer),          //
                    Flag("output_layer", &output_layer),        //
                    Flag("video_width", &video_width),          //
                    Flag("video_height", &video_height),        //
                    Flag("print_threshold", &print_threshold),  //
                    Flag("root_dir", &root_dir)});
  if (!parse_result) {
    LOG(ERROR) << "Error parsing command-line flags.";
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return -1;
  }

  int camera_handle;
  Status open_status = OpenCamera(&camera_handle);
  if (!open_status.ok()) {
    LOG(ERROR) << "OpenCamera failed with " << open_status;
    return -1;
  }

  Status format_status =
      SetCameraFormat(camera_handle, video_width, video_height);
  if (!format_status.ok()) {
    LOG(ERROR) << "SetCameraFormat failed with " << format_status;
    return -1;
  }

  const int how_many_buffers = 2;
  CameraBuffer* buffers;
  Status start_capture_status =
      StartCameraCapture(camera_handle, how_many_buffers, &buffers);
  if (!start_capture_status.ok()) {
    LOG(ERROR) << "StartCameraCapture failed with " << start_capture_status;
    return -1;
  }

  for (int i = 0; i < 200; i++) {
    uint8_t* frame_data;
    int frame_data_size;
    v4l2_buffer buf;
    Status capture_next_status = CaptureNextFrame(
        camera_handle, buffers, &frame_data, &frame_data_size, &buf);
    if (!capture_next_status.ok()) {
      LOG(ERROR) << "CaptureNextFrame failed with " << capture_next_status;
      return -1;
    }

    std::vector<Tensor> resized_tensors;
    Status tensor_from_frame_status =
        TensorFromFrame(frame_data, video_width, video_height, 3, input_height,
                        input_width, input_mean, input_std, &resized_tensors);
    if (!tensor_from_frame_status.ok()) {
      LOG(ERROR) << tensor_from_frame_status;
      return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    Status release_frame_status = ReleaseFrame(camera_handle, &buf);
    if (!release_frame_status.ok()) {
      LOG(ERROR) << "ReleaseFrame failed with " << release_frame_status;
      return -1;
    }

    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
                                     {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }

    // Do something interesting with the results we've generated.
    Status print_status =
        PrintTopLabels(outputs, labels, label_count, print_threshold * 0.01f);
    if (!print_status.ok()) {
      LOG(ERROR) << "Running print failed: " << print_status;
      return -1;
    }
  }

  Status end_capture_status =
      EndCameraCapture(camera_handle, buffers, how_many_buffers);
  if (!end_capture_status.ok()) {
    LOG(ERROR) << "EndCameraCapture failed with " << end_capture_status;
    return -1;
  }

  Status close_status = CloseCamera(camera_handle);
  if (!close_status.ok()) {
    LOG(ERROR) << "CloseCamera failed with " << open_status;
    return -1;
  }

  return 0;
}
