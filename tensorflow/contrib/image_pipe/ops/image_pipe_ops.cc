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


#if GOOGLE_CUDA

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>


namespace tensorflow {
namespace {

using namespace std;

class ImagePipeOpKernel: public AsyncOpKernel {
 public:

  explicit ImagePipeOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("directory_url", &directory_url));
    OP_REQUIRES_OK(c, c->GetAttr("image_format", &image_format));
    OP_REQUIRES_OK(c, c->GetAttr("batch_size", &batch_size));
    OP_REQUIRES_OK(c, c->GetAttr("height", &height));
    OP_REQUIRES_OK(c, c->GetAttr("width", &width));
    OP_REQUIRES_OK(c, c->GetAttr("parallel", &parallel));
    OP_REQUIRES_OK(c, c->GetAttr("seed", &seed));
    OP_REQUIRES_OK(c, c->GetAttr("rescale", &rescale));
    OP_REQUIRES_OK(c, c->GetAttr("synchronize", &synchronize));
    OP_REQUIRES_OK(c, c->GetAttr("cache_size", &cache_size));
    OP_REQUIRES_OK(c, c->GetAttr("logging", &logging));

    if (directory_url.size() > 0 && directory_url[directory_url.size() - 1] != '/')
      directory_url += '/';

    CHECK_EQ(image_format == "NCHW" || image_format == "NHWC", true);

    threadStop = false;
    samples = 0, iter = 0;

    dirent *ep, *ch_ep;
    DIR *root = opendir(directory_url.c_str());
    if (root != nullptr) {
      while ((ep = readdir(root)) != nullptr) {
        if (!ep->d_name[0] || !strcmp(ep->d_name, ".") || !strcmp(ep->d_name, ".."))
          continue;
        string sub_dir = directory_url + ep->d_name + "/";
        DIR *child = opendir(sub_dir.c_str());
        if (child == nullptr)
          continue;
        while ((ch_ep = readdir(child)) != nullptr) {
          if (!ch_ep->d_name[0] || !strcmp(ch_ep->d_name, ".") || !strcmp(ch_ep->d_name, ".."))
            continue;
          string file = ch_ep->d_name;
          size_t found = file.find_last_of('.') + 1;
          if (!found)
            continue;
          string ext = file.substr(found);
          transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
          if (ext != "jpg" && ext != "jpeg")
            continue;
          dict[sub_dir].push_back(file);
        }
        closedir(child);
      }
      closedir(root);

      for (auto &it: dict) {
        keyset.push_back(it.first);
        sort(keyset.begin(), keyset.end());
        samples += it.second.size();
      }
      n_class = keyset.size();

      if (logging) {
        LOG(INFO) << "Total images: " << samples <<", belonging to " << n_class << " classes, loaded from '" << directory_url << "';";
        for (int i = 0; i < n_class; ++i)
          LOG(INFO) << "  [*] class-id " << i << " => " << keyset[i] << " (" << dict[keyset[i]].size() << " samples included);";
      }
    }

    if (samples == 0) {
      LOG(FATAL) << "No valid images found in directory '" << directory_url << "'.";
    }

    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    int gpu_id = gpu_info->gpu_id;

    workers.resize(parallel);
    for (int i = 0; i < parallel; ++i) {
      workers[i].handle = new std::thread([this, i, gpu_id] {
        this->BackgroundWorker(i, gpu_id);
      });
    }
  }

  void BackgroundWorker(int idx, int gpu_id) {
    unsigned int local_seed = seed * parallel + idx;
    auto &worker = workers[idx];
    CHECK_EQ(cudaSuccess, cudaSetDevice(gpu_id));
    int depth = 3;
    while (1) {
      while (1) {
        worker.mu_.lock();
        if (worker.ord_que.size() >= cache_size) {
          worker.mu_.unlock();
          usleep(500000);
          if (threadStop)
            return;
          continue;
        }
        worker.mu_.unlock();
        break;
      }
      size_t image_size = (batch_size * depth * height * width) * sizeof(float), label_size = batch_size * sizeof(int);
      void *image_label_mem = nullptr;
      {
        mutex_lock l(worker.mu_);
        auto &it = worker.buffers;
        if (it.size()) {
          image_label_mem = it.back();
          it.pop_back();
        }
      }

      if (!image_label_mem)
        CHECK_EQ(cudaSuccess, cudaMallocHost(&image_label_mem, image_size + label_size));

      float *image_mem = (float*)image_label_mem;
      int *label_mem = (int*)(((char*)image_label_mem) + image_size);

      for (int i = 0; i < batch_size; ++i) {
        float *image_offset = image_mem + i * depth * height * width;
        int *label_offset = label_mem + i;

        while (1) {
          int label = rand_r(&local_seed) % dict.size();
          auto &files = dict[keyset[label]];
          if (files.size() == 0)
            continue;
          int it = rand_r(&local_seed) % files.size();
          string path = keyset[label] + files[it];

          int width_ = 0, height_ = 0, depths_ = 0;
          vector<uint8> output;
          {
            FILE *fp = fopen(path.c_str(), "rb");
            CHECK_EQ(!!fp, true);
            fseek(fp, 0, SEEK_END);
            size_t input_size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            vector<uint8> input(input_size);
            CHECK_EQ(input_size, fread(input.data(), 1, input_size, fp));
            fclose(fp);


            jpeg::UncompressFlags flags;
            jpeg::Uncompress(input.data(), input_size, flags, nullptr,
              [=, &output, &width_, &height_, &depths_](int width, int height, int depths) -> uint8* {
                 output.resize(width * height * depths);
                 width_ = width, height_ = height, depths_ = depths;
                 return output.data();
            });
            if (!output.size())
              continue;
          }
          CHECK_EQ(depths_ == 3 || depths_ == 1, true);

          uint8 *image_ptr = output.data();
          vector<int> stride;

          if (image_format == "NCHW")
            stride = {width, 1, width * height};
          else // image_format == "NHWC"
            stride = {width * depth, depth, 1};

          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              for (int d = 0; d < depth; ++d) {
                int ih = h * height_ / height, iw = w * width_ / width;
                *(image_offset + h * stride[0] + w * stride[1] + d * stride[2]) =
                    *(image_ptr + ih * width_ * depths_ + iw * depths_ + (depths_ == 3 ? d : 0)) * rescale;
              }
            }
          }
          *label_offset = label;
          break;
        }
      }

      mutex_lock l(worker.mu_);
      worker.ord_que.push(image_label_mem);
    }
  }

  ~ImagePipeOpKernel() {
    {
      threadStop = true;
      for (auto &worker: workers) {
        worker.handle->join();
        delete worker.handle;
      }

      while (recycleBufferAsync() > 0)
        ;

      for (auto &worker: workers) {
        while (worker.ord_que.size()) {
          worker.buffers.push_back(worker.ord_que.front());
          worker.ord_que.pop();
        }

        for (auto *buff: worker.buffers)
          CHECK_EQ(cudaSuccess, cudaFreeHost(buff));
      }
      workers.clear();
    }
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    void *image_label_mem = nullptr;
    int idx = (iter++) % workers.size();
    auto &worker = workers[idx];
    while (!image_label_mem) {
      mutex_lock l(worker.mu_);
      if (worker.ord_que.size() == 0)
        continue;
      image_label_mem = worker.ord_que.front();
      worker.ord_que.pop();
    }

    Tensor* image_t = nullptr, *label_t = nullptr;
    auto image_shape = (image_format == "NCHW") ? tensorflow::TensorShape({batch_size, 3, height, width}):
      tensorflow::TensorShape({batch_size, height, width, 3});
    auto label_shape = tensorflow::TensorShape({batch_size});
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, image_shape, &image_t), done);
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(1, label_shape, &label_t), done);

    se::Stream* tensor_stream = c->op_device_context()->stream();
    const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      tensor_stream->implementation()->GpuStreamMemberHack()); // CudaStreamMemberHack());

    size_t image_size = (batch_size * 3 * height * width) * sizeof(float);
    float *image_mem = (float*)image_label_mem;
    int *label_mem = (int*)(((char*)image_label_mem) + image_size);

    CHECK_EQ(cudaSuccess, cudaMemcpyAsync((void*)image_t->tensor_data().data(), image_mem, image_t->NumElements() * sizeof(float), cudaMemcpyHostToDevice, *cu_stream));
    CHECK_EQ(cudaSuccess, cudaMemcpyAsync((void*)label_t->tensor_data().data(), label_mem, label_t->NumElements() * sizeof(int), cudaMemcpyHostToDevice, *cu_stream));

    if (synchronize) {
      CHECK_EQ(cudaSuccess, cudaStreamSynchronize(*cu_stream));

      mutex_lock l(worker.mu_);
      worker.buffers.push_back(image_label_mem);
    } else {
      cudaEvent_t event;
      recycleBufferAsync();
      CHECK_EQ(cudaSuccess, cudaEventCreate(&event));
      CHECK_EQ(cudaSuccess, cudaEventRecord(event, *cu_stream));
      lazyRecycleBuffers.push_back({event, image_label_mem, &worker});
    }
    done();
  }

  size_t recycleBufferAsync() {
    for (int i = 0; i < lazyRecycleBuffers.size(); ++i) {
      auto res = cudaEventQuery((cudaEvent_t)lazyRecycleBuffers[i][0]);
      if (res == cudaSuccess) {
        CHECK_EQ(cudaSuccess, cudaEventDestroy((cudaEvent_t)lazyRecycleBuffers[i][0]));
        void *buff = lazyRecycleBuffers[i][1];
        Worker *pWorker = (Worker*)lazyRecycleBuffers[i][2];
        lazyRecycleBuffers[i] = lazyRecycleBuffers.back();
        lazyRecycleBuffers.pop_back();

        mutex_lock l(pWorker->mu_);
        pWorker->buffers.push_back(buff);
        continue;
      }
      CHECK_EQ(res, cudaErrorNotReady);
    }
    return lazyRecycleBuffers.size();
  }

 private:
  unordered_map<string, vector<string>> dict;
  vector<string> keyset;

  struct Worker {
    std::thread *handle;
    mutex mu_;
    queue<void*> ord_que;
    vector<void*> buffers;
  };

  vector<Worker> workers;

  string directory_url, image_format;
  int batch_size, height, width;

  int iter, n_class, samples;
  int cache_size, parallel, seed;
  float rescale;

  bool synchronize, logging;
  volatile bool threadStop;

  vector<vector<void*>> lazyRecycleBuffers;

  TF_DISALLOW_COPY_AND_ASSIGN(ImagePipeOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("ImagePipe").Device(DEVICE_GPU), ImagePipeOpKernel);

REGISTER_OP("ImagePipe")
    .Output("image: float")
    .Output("label: int32")
    .Attr("directory_url: string")
    .Attr("image_format: string")
    .Attr("batch_size: int")
    .Attr("height: int")
    .Attr("width: int")
    .Attr("parallel: int = 8")
    .Attr("cache_size: int = 4")
    .Attr("rescale: float = 0.00392156862")
    .Attr("seed: int = 0")
    .Attr("synchronize: bool = true")
    .Attr("logging: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
