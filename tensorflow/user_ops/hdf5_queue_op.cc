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

// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>
#include <iostream>
#include <thread>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include <hdf5.h>
#include <assert.h>

#ifndef STREAM_FILE
#define STREAM_FILE "/data/data.hdf5"
#endif

namespace tensorflow {

  class Stream: public FIFOQueue {
  public:
    Stream(int capacity, const string &stream_id_,
	   const std::vector<string> &substream_names,
	   const DataTypeVector& component_dtypes,
	   const std::vector<TensorShape>& component_shapes,
	   const string& name) :
      FIFOQueue(capacity, component_dtypes, component_shapes, name),
      back_queue(new FIFOQueue(capacity, component_dtypes, component_shapes, name)) {
      // Open stream
      file = H5Fopen(STREAM_FILE, H5F_ACC_RDWR, H5P_DEFAULT);
      
      // Check if stream exists
      status = H5Eset_auto2(0, NULL, NULL);
      status = H5Gget_objinfo(file, stream_id_.c_str(), 0, NULL);
      if (status == 0) { // Stream group exists
	stream = H5Gopen(file, stream_id_.c_str(), H5P_DEFAULT);
      } else {
	stream = H5Gcreate(file, stream_id_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }

      // Create HDF5 subgroups
      for (int i=0;i<num_components();i++) {
	if (H5Lexists(stream, substream_names[i].c_str(), H5P_DEFAULT) > 0) {
	  substreams.push_back(H5Dopen2(stream, substream_names[i].c_str(), H5P_DEFAULT));
	} else {
	  const TensorShape &s = component_shapes[i];
	  std::vector<hsize_t> dims(s.dims());
	  for (int i=0;i<s.dims();i++)
	    dims[i] = s.dim_size(i);
	  dims.insert(dims.begin(), 1);
	  std::vector<hsize_t> maxdims(dims), chunkdims(dims);
	  maxdims[0] = H5S_UNLIMITED;
	  chunkdims[0] = 5; // Design choice

	  hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
	  H5Pset_chunk(prop, s.dims()+1, chunkdims.data());
	  hid_t dspace = H5Screate_simple(s.dims()+1, dims.data(), maxdims.data());
	  substreams.push_back(H5Dcreate2(stream, substream_names[i].c_str(), 
					  get_type(component_dtypes[i]), dspace,
					  H5P_DEFAULT, prop, H5P_DEFAULT));
	}
      }
      
      current_row=0;
    }
    
    Status Initialize() override {
      Status s0, s1;
      s0 = FIFOQueue::Initialize();
      s1 = back_queue->Initialize();

      if (!s0.ok())
	return s0;      
      if (!s1.ok())
	return s1;

      return Status::OK();
    }

    Status MatchesNodeDef(const NodeDef& node_def) override {
      if (!MatchesNodeDefOp(node_def, "Stream").ok()) {
	return errors::InvalidArgument("Expected Stream, found ", node_def.op());
      }
      TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
      TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
      TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
      return Status::OK();
    }

    void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
		    DoneCallback callback) override {
      bool done=false;
      std::thread t ([this, &done, &ctx] () {dequeuer(done, ctx);});
      back_queue->TryEnqueue(tuple, ctx, 
			     [callback, &done]() {
			       done=true;
			       callback();
			     });
      t.join();
    }

    void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
			DoneCallback callback) override {
      bool done=false;
      std::thread t ([this, &done, &ctx] () {dequeuer(done, ctx);});
      back_queue->TryEnqueueMany(tuple, ctx, 
				 [callback, &done]() {
				   done=true;
				   callback();
				 });
      t.join();
    }

    void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override {
      bool done=false;
      std::thread t ([this, &done, &ctx] () {enqueuer(done, ctx);});
      FIFOQueue::TryDequeue(ctx, 
			    [callback, &done](const QueueInterface::Tuple& tuple) {
			      done=true;
			      callback(tuple);
			    });
      t.join();
    }

    void TryDequeueMany(int num_elements, OpKernelContext* ctx,
			bool allow_small_batch,
			CallbackWithTuple callback) override {
      bool done=false;
      std::thread t ([this, &done, &ctx] () {enqueuer(done, ctx);});
      FIFOQueue::TryDequeueMany(num_elements, ctx, allow_small_batch,
				[callback, &done](const QueueInterface::Tuple& tuple) {
				  done=true;
				  callback(tuple);
				});
      t.join();      
    }

  private:
    void dequeuer(bool &done, OpKernelContext *ctx) {
      Notification n;
      while (!done) {
	back_queue->TryDequeue(ctx, 
	   [this, ctx, &n](const QueueInterface::Tuple& tuple) {
	       if (ctx->status().ok()) {
		 for (int i=0;i<substreams.size();i++) {
		   PersistentTensor pt = PersistentTensor(tuple[i]);
		   Tensor &t = *pt.AccessTensor(ctx);
		   // Resize for append
		   hid_t dataspace = H5Dget_space(substreams[i]);

		   std::vector<hsize_t> offset(t.dims()+1, 0), stride(t.dims()+1, 1), 
		     count(t.dims()), dims(t.dims()+1, 0);
		   for (int j=0;j<t.dims();j++)
		     count[j] = t.shape().dim_size(j);

		   H5Sget_simple_extent_dims(dataspace, dims.data(), NULL);
		   dims[0]++;
		   H5Dset_extent(substreams[i], dims.data());
		   
		   // Get Hyperslab
		   offset[0] = dims[0];
		   count.insert(count.begin(), 1); // 1 row
		   dataspace = H5Dget_space(substreams[i]);
		   assert(H5Sget_simple_extent_ndims(dataspace) == dims.size());

		   status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset.data(),
						stride.data(), count.data(), NULL);
		   
		   // Write to Hyperslab
		   H5Dwrite(substreams[i], get_type(component_dtypes_[i]), 
			    H5S_ALL, H5S_ALL, H5P_DEFAULT, const_cast<void*>(DMAHelper::base(&t)));
		 }
	       }
	       n.Notify();
	   }
	);
	n.WaitForNotification();
      }
    }

    void enqueuer(bool &done, OpKernelContext *ctx) {
      Notification n;

      while (!done) {
	Tuple tuple;
	tuple.reserve(num_components());

	assert(num_components() == substreams.size());
	// Pull from HDF5
	for (int i=0;i<substreams.size();i++) {
	  Tensor t;
	  TensorShape s = component_shapes_[i];
	  ctx->allocate_temp(component_dtypes_[i], s, &t);
	  hid_t dataspace = H5Dget_space(substreams[i]);
	  std::vector<hsize_t> offset(t.dims()+1, 0), stride(t.dims()+1, 1), 
	    count(t.shape().dim_sizes().data(), 
		  t.shape().dim_sizes().data() + t.dims()), dims(t.dims()+1, 0);
	  offset[0] = current_row++;
	  count.insert(count.begin(), 1); // 1 row
	  
	  status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset.data(),
				       stride.data(), count.data(), NULL);
	  H5Dread(substreams[i], get_type(component_dtypes_[i]), 
		  H5S_ALL, H5S_ALL, H5P_DEFAULT, const_cast<void*>(DMAHelper::base(&t)));
	  tuple.emplace_back(t);
	}

	FIFOQueue::TryEnqueue(tuple, ctx, [&n]() {n.Notify();});
	n.WaitForNotification();
      }
    }
    
    hid_t get_type(DataType t) {
      switch (t) {
      case DT_FLOAT:      return H5T_IEEE_F32LE;
      case DT_DOUBLE:     return H5T_IEEE_F64LE;
      case DT_INT8:       return H5T_STD_I8LE;
      case DT_INT16:      return H5T_STD_I16LE;
      case DT_INT32:      return H5T_STD_I32LE;
      case DT_INT64:      return H5T_STD_I64LE;
      case DT_UINT8:      return H5T_STD_U8LE;
      case DT_UINT16:     return H5T_STD_U16LE;
	//case DT_STRING:     return H5T_C_STRING; // TODO: Add String support. This doesn't build
      case DT_BOOL:       return H5T_NATIVE_HBOOL;
      case DT_COMPLEX64:  return H5T_IEEE_F64LE; // TODO: Fix
      case DT_COMPLEX128: return H5T_IEEE_F64LE; // TODO: Fix
      case DT_QINT8:      return H5T_STD_I8LE; // TODO: Figure these out
      case DT_QINT32:     return H5T_STD_I32LE;
      case DT_QUINT8:     return H5T_STD_U8LE;
      }
      return H5T_IEEE_F32LE;
    }

    hid_t file, stream, gcpl;
    hsize_t current_row;
    herr_t status;
    string stream_id_;
    std::vector<hid_t> substreams;

    FIFOQueue *back_queue;
  };

  // Defines a StreamOp, which produces a Queue (specifically, one
  // backed by Stream) that persists across different graph
  // executions, and sessions. Running this op produces a single-element
  // tensor of handles to Queues in the corresponding device.
  class StreamOp : public ResourceOpKernel<QueueInterface> {
  public:
    explicit StreamOp(OpKernelConstruction* context) : 
      ResourceOpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
      if (capacity_ < 0) {
	capacity_ = QueueBase::kUnbounded;
      }
      context->GetAttr("stream_id", &stream_id_);
      context->GetAttr("stream_columns", &stream_columns_);
      context->GetAttr("component_types", &component_types_);
      context->GetAttr("shapes", &component_shapes_);
    }

  private:
    Status CreateResource(QueueInterface** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      Stream* queue = new Stream(capacity_, stream_id_, stream_columns_, component_types_,
				 component_shapes_, cinfo_.name());
      
      *ret = queue;
      return queue->Initialize();
    }
  
    Status VerifyResource(QueueInterface* queue) override {
      return queue->MatchesNodeDef(def());
    }

    string stream_id_;
    std::vector<TensorShape> component_shapes_;
    std::vector<string> stream_columns_;
    int32 capacity_;
    DataTypeVector component_types_;

    TF_DISALLOW_COPY_AND_ASSIGN(StreamOp);
  };

  REGISTER_KERNEL_BUILDER(Name("Stream").Device(DEVICE_CPU), StreamOp);

  using shape_inference::DimensionHandle;
  using shape_inference::InferenceContext;
  using shape_inference::ShapeHandle;

  namespace {
    Status TwoElementOutput(InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    }
  }  // namespace

  REGISTER_OP("Stream")
  .Output("handle: resource")
  .Attr("stream_id: string")
  .Attr("stream_columns: list(string)")
  .Attr("component_types: list(type) >= 0 = []")
  .Attr("shapes: list(shape) >= 0 = []")
  .Attr("shared_name: string = ''")
  .Attr("container: string = ''")
  .Attr("capacity: int = -1")
  .SetIsStateful()
  .SetShapeFn(TwoElementOutput);

}  // namespace tensorflow
