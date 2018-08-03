/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/bigtable/kernels/bigtable_lib.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

namespace {

class BigtableClientOp : public OpKernel {
 public:
  explicit BigtableClientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("project_id", &project_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("instance_id", &instance_id_));
    OP_REQUIRES(ctx, !project_id_.empty(),
                errors::InvalidArgument("project_id must be non-empty"));
    OP_REQUIRES(ctx, !instance_id_.empty(),
                errors::InvalidArgument("instance_id must be non-empty"));

    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("connection_pool_size", &connection_pool_size_));
    // If left unset by the client code, set it to a default of 100. Note: the
    // cloud-cpp default of 4 concurrent connections is far too low for high
    // performance streaming.
    if (connection_pool_size_ == -1) {
      connection_pool_size_ = 100;
    }

    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_receive_message_size",
                                     &max_receive_message_size_));
    // If left unset by the client code, set it to a default of 100. Note: the
    // cloud-cpp default of 4 concurrent connections is far too low for high
    // performance streaming.
    if (max_receive_message_size_ == -1) {
      max_receive_message_size_ = 1 << 24;  // 16 MBytes
    }
    OP_REQUIRES(ctx, max_receive_message_size_ > 0,
                errors::InvalidArgument("connection_pool_size must be > 0"));
  }

  ~BigtableClientOp() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<BigtableClientResource>(cinfo_.container(),
                                                cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));
      BigtableClientResource* resource;
      OP_REQUIRES_OK(
          ctx,
          mgr->LookupOrCreate<BigtableClientResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [this, ctx](
                  BigtableClientResource** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                auto client_options =
                    google::cloud::bigtable::ClientOptions()
                        .set_connection_pool_size(connection_pool_size_)
                        .set_data_endpoint("batch-bigtable.googleapis.com");
                auto channel_args = client_options.channel_arguments();
                channel_args.SetMaxReceiveMessageSize(
                    max_receive_message_size_);
                channel_args.SetUserAgentPrefix("tensorflow");
                client_options.set_channel_arguments(channel_args);
                std::shared_ptr<google::cloud::bigtable::DataClient> client =
                    google::cloud::bigtable::CreateDefaultDataClient(
                        project_id_, instance_id_, std::move(client_options));
                *ret = new BigtableClientResource(project_id_, instance_id_,
                                                  std::move(client));
                return Status::OK();
              }));
      core::ScopedUnref resource_cleanup(resource);
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            MakeTypeIndex<BigtableClientResource>()));
  }

 private:
  string project_id_;
  string instance_id_;
  int64 connection_pool_size_;
  int32 max_receive_message_size_;

  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
};

REGISTER_KERNEL_BUILDER(Name("BigtableClient").Device(DEVICE_CPU),
                        BigtableClientOp);

class BigtableTableOp : public OpKernel {
 public:
  explicit BigtableTableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_name", &table_));
    OP_REQUIRES(ctx, !table_.empty(),
                errors::InvalidArgument("table_name must be non-empty"));
  }

  ~BigtableTableOp() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<BigtableTableResource>(cinfo_.container(),
                                               cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    if (!initialized_) {
      ResourceMgr* mgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, cinfo_.Init(mgr, def()));

      BigtableClientResource* client_resource;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client_resource));
      core::ScopedUnref unref_client(client_resource);

      BigtableTableResource* resource;
      OP_REQUIRES_OK(
          ctx, mgr->LookupOrCreate<BigtableTableResource>(
                   cinfo_.container(), cinfo_.name(), &resource,
                   [this, client_resource](BigtableTableResource** ret) {
                     *ret = new BigtableTableResource(client_resource, table_);
                     return Status::OK();
                   }));
      initialized_ = true;
    }
    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            MakeTypeIndex<BigtableTableResource>()));
  }

 private:
  string table_;  // Note: this is const after construction.

  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
};

REGISTER_KERNEL_BUILDER(Name("BigtableTable").Device(DEVICE_CPU),
                        BigtableTableOp);

class ToBigtableOp : public AsyncOpKernel {
 public:
  explicit ToBigtableOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("to_bigtable_op_", SanitizeThreadSuffix(name())),
            /* num_threads = */ 1, /* low_latency_hint = */ false)) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    thread_pool_->Schedule([this, ctx, done]() {
      const Tensor* column_families_tensor;
      OP_REQUIRES_OK_ASYNC(
          ctx, ctx->input("column_families", &column_families_tensor), done);
      OP_REQUIRES_ASYNC(
          ctx, column_families_tensor->dims() == 1,
          errors::InvalidArgument("`column_families` must be a vector."), done);

      const Tensor* columns_tensor;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("columns", &columns_tensor), done);
      OP_REQUIRES_ASYNC(ctx, columns_tensor->dims() == 1,
                        errors::InvalidArgument("`columns` must be a vector."),
                        done);
      OP_REQUIRES_ASYNC(
          ctx,
          columns_tensor->NumElements() ==
              column_families_tensor->NumElements(),
          errors::InvalidArgument("len(column_families) != len(columns)"),
          done);

      std::vector<string> column_families;
      column_families.reserve(column_families_tensor->NumElements());
      std::vector<string> columns;
      columns.reserve(column_families_tensor->NumElements());
      for (uint64 i = 0; i < column_families_tensor->NumElements(); ++i) {
        column_families.push_back(column_families_tensor->flat<string>()(i));
        columns.push_back(columns_tensor->flat<string>()(i));
      }

      DatasetBase* dataset;
      OP_REQUIRES_OK_ASYNC(
          ctx, GetDatasetFromVariantTensor(ctx->input(1), &dataset), done);

      IteratorContext iter_ctx = dataset::MakeIteratorContext(ctx);
      std::unique_ptr<IteratorBase> iterator;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          dataset->MakeIterator(&iter_ctx, "ToBigtableOpIterator", &iterator),
          done);

      int64 timestamp_int;
      OP_REQUIRES_OK_ASYNC(
          ctx, ParseScalarArgument<int64>(ctx, "timestamp", &timestamp_int),
          done);
      OP_REQUIRES_ASYNC(ctx, timestamp_int >= -1,
                        errors::InvalidArgument("timestamp must be >= -1"),
                        done);

      BigtableTableResource* resource;
      OP_REQUIRES_OK_ASYNC(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
      core::ScopedUnref resource_cleanup(resource);

      std::vector<Tensor> components;
      components.reserve(dataset->output_dtypes().size());
      bool end_of_sequence = false;
      do {
        ::google::cloud::bigtable::BulkMutation mutation;
        // TODO(saeta): Make # of mutations configurable.
        for (uint64 i = 0; i < 100 && !end_of_sequence; ++i) {
          OP_REQUIRES_OK_ASYNC(
              ctx, iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
              done);
          if (!end_of_sequence) {
            OP_REQUIRES_OK_ASYNC(
                ctx,
                CreateMutation(std::move(components), column_families, columns,
                               timestamp_int, &mutation),
                done);
          }
          components.clear();
        }
        grpc::Status mutation_status;
        std::vector<::google::cloud::bigtable::FailedMutation> failures =
            resource->table().BulkApply(std::move(mutation), mutation_status);
        if (!mutation_status.ok()) {
          LOG(ERROR) << "Failure applying mutation: "
                     << mutation_status.error_code() << " - "
                     << mutation_status.error_message() << " ("
                     << mutation_status.error_details() << ").";
        }
        if (!failures.empty()) {
          for (const auto& failure : failures) {
            LOG(ERROR) << "Failure applying mutation on row ("
                       << failure.original_index()
                       << "): " << failure.mutation().row_key()
                       << " - error: " << failure.status().error_message()
                       << " (Details: " << failure.status().error_details()
                       << ").";
          }
        }
        OP_REQUIRES_ASYNC(
            ctx, failures.empty() && mutation_status.ok(),
            errors::Unknown("Failure while writing to Cloud Bigtable: ",
                            mutation_status.error_code(), " - ",
                            mutation_status.error_message(), " (",
                            mutation_status.error_details(),
                            "), # of mutation failures: ", failures.size(),
                            ". See the log for the specific error details."),
            done);
      } while (!end_of_sequence);
      done();
    });
  }

 private:
  static string SanitizeThreadSuffix(string suffix) {
    string clean;
    for (int i = 0; i < suffix.size(); ++i) {
      const char ch = suffix[i];
      if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
        clean += ch;
      } else {
        clean += '_';
      }
    }
    return clean;
  }

  Status CreateMutation(
      std::vector<Tensor> tensors, const std::vector<string>& column_families,
      const std::vector<string>& columns, int64 timestamp_int,
      ::google::cloud::bigtable::BulkMutation* bulk_mutation) {
    if (tensors.size() != column_families.size() + 1) {
      return errors::InvalidArgument(
          "Iterator produced a set of Tensors shorter than expected");
    }
    ::google::cloud::bigtable::SingleRowMutation mutation(
        std::move(tensors[0].scalar<string>()()));
    std::chrono::milliseconds timestamp(timestamp_int);
    for (size_t i = 1; i < tensors.size(); ++i) {
      if (!TensorShapeUtils::IsScalar(tensors[i].shape())) {
        return errors::Internal("Output tensor ", i, " was not a scalar");
      }
      if (timestamp_int == -1) {
        mutation.emplace_back(::google::cloud::bigtable::SetCell(
            column_families[i - 1], columns[i - 1],
            std::move(tensors[i].scalar<string>()())));
      } else {
        mutation.emplace_back(::google::cloud::bigtable::SetCell(
            column_families[i - 1], columns[i - 1], timestamp,
            std::move(tensors[i].scalar<string>()())));
      }
    }
    bulk_mutation->emplace_back(std::move(mutation));
    return Status::OK();
  }

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece& argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }

  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

REGISTER_KERNEL_BUILDER(Name("DatasetToBigtable").Device(DEVICE_CPU),
                        ToBigtableOp);

}  // namespace

}  // namespace tensorflow
