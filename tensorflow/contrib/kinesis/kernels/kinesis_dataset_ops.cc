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

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/core/utils/Outcome.h>
#include <aws/kinesis/KinesisClient.h>
#include <aws/kinesis/model/DescribeStreamRequest.h>
#include <aws/kinesis/model/GetRecordsRequest.h>
#include <aws/kinesis/model/GetShardIteratorRequest.h>
#include <aws/kinesis/model/PutRecordsRequest.h>
#include <aws/kinesis/model/ShardIteratorType.h>
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/s3/s3_crypto.h"

namespace tensorflow {
namespace {

Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  static mutex mu(LINKER_INITIALIZED);
  static bool init(false);
  static Aws::Client::ClientConfiguration config;

  std::lock_guard<mutex> lock(mu);

  if (!init) {
    const char* endpoint = getenv("KINESIS_ENDPOINT");
    if (endpoint) {
      config.endpointOverride = Aws::String(endpoint);
    }
    const char* region = getenv("AWS_REGION");
    if (region) {
      config.region = Aws::String(region);
    } else {
      // Load config file (e.g., ~/.aws/config) only if AWS_SDK_LOAD_CONFIG
      // is set with a truthy value.
      const char* load_config_env = getenv("AWS_SDK_LOAD_CONFIG");
      string load_config =
          load_config_env ? str_util::Lowercase(load_config_env) : "";
      if (load_config == "true" || load_config == "1") {
        Aws::String config_file;
        // If AWS_CONFIG_FILE is set then use it, otherwise use ~/.aws/config.
        const char* config_file_env = getenv("AWS_CONFIG_FILE");
        if (config_file_env) {
          config_file = config_file_env;
        } else {
          const char* home_env = getenv("HOME");
          if (home_env) {
            config_file = home_env;
            config_file += "/.aws/config";
          }
        }
        Aws::Config::AWSConfigFileProfileConfigLoader loader(config_file);
        loader.Load();
        auto profiles = loader.GetProfiles();
        if (!profiles["default"].GetRegion().empty()) {
          config.region = profiles["default"].GetRegion();
        }
      }
    }
    const char* use_https = getenv("KINESIS_USE_HTTPS");
    if (use_https) {
      if (use_https[0] == '0') {
        config.scheme = Aws::Http::Scheme::HTTP;
      } else {
        config.scheme = Aws::Http::Scheme::HTTPS;
      }
    }
    const char* verify_ssl = getenv("KINESIS_VERIFY_SSL");
    if (verify_ssl) {
      if (verify_ssl[0] == '0') {
        config.verifySSL = false;
      } else {
        config.verifySSL = true;
      }
    }
    const char* connect_timeout = getenv("KINESIS_CONNECT_TIMEOUT_MSEC");
    if (connect_timeout) {
      int64 timeout;

      if (strings::safe_strto64(connect_timeout, &timeout)) {
        config.connectTimeoutMs = timeout;
      }
    }
    const char* request_timeout = getenv("KINESIS_REQUEST_TIMEOUT_MSEC");
    if (request_timeout) {
      int64 timeout;

      if (strings::safe_strto64(request_timeout, &timeout)) {
        config.requestTimeoutMs = timeout;
      }
    }

    init = true;
  }

  return config;
};

static mutex mu(LINKER_INITIALIZED);
static unsigned count(0);
void AwsInitAPI() {
  std::lock_guard<mutex> lock(mu);
  count++;
  if (count == 1) {
    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<S3SHA256Factory>(S3CryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<S3SHA256HmacFactory>(S3CryptoAllocationTag);
    };
    Aws::InitAPI(options);
  }
}
void AwsShutdownAPI() {
  std::lock_guard<mutex> lock(mu);
  count--;
  if (count == 0) {
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
  }
}
void ShutdownClient(Aws::Kinesis::KinesisClient* client) {
  if (client != nullptr) {
    delete client;
    AwsShutdownAPI();
  }
}
}
class KinesisDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    std::string stream = "";
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<std::string>(ctx, "stream", &stream));
    std::string shard = "";
    OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "shard", &shard));
    bool eof = false;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "eof", &eof));
    int64 interval = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "interval", &interval));
    OP_REQUIRES(ctx, (interval > 0),
                errors::InvalidArgument(
                    "Interval value should be large than 0, got ", interval));
    *output = new Dataset(ctx, stream, shard, eof, interval);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const string& stream, const string& shard,
            const bool eof, const int64 interval)
        : GraphDatasetBase(ctx),
          stream_(stream),
          shard_(shard),
          eof_(eof),
          interval_(interval) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Kinesis")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "KinesisDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* stream = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(stream_, &stream));
      Node* shard = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(shard_, &shard));
      Node* eof = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(eof_, &eof));
      Node* interval = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(interval_, &interval));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {stream, shard, eof, interval}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            client_(nullptr, ShutdownClient) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (iterator_ == "") {
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
        do {
          Aws::Kinesis::Model::GetRecordsRequest request;
          auto outcome = client_->GetRecords(
              request.WithShardIterator(iterator_).WithLimit(1));
          if (!outcome.IsSuccess()) {
            string error =
                strings::StrCat(outcome.GetError().GetExceptionName().c_str(),
                                ": ", outcome.GetError().GetMessage().c_str());
            return errors::Internal(error);
          }
          if (outcome.GetResult().GetRecords().size() == 0) {
            // If return 0 record then nothing available at the moment.
            if (dataset()->eof_) {
              *end_of_sequence = true;
              return Status::OK();
            }
            // Continue the loop after a period of time.
            ctx->env()->SleepForMicroseconds(dataset()->interval_);
            continue;
          }
          if (outcome.GetResult().GetRecords().size() != 1) {
            return errors::Internal("invalid records number ",
                                    outcome.GetResult().GetRecords().size(),
                                    " returned");
          }

          iterator_ = outcome.GetResult().GetNextShardIterator();

          StringPiece value(
              (const char*)outcome.GetResult()
                  .GetRecords()[0]
                  .GetData()
                  .GetUnderlyingData(),
              outcome.GetResult().GetRecords()[0].GetData().GetLength());
          Tensor value_tensor(ctx->allocator({}), DT_STRING, {});
          value_tensor.scalar<std::string>()() = std::string(value);
          out_tensors->emplace_back(std::move(value_tensor));

          *end_of_sequence = false;
          return Status::OK();
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      // Sets up Kinesis streams to read from.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        AwsInitAPI();
        client_.reset(
            new Aws::Kinesis::KinesisClient(GetDefaultClientConfig()));

        Aws::Kinesis::Model::DescribeStreamRequest request;
        auto outcome = client_->DescribeStream(
            request.WithStreamName(dataset()->stream_.c_str()));
        if (!outcome.IsSuccess()) {
          string error =
              strings::StrCat(outcome.GetError().GetExceptionName().c_str(),
                              ": ", outcome.GetError().GetMessage().c_str());
          return errors::Internal(error);
        }
        Aws::String shard;
        Aws::String sequence;
        if (dataset()->shard_ == "") {
          if (outcome.GetResult().GetStreamDescription().GetShards().size() !=
              1) {
            return errors::InvalidArgument(
                "shard has to be provided unless the stream only have one "
                "shard, there are ",
                outcome.GetResult().GetStreamDescription().GetShards().size(),
                " shards in stream ", dataset()->stream_);
          }
          shard = outcome.GetResult()
                      .GetStreamDescription()
                      .GetShards()[0]
                      .GetShardId();
          sequence = outcome.GetResult()
                         .GetStreamDescription()
                         .GetShards()[0]
                         .GetSequenceNumberRange()
                         .GetStartingSequenceNumber();
        } else {
          for (auto entry :
               outcome.GetResult().GetStreamDescription().GetShards()) {
            if (entry.GetShardId() == dataset()->shard_.c_str()) {
              shard = entry.GetShardId();
              sequence =
                  entry.GetSequenceNumberRange().GetStartingSequenceNumber();
              break;
            }
          }
          if (shard == "") {
            return errors::InvalidArgument("no shard ",
                                           dataset()->shard_.c_str(),
                                           " in stream ", dataset()->stream_);
          }
        }

        Aws::Kinesis::Model::GetShardIteratorRequest iterator_request;
        auto iterator_outcome = client_->GetShardIterator(
            iterator_request.WithStreamName(dataset()->stream_.c_str())
                .WithShardId(shard)
                .WithShardIteratorType(
                    Aws::Kinesis::Model::ShardIteratorType::AT_SEQUENCE_NUMBER)
                .WithStartingSequenceNumber(sequence));
        if (!iterator_outcome.IsSuccess()) {
          string error = strings::StrCat(
              iterator_outcome.GetError().GetExceptionName().c_str(), ": ",
              iterator_outcome.GetError().GetMessage().c_str());
          return errors::Internal(error);
        }
        iterator_ = iterator_outcome.GetResult().GetShardIterator();
        return Status::OK();
      }

      // Resets all Kinesis streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        iterator_ = "";
      }

      mutex mu_;
      Aws::String iterator_ GUARDED_BY(mu_);
      std::unique_ptr<Aws::Kinesis::KinesisClient, decltype(&ShutdownClient)>
          client_ GUARDED_BY(mu_);
    };

    const std::string stream_;
    const std::string shard_;
    const bool eof_;
    const int64 interval_;
  };
};

REGISTER_KERNEL_BUILDER(Name("KinesisDataset").Device(DEVICE_CPU),
                        KinesisDatasetOp);

}  // namespace tensorflow
