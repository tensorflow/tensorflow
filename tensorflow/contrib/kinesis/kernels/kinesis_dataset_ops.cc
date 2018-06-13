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
#include "tensorflow/core/platform/s3/aws_crypto.h"

namespace tensorflow {
namespace {

Aws::Client::ClientConfiguration* InitializeDefaultClientConfig() {
  static Aws::Client::ClientConfiguration config;
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
      // Load the configuration. If successful, get the region.
      // If the load is not successful, then generate a warning.
      if (loader.Load()) {
        auto profiles = loader.GetProfiles();
        if (!profiles["default"].GetRegion().empty()) {
          config.region = profiles["default"].GetRegion();
        }
      } else {
        LOG(WARNING) << "Failed to load the profile in " << config_file << ".";
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

  return &config;
}

Aws::Client::ClientConfiguration& GetDefaultClientConfig() {
  static Aws::Client::ClientConfiguration* config =
      InitializeDefaultClientConfig();
  return *config;
}

static mutex mu(LINKER_INITIALIZED);
static unsigned count(0);
void AwsInitAPI() {
  mutex_lock lock(mu);
  count++;
  if (count == 1) {
    Aws::SDKOptions options;
    options.cryptoOptions.sha256Factory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256Factory>(AWSCryptoAllocationTag);
    };
    options.cryptoOptions.sha256HMACFactory_create_fn = []() {
      return Aws::MakeShared<AWSSHA256HmacFactory>(AWSCryptoAllocationTag);
    };
    Aws::InitAPI(options);
  }
}
void AwsShutdownAPI() {
  mutex_lock lock(mu);
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
    bool read_indefinitely = true;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "read_indefinitely",
                                                  &read_indefinitely));
    int64 interval = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "interval", &interval));
    OP_REQUIRES(ctx, (interval > 0),
                errors::InvalidArgument(
                    "Interval value should be large than 0, got ", interval));
    *output = new Dataset(ctx, stream, shard, read_indefinitely, interval);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const string& stream, const string& shard,
            const bool read_indefinitely, const int64 interval)
        : GraphDatasetBase(ctx),
          stream_(stream),
          shard_(shard),
          read_indefinitely_(read_indefinitely),
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
      Node* read_indefinitely = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(read_indefinitely_, &read_indefinitely));
      Node* interval = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(interval_, &interval));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {stream, shard, read_indefinitely, interval}, output));
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
          TF_RETURN_IF_ERROR(SetupStreamsLocked());
        }
        do {
          Aws::Kinesis::Model::GetRecordsRequest request;
          auto outcome = client_->GetRecords(
              request.WithShardIterator(iterator_).WithLimit(1));
          if (!outcome.IsSuccess()) {
            return errors::Unknown(outcome.GetError().GetExceptionName(), ": ",
                                   outcome.GetError().GetMessage());
          }
          if (outcome.GetResult().GetRecords().size() == 0) {
            // If no records were returned then nothing is available at the
            // moment.
            if (!dataset()->read_indefinitely_) {
              *end_of_sequence = true;
              return Status::OK();
            }
            // Continue the loop after a period of time.
            ctx->env()->SleepForMicroseconds(dataset()->interval_);
            continue;
          }
          if (outcome.GetResult().GetRecords().size() != 1) {
            return errors::Unknown("invalid number of records ",
                                   outcome.GetResult().GetRecords().size(),
                                   " returned");
          }

          iterator_ = outcome.GetResult().GetNextShardIterator();

          const auto& data = outcome.GetResult().GetRecords()[0].GetData();
          StringPiece value(
              reinterpret_cast<const char*>(data.GetUnderlyingData()),
              data.GetLength());
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
      Status SetupStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        AwsInitAPI();
        client_.reset(
            new Aws::Kinesis::KinesisClient(GetDefaultClientConfig()));

        Aws::Kinesis::Model::DescribeStreamRequest request;
        auto outcome = client_->DescribeStream(
            request.WithStreamName(dataset()->stream_.c_str()));
        if (!outcome.IsSuccess()) {
          return errors::Unknown(outcome.GetError().GetExceptionName(), ": ",
                                 outcome.GetError().GetMessage());
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
          for (const auto& entry :
               outcome.GetResult().GetStreamDescription().GetShards()) {
            if (entry.GetShardId() == dataset()->shard_.c_str()) {
              shard = entry.GetShardId();
              sequence =
                  entry.GetSequenceNumberRange().GetStartingSequenceNumber();
              break;
            }
          }
          if (shard == "") {
            return errors::InvalidArgument("no shard ", dataset()->shard_,
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
          return errors::Unknown(iterator_outcome.GetError().GetExceptionName(),
                                 ": ",
                                 iterator_outcome.GetError().GetMessage());
        }
        iterator_ = iterator_outcome.GetResult().GetShardIterator();
        return Status::OK();
      }

      mutex mu_;
      Aws::String iterator_ GUARDED_BY(mu_);
      std::unique_ptr<Aws::Kinesis::KinesisClient, decltype(&ShutdownClient)>
          client_ GUARDED_BY(mu_);
    };

    const std::string stream_;
    const std::string shard_;
    const bool read_indefinitely_;
    const int64 interval_;
  };
};

REGISTER_KERNEL_BUILDER(Name("KinesisDataset").Device(DEVICE_CPU),
                        KinesisDatasetOp);

}  // namespace tensorflow
