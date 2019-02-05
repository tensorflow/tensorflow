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

#include <sstream>

#include "include/json/json.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include "tensorflow/core/platform/cloud/oauth_client.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// The default initial delay between retries with exponential backoff.
constexpr int kInitialRetryDelayUsec = 500000;  // 0.5 sec

// The minimum time delta between now and the token expiration time
// for the token to be re-used.
constexpr int kExpirationTimeMarginSec = 60;

// The URL to retrieve the auth bearer token via OAuth with a refresh token.
constexpr char kOAuthV3Url[] = "https://www.googleapis.com/oauth2/v3/token";

// The URL to retrieve the auth bearer token via OAuth with a private key.
constexpr char kOAuthV4Url[] = "https://www.googleapis.com/oauth2/v4/token";

// The authentication token scope to request.
constexpr char kOAuthScope[] = "https://www.googleapis.com/auth/cloud-platform";

Status RetrieveGcsFs(OpKernelContext* ctx, RetryingGcsFileSystem** fs) {
  DCHECK(fs != nullptr);
  *fs = nullptr;

  FileSystem* filesystem = nullptr;
  TF_RETURN_IF_ERROR(
      ctx->env()->GetFileSystemForFile("gs://fake/file.text", &filesystem));
  if (filesystem == nullptr) {
    return errors::FailedPrecondition("The GCS file system is not registered.");
  }

  *fs = dynamic_cast<RetryingGcsFileSystem*>(filesystem);
  if (*fs == nullptr) {
    return errors::Internal(
        "The filesystem registered under the 'gs://' scheme was not a "
        "tensorflow::RetryingGcsFileSystem*.");
  }
  return Status::OK();
}

template <typename T>
Status ParseScalarArgument(OpKernelContext* ctx, StringPiece argument_name,
                           T* output) {
  const Tensor* argument_t;
  TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
  if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
    return errors::InvalidArgument(argument_name, " must be a scalar");
  }
  *output = argument_t->scalar<T>()();
  return Status::OK();
}

// GcsCredentialsOpKernel overrides the credentials used by the gcs_filesystem.
class GcsCredentialsOpKernel : public OpKernel {
 public:
  explicit GcsCredentialsOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    // Get a handle to the GCS file system.
    RetryingGcsFileSystem* gcs = nullptr;
    OP_REQUIRES_OK(ctx, RetrieveGcsFs(ctx, &gcs));

    string json_string;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "json", &json_string));

    Json::Value json;
    Json::Reader reader;
    std::stringstream json_stream(json_string);
    OP_REQUIRES(ctx, reader.parse(json_stream, json),
                errors::InvalidArgument("Could not parse json: ", json_string));

    OP_REQUIRES(
        ctx, json.isMember("refresh_token") || json.isMember("private_key"),
        errors::InvalidArgument("JSON format incompatible; did not find fields "
                                "`refresh_token` or `private_key`."));

    auto provider =
        tensorflow::MakeUnique<ConstantAuthProvider>(json, ctx->env());

    // Test getting a token
    string dummy_token;
    OP_REQUIRES_OK(ctx, provider->GetToken(&dummy_token));
    OP_REQUIRES(ctx, !dummy_token.empty(),
                errors::InvalidArgument(
                    "Could not retrieve a token with the given credentials."));

    // Set the provider.
    gcs->underlying()->SetAuthProvider(std::move(provider));
  }

 private:
  class ConstantAuthProvider : public AuthProvider {
   public:
    ConstantAuthProvider(const Json::Value& json,
                         std::unique_ptr<OAuthClient> oauth_client, Env* env,
                         int64 initial_retry_delay_usec)
        : json_(json),
          oauth_client_(std::move(oauth_client)),
          env_(env),
          initial_retry_delay_usec_(initial_retry_delay_usec) {}

    ConstantAuthProvider(const Json::Value& json, Env* env)
        : ConstantAuthProvider(json, tensorflow::MakeUnique<OAuthClient>(), env,
                               kInitialRetryDelayUsec) {}

    ~ConstantAuthProvider() override {}

    Status GetToken(string* token) override {
      mutex_lock l(mu_);
      const uint64 now_sec = env_->NowSeconds();

      if (!current_token_.empty() &&
          now_sec + kExpirationTimeMarginSec < expiration_timestamp_sec_) {
        *token = current_token_;
        return Status::OK();
      }
      if (json_.isMember("refresh_token")) {
        TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromRefreshTokenJson(
            json_, kOAuthV3Url, &current_token_, &expiration_timestamp_sec_));
      } else if (json_.isMember("private_key")) {
        TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromServiceAccountJson(
            json_, kOAuthV4Url, kOAuthScope, &current_token_,
            &expiration_timestamp_sec_));
      } else {
        return errors::FailedPrecondition(
            "Unexpected content of the JSON credentials file.");
      }

      *token = current_token_;
      return Status::OK();
    }

   private:
    Json::Value json_;
    std::unique_ptr<OAuthClient> oauth_client_;
    Env* env_;

    mutex mu_;
    string current_token_ GUARDED_BY(mu_);
    uint64 expiration_timestamp_sec_ GUARDED_BY(mu_) = 0;

    // The initial delay for exponential backoffs when retrying failed calls.
    const int64 initial_retry_delay_usec_;
    TF_DISALLOW_COPY_AND_ASSIGN(ConstantAuthProvider);
  };
};

REGISTER_KERNEL_BUILDER(Name("GcsConfigureCredentials").Device(DEVICE_CPU),
                        GcsCredentialsOpKernel);

class GcsBlockCacheOpKernel : public OpKernel {
 public:
  explicit GcsBlockCacheOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    // Get a handle to the GCS file system.
    RetryingGcsFileSystem* gcs = nullptr;
    OP_REQUIRES_OK(ctx, RetrieveGcsFs(ctx, &gcs));

    size_t max_cache_size, block_size, max_staleness;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<size_t>(ctx, "max_cache_size",
                                                    &max_cache_size));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<size_t>(ctx, "block_size", &block_size));
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<size_t>(ctx, "max_staleness", &max_staleness));

    if (gcs->underlying()->block_size() == block_size &&
        gcs->underlying()->max_bytes() == max_cache_size &&
        gcs->underlying()->max_staleness() == max_staleness) {
      LOG(INFO) << "Skipping resetting the GCS block cache.";
      return;
    }
    gcs->underlying()->ResetFileBlockCache(block_size, max_cache_size,
                                           max_staleness);
  }
};

REGISTER_KERNEL_BUILDER(Name("GcsConfigureBlockCache").Device(DEVICE_CPU),
                        GcsBlockCacheOpKernel);

}  // namespace
}  // namespace tensorflow
