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

#include "tensorflow/core/platform/cloud/google_auth_provider.h"
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include "include/json/json.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/cloud/retrying_utils.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {

// The environment variable pointing to the file with local
// Application Default Credentials.
constexpr char kGoogleApplicationCredentials[] =
    "GOOGLE_APPLICATION_CREDENTIALS";

// The environment variable to override token generation for testing.
constexpr char kGoogleAuthTokenForTesting[] = "GOOGLE_AUTH_TOKEN_FOR_TESTING";

// The environment variable which can override '~/.config/gcloud' if set.
constexpr char kCloudSdkConfig[] = "CLOUDSDK_CONFIG";

// The default path to the gcloud config folder, relative to the home folder.
constexpr char kGCloudConfigFolder[] = ".config/gcloud/";

// The name of the well-known credentials JSON file in the gcloud config folder.
constexpr char kWellKnownCredentialsFile[] =
    "application_default_credentials.json";

// The minimum time delta between now and the token expiration time
// for the token to be re-used.
constexpr int kExpirationTimeMarginSec = 60;

// The URL to retrieve the auth bearer token via OAuth with a refresh token.
constexpr char kOAuthV3Url[] = "https://www.googleapis.com/oauth2/v3/token";

// The URL to retrieve the auth bearer token via OAuth with a private key.
constexpr char kOAuthV4Url[] = "https://www.googleapis.com/oauth2/v4/token";

// The URL to retrieve the auth bearer token when running in Google Compute
// Engine.
constexpr char kGceTokenUrl[] =
    "http://metadata/computeMetadata/v1/instance/service-accounts/default/"
    "token";

// The authentication token scope to request.
constexpr char kOAuthScope[] = "https://www.googleapis.com/auth/cloud-platform";

// The default initial delay between retries with exponential backoff.
constexpr int kInitialRetryDelayUsec = 500000;  // 0.5 sec

/// Returns whether the given path points to a readable file.
bool IsFile(const string& filename) {
  std::ifstream fstream(filename.c_str());
  return fstream.good();
}

/// Returns the credentials file name from the env variable.
Status GetEnvironmentVariableFileName(string* filename) {
  if (!filename) {
    return errors::FailedPrecondition("'filename' cannot be nullptr.");
  }
  const char* result = std::getenv(kGoogleApplicationCredentials);
  if (!result || !IsFile(result)) {
    return errors::NotFound(strings::StrCat("$", kGoogleApplicationCredentials,
                                            " is not set or corrupt."));
  }
  *filename = result;
  return Status::OK();
}

/// Returns the well known file produced by command 'gcloud auth login'.
Status GetWellKnownFileName(string* filename) {
  if (!filename) {
    return errors::FailedPrecondition("'filename' cannot be nullptr.");
  }
  string config_dir;
  const char* config_dir_override = std::getenv(kCloudSdkConfig);
  if (config_dir_override) {
    config_dir = config_dir_override;
  } else {
    // Determine the home dir path.
    const char* home_dir = std::getenv("HOME");
    if (!home_dir) {
      return errors::FailedPrecondition("Could not read $HOME.");
    }
    config_dir = io::JoinPath(home_dir, kGCloudConfigFolder);
  }
  auto result = io::JoinPath(config_dir, kWellKnownCredentialsFile);
  if (!IsFile(result)) {
    return errors::NotFound(
        "Could not find the credentials file in the standard gcloud location.");
  }
  *filename = result;
  return Status::OK();
}

}  // namespace

GoogleAuthProvider::GoogleAuthProvider()
    : GoogleAuthProvider(
          std::unique_ptr<OAuthClient>(new OAuthClient()),
          std::unique_ptr<HttpRequest::Factory>(new HttpRequest::Factory()),
          Env::Default(), kInitialRetryDelayUsec) {}

GoogleAuthProvider::GoogleAuthProvider(
    std::unique_ptr<OAuthClient> oauth_client,
    std::unique_ptr<HttpRequest::Factory> http_request_factory, Env* env,
    int64 initial_retry_delay_usec)
    : oauth_client_(std::move(oauth_client)),
      http_request_factory_(std::move(http_request_factory)),
      env_(env),
      initial_retry_delay_usec_(initial_retry_delay_usec) {}

Status GoogleAuthProvider::GetToken(string* t) {
  mutex_lock lock(mu_);
  const uint64 now_sec = env_->NowSeconds();

  if (!current_token_.empty() &&
      now_sec + kExpirationTimeMarginSec < expiration_timestamp_sec_) {
    *t = current_token_;
    return Status::OK();
  }

  if (GetTokenForTesting().ok()) {
    *t = current_token_;
    return Status::OK();
  }

  auto token_from_files_status = GetTokenFromFiles();
  auto token_from_gce_status =
      token_from_files_status.ok() ? Status::OK() : GetTokenFromGce();

  if (token_from_files_status.ok() || token_from_gce_status.ok()) {
    *t = current_token_;
    return Status::OK();
  }

  LOG(WARNING)
      << "All attempts to get a Google authentication bearer token failed, "
      << "returning an empty token. Retrieving token from files failed with \""
      << token_from_files_status.ToString() << "\"."
      << " Retrieving token from GCE failed with \""
      << token_from_gce_status.ToString() << "\".";

  // Public objects can still be accessed with an empty bearer token,
  // so return an empty token instead of failing.
  *t = "";

  // From now on, always return the empty token.
  expiration_timestamp_sec_ = UINT64_MAX;
  current_token_ = "";

  return Status::OK();
}

Status GoogleAuthProvider::GetTokenFromFiles() {
  string credentials_filename;
  if (!GetEnvironmentVariableFileName(&credentials_filename).ok() &&
      !GetWellKnownFileName(&credentials_filename).ok()) {
    return errors::NotFound("Could not locate the credentials file.");
  }

  Json::Value json;
  Json::Reader reader;
  std::ifstream credentials_fstream(credentials_filename);
  if (!reader.parse(credentials_fstream, json)) {
    return errors::FailedPrecondition(
        "Couldn't parse the JSON credentials file.");
  }
  if (json.isMember("refresh_token")) {
    TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromRefreshTokenJson(
        json, kOAuthV3Url, &current_token_, &expiration_timestamp_sec_));
  } else if (json.isMember("private_key")) {
    TF_RETURN_IF_ERROR(oauth_client_->GetTokenFromServiceAccountJson(
        json, kOAuthV4Url, kOAuthScope, &current_token_,
        &expiration_timestamp_sec_));
  } else {
    return errors::FailedPrecondition(
        "Unexpected content of the JSON credentials file.");
  }
  return Status::OK();
}

Status GoogleAuthProvider::GetTokenFromGce() {
  const auto get_token_from_gce = [this]() {
    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    std::vector<char> response_buffer;
    const uint64 request_timestamp_sec = env_->NowSeconds();
    TF_RETURN_IF_ERROR(request->Init());
    TF_RETURN_IF_ERROR(request->SetUri(kGceTokenUrl));
    TF_RETURN_IF_ERROR(request->AddHeader("Metadata-Flavor", "Google"));
    TF_RETURN_IF_ERROR(request->SetResultBuffer(&response_buffer));
    TF_RETURN_IF_ERROR(request->Send());
    StringPiece response =
        StringPiece(&response_buffer[0], response_buffer.size());

    TF_RETURN_IF_ERROR(oauth_client_->ParseOAuthResponse(
        response, request_timestamp_sec, &current_token_,
        &expiration_timestamp_sec_));
    return Status::OK();
  };
  return RetryingUtils::CallWithRetries(get_token_from_gce,
                                        initial_retry_delay_usec_);
}

Status GoogleAuthProvider::GetTokenForTesting() {
  const char* token = std::getenv(kGoogleAuthTokenForTesting);
  if (!token) {
    return errors::NotFound("The env variable for testing was not set.");
  }
  expiration_timestamp_sec_ = UINT64_MAX;
  current_token_ = token;
  return Status::OK();
}

}  // namespace tensorflow
