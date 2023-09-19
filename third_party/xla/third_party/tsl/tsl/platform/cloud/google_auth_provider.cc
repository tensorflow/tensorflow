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

#include "tsl/platform/cloud/google_auth_provider.h"
#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#else
#include <sys/types.h>
#endif
#include <fstream>
#include <utility>

#include "absl/strings/match.h"
#include "json/json.h"
#include "tsl/platform/base64.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/retrying_utils.h"

namespace tsl {

namespace {

// The environment variable pointing to the file with local
// Application Default Credentials.
constexpr char kGoogleApplicationCredentials[] =
    "GOOGLE_APPLICATION_CREDENTIALS";

// The environment variable to override token generation for testing.
constexpr char kGoogleAuthTokenForTesting[] = "GOOGLE_AUTH_TOKEN_FOR_TESTING";

// The environment variable which can override '~/.config/gcloud' if set.
constexpr char kCloudSdkConfig[] = "CLOUDSDK_CONFIG";

// The environment variable used to skip attempting to fetch GCE credentials:
// setting this to 'true' (case insensitive) will skip attempting to contact
// the GCE metadata service.
constexpr char kNoGceCheck[] = "NO_GCE_CHECK";

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
constexpr char kGceTokenPath[] = "instance/service-accounts/default/token";

// The authentication token scope to request.
constexpr char kOAuthScope[] = "https://www.googleapis.com/auth/cloud-platform";

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
  return OkStatus();
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
  return OkStatus();
}

}  // namespace

GoogleAuthProvider::GoogleAuthProvider(
    std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client)
    : GoogleAuthProvider(std::unique_ptr<OAuthClient>(new OAuthClient()),
                         std::move(compute_engine_metadata_client),
                         Env::Default()) {}

GoogleAuthProvider::GoogleAuthProvider(
    std::unique_ptr<OAuthClient> oauth_client,
    std::shared_ptr<ComputeEngineMetadataClient> compute_engine_metadata_client,
    Env* env)
    : oauth_client_(std::move(oauth_client)),
      compute_engine_metadata_client_(
          std::move(compute_engine_metadata_client)),
      env_(env) {}

Status GoogleAuthProvider::GetToken(string* t) {
  mutex_lock lock(mu_);
  const uint64 now_sec = env_->NowSeconds();

  if (now_sec + kExpirationTimeMarginSec < expiration_timestamp_sec_) {
    *t = current_token_;
    return OkStatus();
  }

  if (GetTokenForTesting().ok()) {
    *t = current_token_;
    return OkStatus();
  }

  auto token_from_files_status = GetTokenFromFiles();
  if (token_from_files_status.ok()) {
    *t = current_token_;
    return OkStatus();
  }

  char* no_gce_check_var = std::getenv(kNoGceCheck);
  bool skip_gce_check = no_gce_check_var != nullptr &&
                        absl::EqualsIgnoreCase(no_gce_check_var, "true");
  Status token_from_gce_status;
  if (skip_gce_check) {
    token_from_gce_status =
        Status(absl::StatusCode::kCancelled,
               strings::StrCat("GCE check skipped due to presence of $",
                               kNoGceCheck, " environment variable."));
  } else {
    token_from_gce_status = GetTokenFromGce();
  }

  if (token_from_gce_status.ok()) {
    *t = current_token_;
    return OkStatus();
  }

  if (skip_gce_check) {
    LOG(INFO)
        << "Attempting an empty bearer token since no token was retrieved "
        << "from files, and GCE metadata check was skipped.";
  } else {
    LOG(WARNING)
        << "All attempts to get a Google authentication bearer token failed, "
        << "returning an empty token. Retrieving token from files failed with "
           "\""
        << token_from_files_status.ToString() << "\"."
        << " Retrieving token from GCE failed with \""
        << token_from_gce_status.ToString() << "\".";
  }

  // Public objects can still be accessed with an empty bearer token,
  // so return an empty token instead of failing.
  *t = "";

  // We only want to keep returning our empty token if we've tried and failed
  // the (potentially slow) task of detecting GCE.
  if (skip_gce_check) {
    expiration_timestamp_sec_ = 0;
  } else {
    expiration_timestamp_sec_ = UINT64_MAX;
  }
  current_token_ = "";

  return OkStatus();
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
  return OkStatus();
}

Status GoogleAuthProvider::GetTokenFromGce() {
  std::vector<char> response_buffer;
  const uint64 request_timestamp_sec = env_->NowSeconds();

  TF_RETURN_IF_ERROR(compute_engine_metadata_client_->GetMetadata(
      kGceTokenPath, &response_buffer));
  StringPiece response =
      StringPiece(&response_buffer[0], response_buffer.size());

  TF_RETURN_IF_ERROR(oauth_client_->ParseOAuthResponse(
      response, request_timestamp_sec, &current_token_,
      &expiration_timestamp_sec_));

  return OkStatus();
}

Status GoogleAuthProvider::GetTokenForTesting() {
  const char* token = std::getenv(kGoogleAuthTokenForTesting);
  if (!token) {
    return errors::NotFound("The env variable for testing was not set.");
  }
  expiration_timestamp_sec_ = UINT64_MAX;
  current_token_ = token;
  return OkStatus();
}

}  // namespace tsl
