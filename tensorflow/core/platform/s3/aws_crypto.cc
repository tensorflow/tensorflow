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
#include "tensorflow/core/platform/s3/aws_crypto.h"
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/rand.h>

#include <aws/core/utils/crypto/HashResult.h>
#include <aws/s3/S3Client.h>

namespace tensorflow {

class AWSSha256HMACOpenSSLImpl : public Aws::Utils::Crypto::HMAC {
 public:
  AWSSha256HMACOpenSSLImpl() {}

  virtual ~AWSSha256HMACOpenSSLImpl() = default;

  virtual Aws::Utils::Crypto::HashResult Calculate(
      const Aws::Utils::ByteBuffer& toSign,
      const Aws::Utils::ByteBuffer& secret) override {
    unsigned int length = SHA256_DIGEST_LENGTH;
    Aws::Utils::ByteBuffer digest(length);
    memset(digest.GetUnderlyingData(), 0, length);

    HMAC_CTX ctx;
    HMAC_CTX_init(&ctx);

    HMAC_Init_ex(&ctx, secret.GetUnderlyingData(),
                 static_cast<int>(secret.GetLength()), EVP_sha256(), NULL);
    HMAC_Update(&ctx, toSign.GetUnderlyingData(), toSign.GetLength());
    HMAC_Final(&ctx, digest.GetUnderlyingData(), &length);
    HMAC_CTX_cleanup(&ctx);

    return Aws::Utils::Crypto::HashResult(std::move(digest));
  }
};

class AWSSha256OpenSSLImpl : public Aws::Utils::Crypto::Hash {
 public:
  AWSSha256OpenSSLImpl() {}

  virtual ~AWSSha256OpenSSLImpl() = default;

  virtual Aws::Utils::Crypto::HashResult Calculate(
      const Aws::String& str) override {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.data(), str.size());

    Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    SHA256_Final(hash.GetUnderlyingData(), &sha256);

    return Aws::Utils::Crypto::HashResult(std::move(hash));
  }

  virtual Aws::Utils::Crypto::HashResult Calculate(
      Aws::IStream& stream) override {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);

    auto currentPos = stream.tellg();
    if (currentPos == std::streampos(std::streamoff(-1))) {
      currentPos = 0;
      stream.clear();
    }

    stream.seekg(0, stream.beg);

    char streamBuffer
        [Aws::Utils::Crypto::Hash::INTERNAL_HASH_STREAM_BUFFER_SIZE];
    while (stream.good()) {
      stream.read(streamBuffer,
                  Aws::Utils::Crypto::Hash::INTERNAL_HASH_STREAM_BUFFER_SIZE);
      auto bytesRead = stream.gcount();

      if (bytesRead > 0) {
        SHA256_Update(&sha256, streamBuffer, static_cast<size_t>(bytesRead));
      }
    }

    stream.clear();
    stream.seekg(currentPos, stream.beg);

    Aws::Utils::ByteBuffer hash(SHA256_DIGEST_LENGTH);
    SHA256_Final(hash.GetUnderlyingData(), &sha256);

    return Aws::Utils::Crypto::HashResult(std::move(hash));
  }
};

class AWSSecureRandomBytesImpl : public Aws::Utils::Crypto::SecureRandomBytes {
public:
  AWSSecureRandomBytesImpl() {}
  virtual ~AWSSecureRandomBytesImpl() = default;
  virtual void GetBytes(unsigned char* buffer, size_t bufferSize) override {
    assert(buffer);
    int success = RAND_bytes(buffer, static_cast<int>(bufferSize));
    if (success != 1) {
      m_failure = true;
    }
  }

private:
  bool m_failure;
};

std::shared_ptr<Aws::Utils::Crypto::Hash>
AWSSHA256Factory::CreateImplementation() const {
  return Aws::MakeShared<AWSSha256OpenSSLImpl>(AWSCryptoAllocationTag);
}

std::shared_ptr<Aws::Utils::Crypto::HMAC>
AWSSHA256HmacFactory::CreateImplementation() const {
  return Aws::MakeShared<AWSSha256HMACOpenSSLImpl>(AWSCryptoAllocationTag);
}

std::shared_ptr<Aws::Utils::Crypto::SecureRandomBytes>
AWSSecureRandomFactory::CreateImplementation() const {
  return Aws::MakeShared<AWSSecureRandomBytesImpl>(AWSCryptoAllocationTag);
}

}  // namespace tensorflow
