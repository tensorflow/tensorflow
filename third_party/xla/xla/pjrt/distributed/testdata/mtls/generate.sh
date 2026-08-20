#!/bin/bash
# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Regenerates the test-only certificates and keys used by mtls_test.cc.
# Run from this directory: ./generate.sh
#
# Produces two self-signed P-256 CAs and six leaf identities (valid ~100y):
#
#   ca1.pem      root CA trusted by most tests
#   ca2.pem      second root CA, for trust-split tests
#   client.*     URI:spiffe://example.org/coord/client          (signed by ca1)
#   server.*     URI:spiffe://example.org/coord/server          (signed by ca1)
#   multi.*      URI:spiffe://other.example/x,
#                URI:spiffe://example.org/coord/multi           (signed by ca1)
#   evil.*       URI:spiffe://evil.example/intruder              (signed by ca1)
#   nouri.*      DNS:localhost, no URI SAN                       (signed by ca1)
#   outcast.*    URI:spiffe://example.org/coord/outcast         (signed by ca2)
#
# The CA private keys are deleted at the end; rerun the script to mint new
# leaves. These keys must never be used outside of tests.

set -euo pipefail
cd "$(dirname "$0")"

DAYS=36500

make_ca() {
  local name="$1"
  openssl req -x509 -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    -keyout "${name}.key" -out "${name}.pem" -days "${DAYS}" \
    -subj "/CN=${name}" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -addext "subjectKeyIdentifier=hash"
}

# make_leaf <name> <ca> <subjectAltName>
make_leaf() {
  local name="$1" ca="$2" san="$3"
  openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    -keyout "${name}.key" -out "${name}.csr" -subj "/CN=${name}"
  openssl x509 -req -in "${name}.csr" -CA "${ca}.pem" -CAkey "${ca}.key" \
    -set_serial "0x$(openssl rand -hex 8)" -days "${DAYS}" -out "${name}.pem" \
    -extfile <(printf '%s\n' \
      "basicConstraints=CA:FALSE" \
      "keyUsage=critical,digitalSignature" \
      "extendedKeyUsage=serverAuth,clientAuth" \
      "subjectKeyIdentifier=hash" \
      "authorityKeyIdentifier=keyid" \
      "subjectAltName=${san}")
  rm -f "${name}.csr"
}

make_ca ca1
make_ca ca2

make_leaf client ca1 "URI:spiffe://example.org/coord/client"
make_leaf server ca1 "URI:spiffe://example.org/coord/server"
make_leaf multi ca1 \
  "URI:spiffe://other.example/x,URI:spiffe://example.org/coord/multi"
make_leaf evil ca1 "URI:spiffe://evil.example/intruder"
make_leaf nouri ca1 "DNS:localhost"
make_leaf outcast ca2 "URI:spiffe://example.org/coord/outcast"

for leaf in client server multi evil nouri; do
  openssl verify -CAfile ca1.pem "${leaf}.pem" >/dev/null
done
openssl verify -CAfile ca2.pem outcast.pem >/dev/null

rm -f ca1.key ca2.key
echo "Regenerated mTLS test certificates in $(pwd)"
