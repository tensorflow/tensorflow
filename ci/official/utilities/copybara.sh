#!/bin/bash
# Destroy any existing github code
rm -rf "$TFCI_GIT_DIR"
mkdir -p "$TFCI_GIT_DIR"
# Note (See b/257092152): --ecatcher_service, --execution-registry-service
# and --nostreamz-monitoring avoid a Java linker error.
java \
  -Dcom.google.devtools.copybara.runfiles.path="$KOKORO_ARTIFACTS_DIR/mpm/devtools/copybara/tool_kokoro/google3" \
  -jar "$KOKORO_ARTIFACTS_DIR/mpm/devtools/copybara/tool_kokoro/copybara_on_kokoro_deploy.jar" \
  --buildifier-bin "$KOKORO_ARTIFACTS_DIR/mpm/devtools/buildifier/buildifier/buildifier" \
  --buildozer-bin "$KOKORO_ARTIFACTS_DIR/mpm/devtools/buildozer1/buildozer" \
  --leakr-bin "$KOKORO_ARTIFACTS_DIR/mpm/testing/leakr/parser/parser" \
  --disable-reversible-check \
  --ignore-noop \
  --ecatcher_service "" \
  --execution-registry-service "" \
  --nostreamz-monitoring \
  --folder-dir "$TFCI_GIT_DIR" \
  "$KOKORO_PIPER_DIR/google3/third_party/tensorflow/copy.bara.sky" \
  g3folder_to_gitfolder_transform_only \
  "$KOKORO_PIPER_DIR"
PENDING_CL_NUMBER=$(head -n 1 "$KOKORO_PIPER_DIR/presubmit_request.txt" | cut -d" " -f2)
export TFCI_DOCKER_IMAGE=$(echo "$TFCI_DOCKER_IMAGE" | sed "s/PENDING_CL_NUMBER/$PENDING_CL_NUMBER/g")
