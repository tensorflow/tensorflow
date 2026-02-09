#!/bin/bash
set -e  # fail and exit on any command erroring

# Read the bazel version from the path; slowly moving up parents
while [[ $PWD != / && -z $(find ~+/ -maxdepth 1 -type f -name .bazelversion) ]]
do cd ..; done
BAZEL_VERSION=$(head -n 1 .bazelversion)
if [[ -z "$BAZEL_VERSION" ]]; then exit 1; fi

# If (due to e.g., $PATH settings), bazel is actually running bazelisk,
# https://github.com/bazelbuild/bazelisk, make sure it downloads the bazel
# version we want (not the latest one, which we haven't tested yet).
export USE_BAZEL_VERSION=${BAZEL_VERSION}

# Install the given bazel version.
function update_bazel_version {
  local BAZEL_VERSION=$1
  local BAZEL_URL=https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}
  case "$(uname -s)" in
    MSYS_NT*)
      # Windows is a special case: there is no installer, just a big .exe, see
      # https://docs.bazel.build/versions/master/install-windows.html
      local INSTALL_DIR=`pwd`/bazel-${BAZEL_VERSION}
      mkdir -p ${INSTALL_DIR}
      wget -q ${BAZEL_URL}/bazel-${BAZEL_VERSION}-windows-x86_64.exe \
        -O ${INSTALL_DIR}/bazel.exe
      PATH=${INSTALL_DIR}:$PATH
      return
      ;;
    Darwin)
      echo "Delete bazel install cache to address http://b/161389448"
      rm -fr /var/tmp/_bazel_kbuilder/install/9be5dadb2a2b38082dbe665bf2db6464
      local OS_TOKEN=darwin
      ;;
    Linux)
      local OS_TOKEN=linux
      ;;
    *)
      die "Unknown OS: $(uname -s)"
      ;;
  esac
  local BAZEL_INSTALLER=bazel-${BAZEL_VERSION}-installer-${OS_TOKEN}-x86_64.sh
  rm -f ${BAZEL_INSTALLER}
  curl -fSsL -O ${BAZEL_URL}/${BAZEL_INSTALLER}
  chmod +x ${BAZEL_INSTALLER}

  # Due to --user, the installer puts the bazel binary into $HOME/bin (where we
  # have access rights).  We add that at the top of our $PATH.
  ./${BAZEL_INSTALLER} --user --skip-uncompress
  PATH=$HOME/bin:$PATH
  rm -f ${BAZEL_INSTALLER}
}

# First check if new version of Bazel needs to be installed.
# Install different bazel if needed
installed_bazel_version=$(bazel version | grep label | sed -e 's/.*: //')
if [ "$installed_bazel_version" == "$BAZEL_VERSION" ]; then
  echo "Bazel version ${BAZEL_VERSION} is already correctly installed."
else
  update_bazel_version ${BAZEL_VERSION}
  which bazel
  bazel version
fi
