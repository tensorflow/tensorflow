#!/usr/bin/env bash
# Build TensorFlow from source build script
if [ -z "$PYTHON_BIN_PATH" ]; then
  PYTHON_BIN_PATH=$(which python || which python3 || true)
fi

# Setup environment variables
CONFIGURE_DIR=$(dirname "$0")
CURRENT_DIR=pwd
PYTHON_VERSION=$PYTHON_BIN_PATH

## SETUP UTILS ##
function mac_setup_xcode() {
 echo
 echo "Setting XCode for development. This may require your password."
 echo
 sudo xcodebuild -license accept
}

function mac_setup_bazel() {
    export BAZEL_VERSION-0.26.1
    export INSTALL_URL=https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-darwin-x86_64.sh
    export FILE_NAME=bazel-0.26.1-installer-darwin-x86_64.sh
    wget $INSTALL_URL
    chmod +x $FILE_NAME
    bash $FILE_NAME
}

function linux_setup_install() {
    sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3
}

function linux_setup_bazel() {
    export INSTALL_URL=https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
    export FILE_NAME=bazel-0.26.1-installer-linux-x86_64.sh
    wget $INSTALL_URL
    chmod +x $FILE_NAME
    bash $FILE_NAME
}


function windows_setup_install() {
    pacman -S zip unzip patch diffutils git
}
function windows_setup_bazel() {
    export INSTALL_URL=https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-windows-x86_64.exe
    export FILE_NAME=bazel-0.26.1-windows-x86_64.exe
    wget $INSTALL_URL
    chmod +x $FILE_NAME
    bash $FILE_NAME
}

## CALL SETUP UTILS ##

function mac_setup() {
 echo "Setting up for Mac."
 mac_setup_xcode
 mac_setup_bazel
 echo "Setup for Mac done."
}

function linux_setup() {
 echo "Setting up for Linux."
 linux_setup_install
 linux_setup_bazel
 echo "Setup for Linux done."
}

function windows_setup() {
 echo "Setting up for Window."
 windows_setup_install
 windows_setup_bazel
 echo "Setup for Windows done."
}

# read -p "Are you using Linux (y/n)?" choice
# case "$choice" in 
#   y|Y) linux_setup;;
#   n|N) esac;;
#   *) echo "invalid";;
# esac

# read -p "Are you using Windows (y/n)?" choice
# case "$choice" in 
#   y|Y) windows_setup;;
#   n|N) esac;;
#   *) echo "invalid";;
# esac

# read -p "Are you using macOS (y/n)?" choice
# case "$choice" in 
#   y|Y ) mac_setup;;
#   n|N ) esac;;
#   * ) echo "invalid";;
# esac

read -r -p "Are you using Linux? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY])
 linux_setup
 ;;
    [nN][oO]|[nN])
 echo "No. continuing."
 ;;
    *)
 echo "Invalid input. Exiting..."
 exit 1
 ;;
esac

read -r -p "Are you using Mac? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY])
 mac_setup
 ;;
    [nN][oO]|[nN])
 echo "No. Continuing..."
 ;;
    *)
 echo "Invalid input. Exiting..."
 exit 1
 ;;
esac

read -r -p "Are you using Windows? [Y/n] " input
case $input in
    [yY][eE][sS]|[yY])
 windows_setup
 ;;
    [nN][oO]|[nN])
 echo "No. Continuing..."
 ;;
    *)
 echo "Invalid input. Exiting..."
 exit 1
 ;;
esac

# @TODO: @aaronhma make the next line better
tput setaf 2; echo "Successfully finished running program: "; tput setaf 6; echo "build.sh"