/*
Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

pipeline {
    agent none
    environment {
        RELEASE_BRANCH = 'r2.15'
    }
    stages {
        stage("Build Tensorflow") {
            parallel {
                stage("Python 3.9") {
                    agent {
                        label "nightly-build-release"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                        TF_PYTHON_VERSION=3.9
                    }
                    steps {
                        dir('tensorflow') {

                            sh '''
                                pyenv init -
                                pyenv global 3.9.13
                            '''

                            sh 'python --version'

                            git branch: "${RELEASE_BRANCH}",
                                url: "https://github.com/tensorflow/tensorflow.git"

                            sh '''
                                pip install --upgrade pip
                                pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                            '''

                            // Once the build is successful, the "target - (build_pip_package)" had set two mandatory
                            // parameters to be passed. Which are - "--output-name" and "--project-name".
                            // The project_name was not dealt with because it was given a default value here in the 
                            // pipeline. Since the output_name was not mentioned, and since it is cannot be hardcoded
                            // as it will be ".whl" file, we need to handle it dynamically to get the file name of the
                            // ".whl" file, once created under the "dist" directory on the successful execution of 
                            // "Binary Distribution" with helper "python setup.py bdist_wheel" command.

                            // The same code is reproduced for all the following pyhton versions.

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                //tensorflow/tools/pip_package:build_pip_package
                            '''

                            sh 'python setup.py bdist_wheel'

                            sh '''
                                WHEEL_FILE=$(ls dist/*.whl)
                                IFS='-' read -ra TAGS <<< "$WHEEL_FILE"
                                VERSION=${TAGS[1]}
                                PYTHON_TAG=${TAGS[2]}
                                ABI_TAG=${TAGS[3]}
                                PLATFORM_TAG=${TAGS[4]}
                                OUTPUT_NAME="tensorflow-${VERSION}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl"
                                mv dist/*.whl ./$OUTPUT_NAME

                                # Use the dynamically set variables in the final packaging command
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                --output_name "$OUTPUT_NAME"
                                dist
                            '''
                        }
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
                stage("Python 3.10") {
                    agent {
                        label "nightly-build-release"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                        TF_PYTHON_VERSION=3.10
                    }
                    steps {
                        dir('tensorflow') {

                            sh '''
                                pyenv init -
                                pyenv global 3.10.4
                            '''
                            
                            sh 'python --version'

                            git branch: "${RELEASE_BRANCH}",
                                url: "https://github.com/tensorflow/tensorflow.git"

                            sh '''
                                pip install --upgrade pip
                                pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                            '''

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                //tensorflow/tools/pip_package:build_pip_package
                            '''

                            sh 'python setup.py bdist_wheel'

                            sh '''
                                WHEEL_FILE=$(ls dist/*.whl)
                                IFS='-' read -ra TAGS <<< "$WHEEL_FILE"
                                VERSION=${TAGS[1]}
                                PYTHON_TAG=${TAGS[2]}
                                ABI_TAG=${TAGS[3]}
                                PLATFORM_TAG=${TAGS[4]}
                                OUTPUT_NAME="tensorflow-${VERSION}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl"
                                mv dist/*.whl ./$OUTPUT_NAME

                                # Use the dynamically set variables in the final packaging command
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                --output_name "$OUTPUT_NAME"
                                dist
                            '''
                        }
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
                stage("Python 3.11") {
                    agent {
                        label "nightly-build-release"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                        TF_PYTHON_VERSION=3.11
                    }
                    steps {
                        dir('tensorflow') {

                            sh '''
                                pyenv init -
                                pyenv global 3.11.2
                            '''
                            
                            sh 'python --version'

                            git branch: "${RELEASE_BRANCH}",
                                url: "https://github.com/tensorflow/tensorflow.git"

                            sh '''
                                pip install --upgrade pip
                                pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                            '''

                            // The main reason to add the below script is to generate the BUILD file from BUILD.tpl
                            // We are doing that because we are fetching the username of the user's system, and 
                            // dynamically setting that to the path to fetch the "PYHTON_BIN_PATH_WINDOWS" 
                            // environment variable in the BUILD.tpl files, and then running the bazel build command.  
                            // sh 'python tensorflow/tools/ci_build/generate_build.py'

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                //tensorflow/tools/pip_package:build_pip_package
                            '''

                            sh 'python setup.py bdist_wheel'

                            sh '''
                                WHEEL_FILE=$(ls dist/*.whl)
                                IFS='-' read -ra TAGS <<< "$WHEEL_FILE"
                                VERSION=${TAGS[1]}
                                PYTHON_TAG=${TAGS[2]}
                                ABI_TAG=${TAGS[3]}
                                PLATFORM_TAG=${TAGS[4]}
                                OUTPUT_NAME="tensorflow-${VERSION}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl"
                                mv dist/*.whl ./$OUTPUT_NAME

                                # Use the dynamically set variables in the final packaging command
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                --output_name "$OUTPUT_NAME"
                                dist
                            '''
                        }
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
            }
        } 
    }
}
