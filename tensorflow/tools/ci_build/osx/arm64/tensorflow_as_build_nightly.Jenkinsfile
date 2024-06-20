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
    stages {
        stage("Build Tensorflow") {
            parallel {
                stage("Python 3.9") {
                    agent {
                        label "nightly-build"
                    }

                    // LINUX FORMAT:

                    // environment {
                    //     PYENV_ROOT="$HOME/.pyenv"
                    //     PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    //     TF_PYTHON_VERSION=3.9
                    // }

                    // WINDOWS FORMAT:

                    environment {
                        PYENV_ROOT="$env:USERPROFILE/.pyenv"
                        PATH="$PYENV_ROOT/shims;C:/opt/homebrew/bin;$env:PATH"
                        TF_PYTHON_VERSION=3.9
                        PYTHONPATH="$WORKSPACE/tensorflow/tools/pip_package"  
                    }
                    steps {
                        dir('tensorflow') {

                            sh '''
                                pyenv init -
                                pyenv global 3.9.13
                            '''

                            sh 'python --version'

                            git branch: "nightly",
                                url: "https://github.com/tensorflow/tensorflow.git"
                                

                            sh '''
                                pip install --upgrade pip
                                pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                                python tensorflow/tools/ci_build/update_version.py --nightly
                            '''

                            // Install Pillow for metal plugin tests
                            sh 'pip install Pillow'

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
                            
                            sh '''
                                # Use the dynamically set variables in the final packaging command
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                    --nightly_flag \
                                    --output-name "${OUTPUT_NAME}" \
                                    --project-name "tf-nightly-macos" 
                            '''
                        }
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true

                        sh 'python ${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/tensorflow_metal_plugin_test.py'

                    }
                }
                stage("Python 3.10") {
                    agent {
                        label "nightly-build"
                    }

                    // LINUX FORMAT:

                    // environment {
                    //     PYENV_ROOT="$HOME/.pyenv"
                    //     PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    //     TF_PYTHON_VERSION=3.10
                    // }

                    // WINDOWS FORMAT:

                    environment {
                        PYENV_ROOT="$env:USERPROFILE/.pyenv"
                        PATH="$PYENV_ROOT/shims;C:/opt/homebrew/bin;$env:PATH"
                        TF_PYTHON_VERSION=3.10
                        PYTHONPATH="$WORKSPACE/tensorflow/tools/pip_package"  
                    }
                    steps {
                        dir('tensorflow') {

                            sh '''
                                pyenv init -
                                pyenv global 3.10.4
                            '''
                            
                            sh 'python --version'

                            git branch: "nightly",
                                url: "https://github.com/tensorflow/tensorflow.git"

                            sh '''
                                pip install --upgrade pip
                                pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                                python tensorflow/tools/ci_build/update_version.py --nightly
                            '''

                            // Install Pillow for metal plugin tests
                            sh 'pip install Pillow'

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                //tensorflow/tools/pip_package:build_pip_package
                            '''

                            sh '''
                                # Use the dynamically set variables in the final packaging command
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                    --nightly_flag \
                                    --output-name "${OUTPUT_NAME}" \
                                    --project-name "tf-nightly-macos" 
                            '''
                        }
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true

                        sh 'python ${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/tensorflow_metal_plugin_test.py'
                    }
                }
                stage("Python 3.11") {
                    agent {
                        label "nightly-build"
                    }

                    // LINUX FORMAT:

                    // environment {
                    //     PYENV_ROOT="$HOME/.pyenv"
                    //     PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    //     TF_PYTHON_VERSION=3.11
                    // }

                    // WINDOWS FORMAT:

                    environment {
                        PYENV_ROOT="$env:USERPROFILE/.pyenv"
                        PATH="$PYENV_ROOT/shims;C:/opt/homebrew/bin;$env:PATH"
                        TF_PYTHON_VERSION=3.11
                        PYTHONPATH="$WORKSPACE/tensorflow/tools/pip_package"  
                    }
                    steps {

                        dir('tensorflow') {

                            // Diagnostic echo statements to check environment variables
                            sh 'echo PYENV_ROOT: $PYENV_ROOT'
                            sh 'echo PATH: $PATH'
                            sh 'echo TF_PYTHON_VERSION: $TF_PYTHON_VERSION'
                            sh 'echo PYTHONPATH: $PYTHONPATH'
                            sh 'echo WORKSPACE: $WORKSPACE'
                            sh 'echo USERPROFILE: $USERPROFILE'

                            sh '''
                                pyenv init -
                                pyenv global 3.11.2
                            '''
                            
                            sh 'python --version'

                            git branch: "nightly",
                                url: "https://github.com/tensorflow/tensorflow.git"

                            sh '''
                                pip install --upgrade pip
                                pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                                python tensorflow/tools/ci_build/update_version.py --nightly
                            '''

                            // Install Pillow for metal plugin tests
                            sh 'pip install Pillow'

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                //tensorflow/tools/pip_package:build_pip_package
                            '''

                            sh '''
                                # Use the dynamically set variables in the final packaging command
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                    --nightly_flag \
                                    --output-name "${OUTPUT_NAME}" \
                                    --project-name "tf-nightly-macos" 
                            '''
                        }
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true

                        sh 'python ${WORKSPACE}/tensorflow/tensorflow/tools/ci_build/osx/arm64/tensorflow_metal_plugin_test.py'
                    }
                }
            }
        } 
    }
    post {
        always {
            build 'upload-nightly'
        }
    }
}
