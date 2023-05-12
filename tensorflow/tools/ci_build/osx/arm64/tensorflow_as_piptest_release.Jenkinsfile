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
                stage("Python 3.8") {
                    agent {
                        label "parallel-tensorflow-silicon"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"

                    }
                    steps {

                        sh '''
                            pyenv init -
                            pyenv global 3.8.13
                        '''

                        sh 'python --version'

                        git branch: "r2.13",
                            url: "https://github.com/tensorflow/tensorflow.git"

                        sh '''
                            pip install --upgrade pip
                            pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                        '''

                        configFileProvider([configFile(fileId: '561b70ba-de73-428b-919e-99346716e33c', targetLocation: '.macos.bazelrc')]) {}

                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        sh "cat .macos.bazelrc"
                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

                        sh '''
                            /opt/homebrew/bin/bazel build \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.8.13/lib/python3.8/site-packages" \
                            --config=macos_arm64 \
                            //tensorflow/tools/pip_package:build_pip_package
                            
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag --cpu
                            
                            mkdir -p "$(pwd)/bazel_pip"
                            ln -s "$(pwd)"/tensorflow "$(pwd)/bazel_pip"/tensorflow
                            pip install $(pwd)/tensorflow/pkg/*.whl
                            
                            bazel --bazelrc=".macos.bazelrc" test \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.8.13/lib/python3.8/site-packages" \
                            --config=macos_arm64 \
                            --config=pip
                        '''

                    }
                }
                stage("Python 3.9") {
                    agent {
                        label "parallel-tensorflow-silicon"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {

                        sh '''
                            pyenv init -
                            pyenv global 3.9.13
                        '''

                        sh 'python --version'

                        git branch: "r2.13",
                            url: "https://github.com/tensorflow/tensorflow.git"

                        sh '''
                            pip install --upgrade pip
                            pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                        '''

                        configFileProvider([configFile(fileId: '561b70ba-de73-428b-919e-99346716e33c', targetLocation: '.macos.bazelrc')]) {}

                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        sh "cat .macos.bazelrc"
                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

                        sh '''
                            /opt/homebrew/bin/bazel build \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.9.13/lib/python3.9/site-packages" \
                            --config=macos_arm64 \
                            //tensorflow/tools/pip_package:build_pip_package
                                
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag --cpu
                            
                            mkdir -p "$(pwd)/bazel_pip"
                            ln -s "$(pwd)"/tensorflow "$(pwd)/bazel_pip"/tensorflow
                            pip install $(pwd)/tensorflow/pkg/*.whl
                            
                            bazel --bazelrc=".macos.bazelrc" test \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.9.13/lib/python3.9/site-packages" \
                            --config=macos_arm64 \
                            --config=pip
                            '''
                    }
                }
                stage("Python 3.10") {
                    agent {
                        label "parallel-tensorflow-silicon"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {
                        sh '''
                            pyenv init -
                            pyenv global 3.10.4
                        '''

                        sh 'python --version'

                        git branch: "r2.13",
                            url: "https://github.com/tensorflow/tensorflow.git"

                        sh '''
                            pip install --upgrade pip
                            pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                        '''

                        configFileProvider([configFile(fileId: '561b70ba-de73-428b-919e-99346716e33c', targetLocation: '.macos.bazelrc')]) {}

                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        sh "cat .macos.bazelrc"
                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

                        sh '''
                            /opt/homebrew/bin/bazel build \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.10.4/lib/python3.10/site-packages" \
                            --config=macos_arm64 \
                            //tensorflow/tools/pip_package:build_pip_package
                                
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag --cpu
                            
                            mkdir -p "$(pwd)/bazel_pip"
                            ln -s "$(pwd)"/tensorflow "$(pwd)/bazel_pip"/tensorflow
                            pip install $(pwd)/tensorflow/pkg/*.whl
                            
                            bazel --bazelrc=".macos.bazelrc" test \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.10.4/lib/python3.10/site-packages" \
                            --config=macos_arm64 \
                            --config=pip
                            '''

                    }
                }
                stage("Python 3.11") {
                    agent {
                        label "parallel-tensorflow-silicon"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {
                        sh '''
                            pyenv init -
                            pyenv global 3.11.2
                        '''

                        sh 'python --version'

                        git branch: "r2.13",
                            url: "https://github.com/tensorflow/tensorflow.git"

                        sh '''
                            pip install --upgrade pip
                            pip install -r ./tensorflow/tools/ci_build/release/requirements_mac.txt
                        '''

                        configFileProvider([configFile(fileId: '561b70ba-de73-428b-919e-99346716e33c', targetLocation: '.macos.bazelrc')]) {}

                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
                        sh "cat .macos.bazelrc"
                        echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

                        sh '''
                            /opt/homebrew/bin/bazel build \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.11.2/lib/python3.11/site-packages" \
                            --config=macos_arm64 \
                            //tensorflow/tools/pip_package:build_pip_package
                                
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag
                            ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "./tensorflow/pkg" --nightly_flag --cpu
                            
                            mkdir -p "$(pwd)/bazel_pip"
                            ln -s "$(pwd)"/tensorflow "$(pwd)/bazel_pip"/tensorflow
                            pip install $(pwd)/tensorflow/pkg/*.whl
                            
                            bazel --bazelrc=".macos.bazelrc" test \
                            --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.11.2/lib/python3.11/site-packages" \
                            --config=macos_arm64 \
                            --config=pip
                            '''

                    }
                }
            }
        }
    }
}