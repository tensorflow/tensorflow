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
                        label "silicon-ci-release"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                        TF_PYTHON_VERSION=3.9
                    }
                    steps {

                        sh '''
                            echo 3.9.13 > /Users/admin/.python-version
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

                        sh '''
                            bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" test \
                            --config=nonpip
                            '''
                    }
                }
                stage("Python 3.10") {
                    agent {
                        label "silicon-ci-release"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                        TF_PYTHON_VERSION=3.10
                    }
                    steps {
                        sh '''
                            echo 3.10.4 > /Users/admin/.python-version
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
                            bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" test \
                            --config=nonpip
                            '''

                    }
                }
                stage("Python 3.11") {
                    agent {
                        label "silicon-ci-release"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                        TF_PYTHON_VERSION=3.11
                    }
                    steps {
                        sh '''
                            echo 3.11.2 > /Users/admin/.python-version
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

                        sh '''
                            bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" test \
                            --config=nonpip
                            '''

                    }
                }
            }
        }
    }
}
