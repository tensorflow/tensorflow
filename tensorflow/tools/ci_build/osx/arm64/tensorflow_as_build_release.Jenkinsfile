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
                        label "nightly-build"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {
                        dir('tensorflow') {

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

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.8.13/lib/python3.8/site-packages" \
                                //tensorflow/tools/pip_package:build_pip_package
                                    
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                dist
                            '''
                        }

                        sh '''
                            python -m pip install ${WORKSPACE}/tensorflow/dist/*.whl

                            python -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.keras" in tf.keras.__name__ else 1)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.estimator" in tf.estimator.__name__ else 1)'
                        '''
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
                stage("Python 3.9") {
                    agent {
                        label "nightly-build"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {
                        dir('tensorflow') {

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

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.9.13/lib/python3.9/site-packages" \
                                //tensorflow/tools/pip_package:build_pip_package
                                    
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                dist
                                '''
                        }

                        sh '''
                            python -m pip install ${WORKSPACE}/tensorflow/dist/*.whl

                            python -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.keras" in tf.keras.__name__ else 1)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.estimator" in tf.estimator.__name__ else 1)'
                        '''
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
                stage("Python 3.10") {
                    agent {
                        label "nightly-build"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {
                        dir('tensorflow') {

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

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.10.4/lib/python3.10/site-packages" \
                                //tensorflow/tools/pip_package:build_pip_package
                                
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                dist
                            '''
                        }

                        sh '''
                            python -m pip install ${WORKSPACE}/tensorflow/dist/*.whl

                            python -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.keras" in tf.keras.__name__ else 1)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.estimator" in tf.estimator.__name__ else 1)'
                        '''
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
                stage("Python 3.11") {
                    agent {
                        label "nightly-build"
                    }
                    environment {
                        PYENV_ROOT="$HOME/.pyenv"
                        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
                    }
                    steps {
                        dir('tensorflow') {

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

                            sh '''
                                /opt/homebrew/bin/bazel --bazelrc="${WORKSPACE}/tensorflow/tools/ci_build/osx/arm64/.macos.bazelrc" build \
                                --action_env PYTHON_LIB_PATH="/Users/admin/.pyenv/versions/3.11.2/lib/python3.11/site-packages" \
                                //tensorflow/tools/pip_package:build_pip_package
                                
                                ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
                                --project_name tensorflow_macos \
                                dist
                            '''
                        }

                        sh '''
                            python -m pip install ${WORKSPACE}/tensorflow/dist/*.whl

                            python -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.keras" in tf.keras.__name__ else 1)'
                            python -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.estimator" in tf.estimator.__name__ else 1)'
                        '''
                            
                        archiveArtifacts artifacts: "tensorflow/dist/*.whl", followSymlinks: false, onlyIfSuccessful: true
                    }
                }
            }
        } 
    }
}