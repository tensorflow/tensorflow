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
    agent { 
        label "nightly-upload" 
    }
    environment {
        PYENV_ROOT="$HOME/.pyenv"
        PATH="$PYENV_ROOT/shims:/opt/homebrew/bin/:$PATH"
        TWINE_NON_INTERACTIVE=true
    }
    stages {
        stage('build') {
            steps {

                git 'https://github.com/tensorflow/tensorflow'
                
                sh 'mkdir dist'
                
                copyArtifacts fingerprintArtifacts: true, projectName: 'tensorflow-as-build-nightly', selector: upstream()
                
                sh 'pyenv global 3.10.10'
                
                withCredentials([string(credentialsId: 'ef67da81-2d62-4ae6-a200-cbd2bcab8429', variable: 'PYPI_API_TOKEN')]) {
                    sh 'twine check tensorflow/dist/*'
                    sh 'twine upload tensorflow/dist/* -u __token__ -p $PYPI_API_TOKEN --verbose --disable-progress-bar'
                }
            }
        }
    }
}