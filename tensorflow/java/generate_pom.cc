/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <iostream>
#include <string>

#include "tensorflow/c/c_api.h"

int main(int argc, char** argv) {
  std::string tmpl(R"EOF(
<project xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd" xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <modelVersion>4.0.0</modelVersion>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>{{TENSORFLOW_VERSION}}</version>
  <packaging>jar</packaging>

  <name>tensorflow</name>
  <url>https://www.tensorflow.org</url>
  <inceptionYear>2015</inceptionYear>

  <licenses>
    <license>
      <name>The Apache Software License, Version 2.0</name>
      <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
      <distribution>repo</distribution>
    </license>
  </licenses>

  <scm>
    <url>https://github.com/tensorflow/tensorflow.git</url>
    <connection>git@github.com:tensorflow/tensorflow.git</connection>
    <developerConnection>scm:git:https://github.com/tensorflow/tensorflow.git</developerConnection>
  </scm>
</project>
  )EOF");

  const std::string var("{{TENSORFLOW_VERSION}}");
  const std::string val(TF_Version());
  for (size_t pos = tmpl.find(var); pos != std::string::npos;
       pos = tmpl.find(var)) {
    tmpl.replace(pos, var.size(), val);
  }
  std::cout << tmpl;
  return 0;
}
