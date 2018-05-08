#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Merge TensorFlow Spark connector pom from with deployment template.

The TensorFlow Spark connector pom is here: https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import string
import xml.etree.ElementTree as ET

POM_NAMESPACE = "http://maven.apache.org/POM/4.0.0"
SCALA_VERSION_TAG = "scala.binary.version"


def get_args():
  """Parse command line args."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--version',
    required=True,
    help='Version for the artifact.')
  parser.add_argument(
    '--scala_version',
    required=True,
    choices=['2.10', '2.11'],
    help='Scala version for the artifact.')
  parser.add_argument(
    '--template',
    required=True,
    help='Path to the pom file template.')
  parser.add_argument(
    '--input_pom',
    required=True,
    help='Path to input pom file to merge with template.')
  parser.add_argument(
    '--output_pom',
    required=True,
    help='Path to output pom file.')
  return parser.parse_args()


def load_pom(input_path):
  """ Loads POM file to XML tree"""
  ET.register_namespace("", POM_NAMESPACE)
  tree = ET.parse(input_path)
  return tree


def update_scala_version(tree, version, is_template=False):
  """ Updates scala version in XML tree"""

  if is_template:
    tag = "{%s}artifactId" % POM_NAMESPACE
    nodes = tree.findall(tag)

    if nodes is None:
      raise ValueError("Missing artifactId in template pom")

    for node in nodes:
      template = string.Template(node.text)

      text = template.substitute({"scala_version": version})
      node.text = text
  else:
    # Update scala version property in pom
    tag = "{%s}%s" % (POM_NAMESPACE, SCALA_VERSION_TAG)
    nodes = nodes = list(tree.iter(tag))

    if len(nodes) == 0:
      raise ValueError("Missing %s property in Spark connector pom")

    for node in nodes:
      node.text = version

  return tree


def update_version(tree, version):
  """ Updates version tags in XML tree """
  version_tag = "{%s}version" % POM_NAMESPACE
  nodes = list(tree.iter(version_tag))

  if len(nodes) == 0:
    raise ValueError("Missing version in template pom")

  for node in nodes:
    node.text = version

  return tree


def merge_tags(template_root, pom_root):
  """ Merge pom file from TensorFlow Spark connector with deployment template.

  Modify the TensorFlow Spark connector pom to inherit parent pom and version info and
  other tags provided by deployment template.

  TODO: Figure out if there is a cleaner way of doing this. Inheritance is needed
   for propagating the deployment profile.

  Args:
    template_root: Root XML element for template file.
    pom_root: Root XML element for TensorFlow Spark connector pom file.

  Return:
    template_root: Root XML element with merged tree.
  """
  template_tags = [child.tag for child in template_root]
  template_tags.append("{%s}groupId" % POM_NAMESPACE) # skip groupId since it is inherited from parent

  for child in pom_root:
    if child.tag not in template_tags:
      template_root.append(child)

  return template_root


def main():
  args = get_args()
  template_tree = load_pom(args.template)
  pom_tree = load_pom(args.input_pom)

  template_tree = update_version(template_tree, args.version)
  template_tree = update_scala_version(template_tree, args.scala_version, is_template=True)
  pom_tree = update_scala_version(pom_tree, args.scala_version, is_template=False)
  template_root = merge_tags(template_tree.getroot(), pom_tree.getroot())

  with open(args.output_pom, "w") as f:
    f.write(ET.tostring(template_root))


if __name__ == '__main__':
  sys.exit(main())
