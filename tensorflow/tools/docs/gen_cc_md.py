# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convert Doxygen .xml files to MarkDown (.md files)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from BeautifulSoup import BeautifulStoneSoup
import tensorflow as tf


ANCHOR_RE = re.compile(r'\W+')

PAGE_TEMPLATE = '''# `{0} {1}`

{2}

###Member Details

{3}'''

INDEX_TEMPLATE = '''# TensorFlow C++ Session API reference documentation

TensorFlow's public C++ API includes only the API for executing graphs, as of
version 0.5. To control the execution of a graph from C++:

1. Build the computation graph using the [Python API](../python/).
1. Use [`tf.train.write_graph()`](../python/train.md#write_graph) to
write the graph to a file.
1. Load the graph using the C++ Session API. For example:

  ```c++
  // Reads a model graph definition from disk, and creates a session object you
  // can use to run it.
  Status LoadGraph(string graph_file_name, Session** session) {
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(
        ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
    TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
    TF_RETURN_IF_ERROR((*session)->Create(graph_def));
    return Status::OK();
  }
```

1. Run the graph with a call to `session->Run()`

## Env

@@Env
@@RandomAccessFile
@@WritableFile
@@EnvWrapper

## Session

@@Session
@@SessionOptions

## Status

@@Status
@@Status::State

## Tensor

@@Tensor
@@TensorShape
@@TensorShapeDim
@@TensorShapeUtils
@@PartialTensorShape
@@PartialTensorShapeUtils
@@TF_Buffer

## Thread

@@Thread
@@ThreadOptions
'''

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('src_dir', None,
                       'Directory containing the doxygen output.')
tf.flags.DEFINE_string('out_dir', None,
                       'Directory to which docs should be written.')


def member_definition(member_elt):
  def_text = ''

  def_elt = member_elt.find('definition')
  if def_elt:
    def_text = def_elt.text

  return def_text


def member_sig(member_elt):
  def_text = member_definition(member_elt)

  argstring_text = ''
  argstring = member_elt.find('argsstring')
  if argstring:
    argstring_text = argstring.text

  sig = def_text + argstring_text
  return sig


def anchorize(name):
  return ANCHOR_RE.sub('_', name)


def element_text(member_elt, elt_name):
  """Extract all `para` text from (`elt_name` in) `member_elt`."""
  text = []
  if elt_name:
    elt = member_elt.find(elt_name)
  else:
    elt = member_elt

  if elt:
    paras = elt.findAll('para')
    for p in paras:
      text.append(p.getText(separator=u' ').strip())
  return '\n\n'.join(text)


def full_member_entry(member_elt):
  """Generate the description of `member_elt` for "Member Details"."""
  anchor = '{#' + anchorize(member_definition(member_elt)) + '}'
  full_entry = '#### `%s` %s' % (member_sig(member_elt), anchor)

  complete_descr = element_text(member_elt, 'briefdescription') + '\n\n'
  complete_descr += element_text(member_elt, 'detaileddescription')

  if complete_descr:
    full_entry += '\n\n' + complete_descr

  return full_entry


def brief_member_entry(member_elt):
  """Generate the description of `member_elt` for the "Member Summary"."""
  brief_item = ''
  brief_descr = element_text(member_elt, 'briefdescription')
  if brief_descr:
    brief_item = '\n  * ' + brief_descr
  sig = member_sig(member_elt)
  memdef = member_definition(member_elt)
  linkified_sig = '[`{0}`](#{1})'.format(sig, anchorize(memdef))

  return '* ' + linkified_sig + brief_item


def all_briefs(members):
  briefs = [brief_member_entry(member_elt) for member_elt in members]
  return '\n'.join(briefs)


def all_fulls(members):
  fulls = [full_member_entry(member_elt) for member_elt in members]
  return '\n\n'.join(fulls)


def page_overview(class_elt):
  """Returns the contents of the .md file for `class_elt`."""
  overview_brief = ''
  overview_details = ''

  briefs = class_elt.findAll('briefdescription', recursive=False)
  if briefs:
    overview_brief = element_text(briefs[0], None)

  details = class_elt.findAll('detaileddescription', recursive=False)
  if details:
    overview_details = element_text(details[0], None)

  return overview_brief + '\n\n' + overview_details


def page_with_name(pages, name):
  def match(n):
    for i in xrange(len(pages)):
      if pages[i].get_name() == n:
        return i
    return None
  return match(name) or match('tensorflow::' + name)


def get_all_indexed_pages():
  all_pages = set()
  lines = INDEX_TEMPLATE.split('\n')
  for i in range(len(lines)):
    if lines[i].startswith('@@'):
      name = lines[i][2:]
      all_pages.add(name)
  return all_pages


def index_page(pages):
  """Create the index page linking to `pages` using INDEX_TEMPLATE."""
  pages = pages[:]
  lines = INDEX_TEMPLATE.split('\n')
  all_md_files = []
  for i in range(len(lines)):
    if lines[i].startswith('@@'):
      name = lines[i][2:]
      page_index = page_with_name(pages, name)
      if page_index is None:
        raise ValueError('Missing page with name: ' + name)
      lines[i] = '* [{0}]({1})'.format(
          pages[page_index].get_name(), pages[page_index].get_md_filename())
      all_md_files.append(pages[page_index].get_md_filename())
      pages.pop(page_index)

  return '\n'.join(lines)


def page_in_name_list(page, names):
  for name in names:
    if page.get_name() == name or page.get_name() == 'tensorflow::' + name:
      return True
  return False


class Page(object):
  """Holds the MarkDown converted contents of a .xml page."""

  def __init__(self, xml_path, deftype):
    self.type = deftype
    xml_file = open(xml_path)
    xml = xml_file.read()
    xml = xml.replace('<computeroutput>', '`').replace('</computeroutput>', '`')
    # TODO(josh11b): Should not use HTML entities inside ```...```.
    soup = BeautifulStoneSoup(
        xml, convertEntities=BeautifulStoneSoup.HTML_ENTITIES)
    self.name = soup.find('compoundname').text
    print('Making page with name ' + self.name + ' (from ' + xml_path + ')')
    members = soup('memberdef', prot='public')
    fulls = all_fulls(members)
    self.overview = page_overview(soup.find('compounddef'))
    self.page_text = PAGE_TEMPLATE.format(
        self.type, self.name, self.overview, fulls)

  def get_text(self):
    return self.page_text

  def get_name(self):
    return self.name

  def get_short_name(self):
    parse = self.get_name().split('::')
    return parse[len(parse)-1]

  def get_type(self):
    return self.type

  def get_md_filename(self):
    capitalized_type = self.get_type()[0].upper() + self.get_type()[1:]
    return capitalized_type + anchorize(self.get_short_name()) + '.md'


def main(unused_argv):
  print('Converting in ' + FLAGS.src_dir)
  pages = []
  all_pages = get_all_indexed_pages()
  xml_files = os.listdir(FLAGS.src_dir)
  for fname in xml_files:
    if len(fname) < 6: continue
    newpage = None
    if fname[0:5] == 'class':
      newpage = Page(os.path.join(FLAGS.src_dir, fname), 'class')
    elif fname[0:6] == 'struct':
      newpage = Page(os.path.join(FLAGS.src_dir, fname), 'struct')
    if newpage is not None and page_in_name_list(newpage, all_pages):
      pages.append(newpage)
      md_filename = newpage.get_md_filename()
      print('Writing ' + md_filename)
      md_file = open(os.path.join(FLAGS.out_dir, md_filename), 'w')
      print(newpage.get_text(), file=md_file)

  index_text = index_page(pages)
  index_md_file = open(os.path.join(FLAGS.out_dir, 'index.md'), 'w')
  print(index_text, file=index_md_file)
  return 0

if __name__ == '__main__':
  tf.app.run()
