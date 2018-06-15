# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Converter construction support.

This module contains a base class for all converters, as well as supporting
structures. These structures are referred to as contexts.

The class hierarchy is as follows:

    <your converter>
      [extends] converter.Base
        [extends] transformer.Base
            [extends] gast.nodeTransformer
          [uses] transfomer.SourceInfo
        [uses] converter.EntityContext
          [uses] converter.ProgramContext
          [uses] transfomer.SourceInfo

converter.Base is a specialization of transformer.Base for AutoGraph. It's a
very lightweight subclass that adds a `ctx` attribute holding the corresponding
EntityContext object (see below). Note that converters are not reusable, and
`visit` will raise an error if called more than once.

converter.EntityContext contains mutable state associated with an entity that
the converter processes.

converter.ProgramContext contains mutable state across related entities. For
example, when converting several functions that call one another, the
ProgramContext should be shared across these entities.

Below is the overal flow at conversion:

    program_ctx = ProgramContext(<entities to convert>, <global settings>, ...)
    while <program_ctx has more entities to convert>:
      entity, source_info = <get next entity from program_ctx>
      entity_ctx = EntityContext(program_ctx, source_info)
      for <each ConverterClass>:
        converter = ConverterClass(entity_ctx)

        # May update entity_ctx and program_ctx
        entity = converter.visit(entity)

      <add entity's dependencies to program_ctx>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.autograph.core import config
from tensorflow.contrib.autograph.core import naming
from tensorflow.contrib.autograph.pyct import transformer

# TODO(mdan): These contexts can be refactored into first class objects.
# For example, we could define Program and Entity abstractions that hold on
# to the actual entity and have conversion methods.


class ProgramContext(object):
  """ProgramContext keeps track of converting function hierarchies.

  This object is mutable, and is updated during conversion. Not thread safe.

  Attributes:
    recursive: bool, whether to recursively convert any functions that the
        decorator function may call.
    autograph_decorators: Tuple[Callable, ...], decorator functions that belong
        to AutoGraph. These require special treatment.
    dependency_cache: Dict[Any, ast.AST], the original entities mapped to their
        converted AST
    additional_imports: Set[Any], additional entities which for any reason
        cannot be attached after loading and need to be explicitly imported
        in the generated code
    name_map: Dict[str, str], map of original entity name to the name of
        their converted counterparts
    ag_module: Module, a reference to the autograph module. This
        needs to be specified by the caller to avoid circular dependencies.
    uncompiled_modules: Set[Tuple[str, ...]], with each tuple representing the
        fully qualified name of a package containing functions that will not be
        compiled.
    required_imports: str, containing an import statement on each line. These
        are all the imports necessary for the compiled code to run, in addition
        to the closures of each entity, which are attached dynamically.
  """

  # TODO(mdan): Rename ag_module to autograph_module?
  def __init__(
      self,
      recursive,
      autograph_decorators,
      partial_types,
      ag_module,
      uncompiled_modules,
  ):
    self.recursive = recursive
    self.autograph_decorators = autograph_decorators
    self.partial_types = partial_types if partial_types else ()
    self.ag_module = ag_module
    self.uncompiled_modules = uncompiled_modules

    # Required to output dependencies in discovery order, which should match
    # the reverse dependency order.
    self.dependency_cache = collections.OrderedDict()
    self.additional_imports = set()
    self.name_map = {}

  @property
  def required_imports(self):
    """Returns a block containing all imports required by the converted code."""
    # TODO(mdan): Check that these don't clobber one another.
    return '\n'.join(config.COMPILED_IMPORT_STATEMENTS +
                     tuple(self.additional_imports))

  def new_namer(self, namespace):
    return naming.Namer(namespace, self.recursive, self.name_map,
                        self.partial_types)

  def update_name_map(self, namer):
    """Updates renamed_calls based on the recent activity from the namer.

    Whenever we convert a new entity, any references to other entities are being
    renamed to match their soon-to-be-converted counterparts. The namer keeps
    track of these renames. When conversion is complete, we copy those renames
    so that when those referenced entities are being converted, their new name
    matches.

    Args:
      namer: naming.Namer

    Raises:
      ValueError: when an entity was renamed twice and to different names.
    """
    # TODO(mdan): Have call_trees do this directly.
    # This is done so indirectly, via the namer, for historic reasons. But
    # now we can have the converter that does the rename record the new name
    # as well and skip this step altogether.
    for o, name in namer.renamed_calls.items():
      if o in self.name_map:
        if self.name_map[o] != name:
          raise ValueError(
              'Calls to %s were converted using multiple names (%s). This is '
              'possible when an entity with one of these names already '
              'existed. To fix, avoid using any of these names.' %
              (o, (name, self.name_map[o])))
      else:
        self.name_map[o] = name

  def add_to_cache(self, original_entity, converted_ast):
    self.dependency_cache[original_entity] = converted_ast


class EntityContext(object):
  """Tracks the conversion of a single entity.

  This object is mutable, and is updated during conversion. Not thread safe.

  Attributes:
    namer: Namer
    info: transformer.EntityInfo
    program: ProgramContext
  """

  def __init__(self, namer, entity_info, program_ctx):
    self.namer = namer
    self.info = entity_info
    self.program = program_ctx


class Base(transformer.Base):
  """All converters should inherit from this class.

  Attributes:
    ctx: EntityContext
  """

  def __init__(self, ctx):
    super(Base, self).__init__(ctx.info)
    self._used = False
    self.ctx = ctx  # Keeping this short because it's used frequently.

  def visit(self, node):
    if self._used:
      raise ValueError('visit may only be called once')
    self._used = True
    super(Base, self).visit(node)
