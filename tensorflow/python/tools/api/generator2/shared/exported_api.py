# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Reads and writes files with TF Python exports metadata."""

from collections.abc import Iterable, Sequence
import json
from typing import Any, NamedTuple


class ExportedSymbol(NamedTuple):
  """Information about a single tf_export instance."""

  file_name: str
  line_no: int
  symbol_name: str
  v1_apis: tuple[str, ...]
  v2_apis: tuple[str, ...]

  @classmethod
  def create(
      cls, *, v1_apis: Sequence[str], v2_apis: Sequence[str], **kwargs
  ) -> "ExportedSymbol":
    return cls(v1_apis=tuple(v1_apis), v2_apis=tuple(v2_apis), **kwargs)


class ExportedDoc(NamedTuple):
  """Information about an export Module docstring."""

  file_name: str
  line_no: int
  modules: tuple[str, ...]
  docstring: str

  @classmethod
  def create(cls, *, modules: Sequence[str], **kwargs) -> "ExportedDoc":
    return cls(modules=tuple(modules), **kwargs)


class ExportedApi(object):
  """ExportedApi is a collection of ExportedSymbols."""

  _docs: set[ExportedDoc]
  _symbols: set[ExportedSymbol]

  def __init__(
      self,
      *,
      docs: Iterable[ExportedDoc] = (),
      symbols: Iterable[ExportedSymbol] = (),
  ):
    self._docs = set(docs)
    self._symbols = set(symbols)

  def write(self, filename: str, **kwargs) -> None:
    """Writes exports to filename."""
    with open(filename, mode="w", encoding="utf-8") as f:
      json.dump(
          {
              "docs": [d._asdict() for d in sorted(self.docs)],
              "symbols": [s._asdict() for s in sorted(self.symbols)],
          },
          f,
          **kwargs,
      )

  def read(self, filename: str) -> None:
    """Reads exports from filename."""
    with open(filename, mode="r", encoding="utf-8") as f:
      data = json.load(f)
      self._docs.update(ExportedDoc.create(**d) for d in data["docs"])
      self._symbols.update(ExportedSymbol.create(**s) for s in data["symbols"])

  def add_symbol(self, export: ExportedSymbol) -> None:
    self._symbols.add(export)

  def add_doc(self, export: ExportedDoc) -> None:
    self._docs.add(export)

  @property
  def docs(self) -> Iterable[ExportedDoc]:
    return self._docs

  @property
  def symbols(self) -> Iterable[ExportedSymbol]:
    return self._symbols

  def __str__(self) -> str:
    return json.dumps({
        "docs": [d._asdict() for d in sorted(self.docs)],
        "symbols": [s._asdict() for s in sorted(self.symbols)],
    })

  def __repr__(self) -> str:
    return str(self)

  def __eq__(self, o: Any) -> bool:
    return (
        type(self) is type(o)
        and self.docs == o.docs
        and self.symbols == o.symbols
    )
