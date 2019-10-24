#
# ElementTree
# $Id: ElementPath.py 1858 2004-06-17 21:31:41Z Fredrik $
#
# limited xpath support for element trees
#
# history:
# 2003-05-23 fl   created
# 2003-05-28 fl   added support for // etc
# 2003-08-27 fl   fixed parsing of periods in element names
#
# Copyright (c) 2003-2004 by Fredrik Lundh.  All rights reserved.
#
# fredrik@pythonware.com
# http://www.pythonware.com
#
# --------------------------------------------------------------------
# The ElementTree toolkit is
#
# Copyright (c) 1999-2004 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------

# Licensed to PSF under a Contributor Agreement.
# See http://www.python.org/2.4/license for licensing details.

##
# Implementation module for XPath support.  There's usually no reason
# to import this module directly; the <b>ElementTree</b> does this for
# you, if needed.
##

import re

xpath_tokenizer = re.compile(
    "(::|\.\.|\(\)|[/.*:\[\]\(\)@=])|((?:\{[^}]+\})?[^/:\[\]\(\)@=\s]+)|\s+"
    ).findall

class xpath_descendant_or_self:
    pass

##
# Wrapper for a compiled XPath.

class Path:

    ##
    # Create an Path instance from an XPath expression.

    def __init__(self, path):
        tokens = xpath_tokenizer(path)
        # the current version supports 'path/path'-style expressions only
        self.path = []
        self.tag = None
        if tokens and tokens[0][0] == "/":
            raise SyntaxError("cannot use absolute path on element")
        while tokens:
            op, tag = tokens.pop(0)
            if tag or op == "*":
                self.path.append(tag or op)
            elif op == ".":
                pass
            elif op == "/":
                self.path.append(xpath_descendant_or_self())
                continue
            else:
                raise SyntaxError("unsupported path syntax (%s)" % op)
            if tokens:
                op, tag = tokens.pop(0)
                if op != "/":
                    raise SyntaxError(
                        "expected path separator (%s)" % (op or tag)
                        )
        if self.path and isinstance(self.path[-1], xpath_descendant_or_self):
            raise SyntaxError("path cannot end with //")
        if len(self.path) == 1 and isinstance(self.path[0], type("")):
            self.tag = self.path[0]

    ##
    # Find first matching object.

    def find(self, element):
        tag = self.tag
        if tag is None:
            nodeset = self.findall(element)
            if not nodeset:
                return None
            return nodeset[0]
        for elem in element:
            if elem.tag == tag:
                return elem
        return None

    ##
    # Find text for first matching object.

    def findtext(self, element, default=None):
        tag = self.tag
        if tag is None:
            nodeset = self.findall(element)
            if not nodeset:
                return default
            return nodeset[0].text or ""
        for elem in element:
            if elem.tag == tag:
                return elem.text or ""
        return default

    ##
    # Find all matching objects.

    def findall(self, element):
        nodeset = [element]
        index = 0
        while 1:
            try:
                path = self.path[index]
                index = index + 1
            except IndexError:
                return nodeset
            set = []
            if isinstance(path, xpath_descendant_or_self):
                try:
                    tag = self.path[index]
                    if not isinstance(tag, type("")):
                        tag = None
                    else:
                        index = index + 1
                except IndexError:
                    tag = None # invalid path
                for node in nodeset:
                    new = list(node.getiterator(tag))
                    if new and new[0] is node:
                        set.extend(new[1:])
                    else:
                        set.extend(new)
            else:
                for node in nodeset:
                    for node in node:
                        if path == "*" or node.tag == path:
                            set.append(node)
            if not set:
                return []
            nodeset = set

_cache = {}

##
# (Internal) Compile path.

def _compile(path):
    p = _cache.get(path)
    if p is not None:
        return p
    p = Path(path)
    if len(_cache) >= 100:
        _cache.clear()
    _cache[path] = p
    return p

##
# Find first matching object.

def find(element, path):
    return _compile(path).find(element)

##
# Find text for first matching object.

def findtext(element, path, default=None):
    return _compile(path).findtext(element, default)

##
# Find all matching objects.

def findall(element, path):
    return _compile(path).findall(element)
