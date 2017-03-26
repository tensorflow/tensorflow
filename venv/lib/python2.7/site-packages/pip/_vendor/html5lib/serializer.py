from __future__ import absolute_import, division, unicode_literals
from pip._vendor.six import text_type

import re

from codecs import register_error, xmlcharrefreplace_errors

from .constants import voidElements, booleanAttributes, spaceCharacters
from .constants import rcdataElements, entities, xmlEntities
from . import treewalkers, _utils
from xml.sax.saxutils import escape

_quoteAttributeSpecChars = "".join(spaceCharacters) + "\"'=<>`"
_quoteAttributeSpec = re.compile("[" + _quoteAttributeSpecChars + "]")
_quoteAttributeLegacy = re.compile("[" + _quoteAttributeSpecChars +
                                   "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n"
                                   "\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15"
                                   "\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
                                   "\x20\x2f\x60\xa0\u1680\u180e\u180f\u2000"
                                   "\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
                                   "\u2008\u2009\u200a\u2028\u2029\u202f\u205f"
                                   "\u3000]")


_encode_entity_map = {}
_is_ucs4 = len("\U0010FFFF") == 1
for k, v in list(entities.items()):
    # skip multi-character entities
    if ((_is_ucs4 and len(v) > 1) or
            (not _is_ucs4 and len(v) > 2)):
        continue
    if v != "&":
        if len(v) == 2:
            v = _utils.surrogatePairToCodepoint(v)
        else:
            v = ord(v)
        if v not in _encode_entity_map or k.islower():
            # prefer &lt; over &LT; and similarly for &amp;, &gt;, etc.
            _encode_entity_map[v] = k


def htmlentityreplace_errors(exc):
    if isinstance(exc, (UnicodeEncodeError, UnicodeTranslateError)):
        res = []
        codepoints = []
        skip = False
        for i, c in enumerate(exc.object[exc.start:exc.end]):
            if skip:
                skip = False
                continue
            index = i + exc.start
            if _utils.isSurrogatePair(exc.object[index:min([exc.end, index + 2])]):
                codepoint = _utils.surrogatePairToCodepoint(exc.object[index:index + 2])
                skip = True
            else:
                codepoint = ord(c)
            codepoints.append(codepoint)
        for cp in codepoints:
            e = _encode_entity_map.get(cp)
            if e:
                res.append("&")
                res.append(e)
                if not e.endswith(";"):
                    res.append(";")
            else:
                res.append("&#x%s;" % (hex(cp)[2:]))
        return ("".join(res), exc.end)
    else:
        return xmlcharrefreplace_errors(exc)

register_error("htmlentityreplace", htmlentityreplace_errors)


def serialize(input, tree="etree", encoding=None, **serializer_opts):
    # XXX: Should we cache this?
    walker = treewalkers.getTreeWalker(tree)
    s = HTMLSerializer(**serializer_opts)
    return s.render(walker(input), encoding)


class HTMLSerializer(object):

    # attribute quoting options
    quote_attr_values = "legacy"  # be secure by default
    quote_char = '"'
    use_best_quote_char = True

    # tag syntax options
    omit_optional_tags = True
    minimize_boolean_attributes = True
    use_trailing_solidus = False
    space_before_trailing_solidus = True

    # escaping options
    escape_lt_in_attrs = False
    escape_rcdata = False
    resolve_entities = True

    # miscellaneous options
    alphabetical_attributes = False
    inject_meta_charset = True
    strip_whitespace = False
    sanitize = False

    options = ("quote_attr_values", "quote_char", "use_best_quote_char",
               "omit_optional_tags", "minimize_boolean_attributes",
               "use_trailing_solidus", "space_before_trailing_solidus",
               "escape_lt_in_attrs", "escape_rcdata", "resolve_entities",
               "alphabetical_attributes", "inject_meta_charset",
               "strip_whitespace", "sanitize")

    def __init__(self, **kwargs):
        """Initialize HTMLSerializer.

        Keyword options (default given first unless specified) include:

        inject_meta_charset=True|False
          Whether it insert a meta element to define the character set of the
          document.
        quote_attr_values="legacy"|"spec"|"always"
          Whether to quote attribute values that don't require quoting
          per legacy browser behaviour, when required by the standard, or always.
        quote_char=u'"'|u"'"
          Use given quote character for attribute quoting. Default is to
          use double quote unless attribute value contains a double quote,
          in which case single quotes are used instead.
        escape_lt_in_attrs=False|True
          Whether to escape < in attribute values.
        escape_rcdata=False|True
          Whether to escape characters that need to be escaped within normal
          elements within rcdata elements such as style.
        resolve_entities=True|False
          Whether to resolve named character entities that appear in the
          source tree. The XML predefined entities &lt; &gt; &amp; &quot; &apos;
          are unaffected by this setting.
        strip_whitespace=False|True
          Whether to remove semantically meaningless whitespace. (This
          compresses all whitespace to a single space except within pre.)
        minimize_boolean_attributes=True|False
          Shortens boolean attributes to give just the attribute value,
          for example <input disabled="disabled"> becomes <input disabled>.
        use_trailing_solidus=False|True
          Includes a close-tag slash at the end of the start tag of void
          elements (empty elements whose end tag is forbidden). E.g. <hr/>.
        space_before_trailing_solidus=True|False
          Places a space immediately before the closing slash in a tag
          using a trailing solidus. E.g. <hr />. Requires use_trailing_solidus.
        sanitize=False|True
          Strip all unsafe or unknown constructs from output.
          See `html5lib user documentation`_
        omit_optional_tags=True|False
          Omit start/end tags that are optional.
        alphabetical_attributes=False|True
          Reorder attributes to be in alphabetical order.

        .. _html5lib user documentation: http://code.google.com/p/html5lib/wiki/UserDocumentation
        """
        unexpected_args = frozenset(kwargs) - frozenset(self.options)
        if len(unexpected_args) > 0:
            raise TypeError("__init__() got an unexpected keyword argument '%s'" % next(iter(unexpected_args)))
        if 'quote_char' in kwargs:
            self.use_best_quote_char = False
        for attr in self.options:
            setattr(self, attr, kwargs.get(attr, getattr(self, attr)))
        self.errors = []
        self.strict = False

    def encode(self, string):
        assert(isinstance(string, text_type))
        if self.encoding:
            return string.encode(self.encoding, "htmlentityreplace")
        else:
            return string

    def encodeStrict(self, string):
        assert(isinstance(string, text_type))
        if self.encoding:
            return string.encode(self.encoding, "strict")
        else:
            return string

    def serialize(self, treewalker, encoding=None):
        # pylint:disable=too-many-nested-blocks
        self.encoding = encoding
        in_cdata = False
        self.errors = []

        if encoding and self.inject_meta_charset:
            from .filters.inject_meta_charset import Filter
            treewalker = Filter(treewalker, encoding)
        # Alphabetical attributes is here under the assumption that none of
        # the later filters add or change order of attributes; it needs to be
        # before the sanitizer so escaped elements come out correctly
        if self.alphabetical_attributes:
            from .filters.alphabeticalattributes import Filter
            treewalker = Filter(treewalker)
        # WhitespaceFilter should be used before OptionalTagFilter
        # for maximum efficiently of this latter filter
        if self.strip_whitespace:
            from .filters.whitespace import Filter
            treewalker = Filter(treewalker)
        if self.sanitize:
            from .filters.sanitizer import Filter
            treewalker = Filter(treewalker)
        if self.omit_optional_tags:
            from .filters.optionaltags import Filter
            treewalker = Filter(treewalker)

        for token in treewalker:
            type = token["type"]
            if type == "Doctype":
                doctype = "<!DOCTYPE %s" % token["name"]

                if token["publicId"]:
                    doctype += ' PUBLIC "%s"' % token["publicId"]
                elif token["systemId"]:
                    doctype += " SYSTEM"
                if token["systemId"]:
                    if token["systemId"].find('"') >= 0:
                        if token["systemId"].find("'") >= 0:
                            self.serializeError("System identifer contains both single and double quote characters")
                        quote_char = "'"
                    else:
                        quote_char = '"'
                    doctype += " %s%s%s" % (quote_char, token["systemId"], quote_char)

                doctype += ">"
                yield self.encodeStrict(doctype)

            elif type in ("Characters", "SpaceCharacters"):
                if type == "SpaceCharacters" or in_cdata:
                    if in_cdata and token["data"].find("</") >= 0:
                        self.serializeError("Unexpected </ in CDATA")
                    yield self.encode(token["data"])
                else:
                    yield self.encode(escape(token["data"]))

            elif type in ("StartTag", "EmptyTag"):
                name = token["name"]
                yield self.encodeStrict("<%s" % name)
                if name in rcdataElements and not self.escape_rcdata:
                    in_cdata = True
                elif in_cdata:
                    self.serializeError("Unexpected child element of a CDATA element")
                for (_, attr_name), attr_value in token["data"].items():
                    # TODO: Add namespace support here
                    k = attr_name
                    v = attr_value
                    yield self.encodeStrict(' ')

                    yield self.encodeStrict(k)
                    if not self.minimize_boolean_attributes or \
                        (k not in booleanAttributes.get(name, tuple()) and
                         k not in booleanAttributes.get("", tuple())):
                        yield self.encodeStrict("=")
                        if self.quote_attr_values == "always" or len(v) == 0:
                            quote_attr = True
                        elif self.quote_attr_values == "spec":
                            quote_attr = _quoteAttributeSpec.search(v) is not None
                        elif self.quote_attr_values == "legacy":
                            quote_attr = _quoteAttributeLegacy.search(v) is not None
                        else:
                            raise ValueError("quote_attr_values must be one of: "
                                             "'always', 'spec', or 'legacy'")
                        v = v.replace("&", "&amp;")
                        if self.escape_lt_in_attrs:
                            v = v.replace("<", "&lt;")
                        if quote_attr:
                            quote_char = self.quote_char
                            if self.use_best_quote_char:
                                if "'" in v and '"' not in v:
                                    quote_char = '"'
                                elif '"' in v and "'" not in v:
                                    quote_char = "'"
                            if quote_char == "'":
                                v = v.replace("'", "&#39;")
                            else:
                                v = v.replace('"', "&quot;")
                            yield self.encodeStrict(quote_char)
                            yield self.encode(v)
                            yield self.encodeStrict(quote_char)
                        else:
                            yield self.encode(v)
                if name in voidElements and self.use_trailing_solidus:
                    if self.space_before_trailing_solidus:
                        yield self.encodeStrict(" /")
                    else:
                        yield self.encodeStrict("/")
                yield self.encode(">")

            elif type == "EndTag":
                name = token["name"]
                if name in rcdataElements:
                    in_cdata = False
                elif in_cdata:
                    self.serializeError("Unexpected child element of a CDATA element")
                yield self.encodeStrict("</%s>" % name)

            elif type == "Comment":
                data = token["data"]
                if data.find("--") >= 0:
                    self.serializeError("Comment contains --")
                yield self.encodeStrict("<!--%s-->" % token["data"])

            elif type == "Entity":
                name = token["name"]
                key = name + ";"
                if key not in entities:
                    self.serializeError("Entity %s not recognized" % name)
                if self.resolve_entities and key not in xmlEntities:
                    data = entities[key]
                else:
                    data = "&%s;" % name
                yield self.encodeStrict(data)

            else:
                self.serializeError(token["data"])

    def render(self, treewalker, encoding=None):
        if encoding:
            return b"".join(list(self.serialize(treewalker, encoding)))
        else:
            return "".join(list(self.serialize(treewalker)))

    def serializeError(self, data="XXX ERROR MESSAGE NEEDED"):
        # XXX The idea is to make data mandatory.
        self.errors.append(data)
        if self.strict:
            raise SerializeError


class SerializeError(Exception):
    """Error in serialized tree"""
    pass
