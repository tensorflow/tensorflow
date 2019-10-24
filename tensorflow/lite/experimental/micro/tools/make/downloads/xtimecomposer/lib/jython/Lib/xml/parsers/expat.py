# coding: utf-8

#------------------------------------------------------------------------------
# Copyright (c) 2008 Sébastien Boisgérault
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

__all__ = ["ExpatError", "ParserCreate", "XMLParserType", "error", "errors"]

# Jython check
import sys
if not sys.platform.startswith('java'):
    raise ImportError("this version of expat requires the jython interpreter")

# Standard Python Library
import re
import types

# Jython
from org.python.core import Py
from org.python.core.util import StringUtil
from jarray import array

# Java Standard Edition
from java.io import ByteArrayInputStream
from java.lang import String, StringBuilder
from org.xml.sax import InputSource
from org.xml.sax import SAXNotRecognizedException, SAXParseException
from org.xml.sax.helpers import XMLReaderFactory
from org.xml.sax.ext import DefaultHandler2

# Xerces
try:
    # Name mangled by jarjar?
    import org.python.apache.xerces.parsers.SAXParser
    _xerces_parser = "org.python.apache.xerces.parsers.SAXParser"
except ImportError:
    _xerces_parser = "org.apache.xerces.parsers.SAXParser"


# @expat args registry
_register = {}


def ParserCreate(encoding=None, namespace_separator=None):
    return XMLParser(encoding, namespace_separator)


class XMLParser(object):

    def __init__(self, encoding, namespace_separator):
        self.encoding = encoding
        self.CurrentLineNumber = 1
        self.CurrentColumnNumber = 0
        self._NextLineNumber = 1
        self._NextColumnNumber = 0
        self.ErrorLineNumber = -1
        self.ErrorColumnNumber = -1
        self.ErrorCode = None

        if namespace_separator is None:
            self.namespace_separator = namespace_separator
        elif isinstance(namespace_separator, basestring):
            self.namespace_separator = str(namespace_separator)
            if len(self.namespace_separator) > 1:
                error = ("namespace_separator must be at most one character, "
                         "omitted, or None")
                raise ValueError(error)
        else:
            error = ("ParserCreate() argument 2 must be string or None, "
                     "not %s" % type(namespace_separator).__name__)
            raise TypeError(error)

        self._reader = XMLReaderFactory.createXMLReader(_xerces_parser)

        if self.namespace_separator is None:
            try:
                feature = "http://xml.org/sax/features/namespaces"
                self._reader.setFeature(feature, False)
            except SAXNotRecognizedException:
                error = ("namespace support cannot be disabled; "
                         "set namespace_separator to a string of length 1.")
                raise ValueError(error)

        self._base = None
        self._buffer_text = True
        self._returns_unicode = True

        self._data = StringBuilder()

        self._handler = XMLEventHandler(self)
        self._reader.setContentHandler(self._handler)
        self._reader.setErrorHandler(self._handler)
        self._reader.setDTDHandler(self._handler)
        self._reader.setEntityResolver(self._handler)

        sax_properties = ("lexical-handler", "declaration-handler")
        for name in sax_properties:
            try:
                name = "http://xml.org/sax/properties/" + name
                self._reader.setProperty(name, self._handler)
            except SAXNotRecognizedException:
                error = "can't set property %r" % name
                raise NotImplementedError(error)

        apache_features = (("nonvalidating/load-external-dtd", False),)
        for name, value in apache_features:
            try:
                name = "http://apache.org/xml/features/" + name
                self._reader.setFeature(name, value)
            except SAXNotRecognizedException:
                error = "can't set feature %r" % name
                raise NotImplementedError(error)

        # experimental
        #f = "http://xml.org/sax/features/external-general-entities"
        f = "http://xml.org/sax/features/external-parameter-entities"
        #self._reader.setFeature(f, False)

        # check
        f = "http://xml.org/sax/features/use-entity-resolver2"
        assert self._reader.getFeature(f)

    def GetBase(self):
        return self._base

    def SetBase(self, base):
        self._base = base

    def _error(self, value=None):
        raise AttributeError("'XMLParser' has no such attribute")

    def _get_buffer_text(self):
        return self._buffer_text

    def _set_buffer_text(self, value):
        self._buffer_text = bool(value)

    def _get_returns_unicode(self):
        return bool(self._returns_unicode)

    def _set_returns_unicode(self, value):
        self._returns_unicode = value

    # 'ordered' and 'specified' attributes are not supported
    ordered_attributes = property(_error, _error)
    specified_attributes = property(_error, _error)
    # any setting is allowed, but it won't make a difference
    buffer_text = property(_get_buffer_text, _set_buffer_text)
    # non-significant read-only values
    buffer_used = property(lambda self: None)
    buffer_size = property(lambda self: None)
    # 'returns_unicode' attribute is properly supported
    returns_unicode = property(_get_returns_unicode, _set_returns_unicode)

    def _expat_error(self, sax_error):
        sax_message = sax_error.getMessage()
        pattern = 'The entity ".*" was referenced, but not declared\.'
        if re.match(pattern, sax_message):
            expat_message = "undefined entity: line %s, column %s" % \
                            (self.ErrorLineNumber, self.ErrorColumnNumber)
        else:
            expat_message = sax_message
        error = ExpatError(expat_message)
        error.lineno = self.ErrorLineNumber
        error.offset = self.ErrorColumnNumber
        error.code = self.ErrorCode
        return error

    def Parse(self, data, isfinal=False):
        # The 'data' argument should be an encoded text: a str instance that
        # represents an array of bytes. If instead it is a unicode string,
        # only the us-ascii range is considered safe enough to be silently
        # converted.
        if isinstance(data, unicode):
            data = data.encode(sys.getdefaultencoding())

        self._data.append(data)

        if isfinal:
            bytes = StringUtil.toBytes(self._data.toString())
            byte_stream = ByteArrayInputStream(bytes)
            source = InputSource(byte_stream)
            if self.encoding is not None:
                source.setEncoding(self.encoding)
            try:
                self._reader.parse(source)
            except SAXParseException, sax_error:
                # Experiments tend to show that the '_Next*' parser locations
                # match more closely expat behavior than the 'Current*' or sax
                # error locations.
                self.ErrorLineNumber = self._NextLineNumber
                self.ErrorColumnNumber = self._NextColumnNumber
                self.ErrorCode = None
                raise self._expat_error(sax_error)
            return 1

    def ParseFile(self, file):
        # TODO: pseudo-buffering if a read without argument is not supported.
        #       document parse / parsefile usage.
        return self.Parse(file.read(), isfinal=True)


XMLParserType = XMLParser


def _encode(arg, encoding):
    if isinstance(arg, unicode):
        return arg.encode(encoding)
    else:
        if isinstance(arg, dict):
            iterator = arg.iteritems()
        else:
            iterator = iter(arg)
        return type(arg)(_encode(_arg, encoding) for _arg in iterator)


def expat(callback=None, guard=True, force=False, returns=None):
    def _expat(method):
        name = method.__name__
        context = id(sys._getframe(1))
        key = name, context
        append = _register.setdefault(key, []).append
        append((method, callback, guard, force, returns))

        def new_method(*args):
            self = args[0]
            parser = self.parser
            self._update_location(event=name) # bug if multiple method def
            for (method, callback, guard, force, returns) in _register[key]:
                if guard not in (True, False):
                    guard = getattr(self, guard)
                _callback = callback and guard and \
                            getattr(parser, callback, None)
                if _callback or force:
                    results = method(*args)
                    if _callback:
                        if not isinstance(results, tuple):
                            results = (results,)
                        if not parser.returns_unicode:
                            results = _encode(results, "utf-8")
                        _callback(*results)
                    return returns

        new_method.__name__ = name
        #new_method.__doc__ = method.__doc__ # what to do with multiple docs ?
        return new_method
    return _expat


class XMLEventHandler(DefaultHandler2):

    def __init__(self, parser):
        self.parser = parser
        self._tags = {}
        self.not_in_dtd = True
        self._entity = {}
        self._previous_event = None

    # --- Helpers -------------------------------------------------------------

    def _intern(self, tag):
        return self._tags.setdefault(tag, tag)

    def _qualify(self, local_name, qname, namespace=None):
        namespace_separator = self.parser.namespace_separator
        if namespace_separator is None:
            return qname
        if not namespace:
            return local_name
        else:
            return namespace + namespace_separator + local_name

    def _char_slice_to_unicode(self, characters, start, length):
        """Convert a char[] slice to a PyUnicode instance"""
        text = Py.newUnicode(String(characters[start:start + length]))
        return text

    def _expat_content_model(self, name, model_):
        # TODO : implement a model parser
        return (name, model_) # does not fit expat conventions

    def _update_location(self, event=None):
        parser = self.parser
        locator = self._locator

        # ugly hack that takes care of a xerces-specific (?) locator issue:
        # locate start and end elements at the '<' instead of the first tag
        # type character.
        if event == "startElement" and self._previous_event == "characters":
            parser._NextColumnNumber = max(parser._NextColumnNumber - 1, 0)
        if event == "endElement" and self._previous_event == "characters":
            parser._NextColumnNumber = max(parser._NextColumnNumber - 2, 0)
        # TODO: use the same trick to report accurate error locations ?

        parser.CurrentLineNumber = parser._NextLineNumber
        parser.CurrentColumnNumber = parser._NextColumnNumber
        parser._NextLineNumber = locator.getLineNumber()
        parser._NextColumnNumber = locator.getColumnNumber() - 1

        self._previous_event = event

    # --- ContentHandler Interface --------------------------------------------

    @expat("ProcessingInstructionHandler")
    def processingInstruction(self, target, data):
        return target, data

    @expat("StartElementHandler")
    def startElement(self, namespace, local_name, qname, attributes):
        tag = self._qualify(local_name, qname, namespace)
        attribs = {}
        length = attributes.getLength()
        for index in range(length):
            local_name = attributes.getLocalName(index)
            qname = attributes.getQName(index)
            namespace = attributes.getURI(index)
            name = self._qualify(local_name, qname, namespace)
            value = attributes.getValue(index)
            attribs[name] = value
        return self._intern(tag), attribs

    @expat("EndElementHandler")
    def endElement(self, namespace, local_name, qname):
        return self._intern(self._qualify(local_name, qname, namespace))

    @expat("CharacterDataHandler")
    def characters(self, characters, start, length):
        return self._char_slice_to_unicode(characters, start, length)

    @expat("DefaultHandlerExpand")
    def characters(self, characters, start, length):
        return self._char_slice_to_unicode(characters, start, length)

    @expat("DefaultHandler")
    def characters(self, characters, start, length):
        # TODO: make a helper function here
        if self._entity["location"] == (self.parser.CurrentLineNumber,
                                        self.parser.CurrentColumnNumber):
            return "&%s;" % self._entity["name"]
        else:
            return self._char_slice_to_unicode(characters, start, length)

    @expat("StartNamespaceDeclHandler")
    def startPrefixMapping(self, prefix, uri):
        return prefix, uri

    @expat("EndNamespaceDeclHandler")
    def endPrefixMapping(self, prefix):
        return prefix

    empty_source = InputSource(ByteArrayInputStream(array([], "b")))

    @expat("ExternalEntityRefHandler", guard="not_in_dtd",
                                       returns=empty_source)
    def resolveEntity(self, name, publicId, baseURI, systemId):
        context = name # wrong. see expat headers documentation.
        base = self.parser.GetBase()
        return context, base, systemId, publicId

    @expat("DefaultHandlerExpand", guard="not_in_dtd",
                                   returns=empty_source)
    def resolveEntity(self, name, publicId, baseURI, systemId):
        return "&%s;" % name

    @expat("DefaultHandler", guard="not_in_dtd",
                             returns=empty_source)
    def resolveEntity(self, name, publicId, baseURI, systemId):
        return "&%s;" % name

    @expat(force=True, returns=empty_source)
    def resolveEntity(self, name, publicId, baseURI, systemId):
        pass

    def setDocumentLocator(self, locator):
        self._locator = locator

    def skippedEntity(self, name):
        error = ExpatError()
        error.lineno = self.ErrorLineNumber = self.parser._NextLineNumber
        error.offset = self.ErrorColumnNumber = self.parser._NextColumnNumber
        error.code = self.ErrorCode = None
        message = "undefined entity &%s;: line %s, column %s"
        message = message % (name, error.lineno, error.offset)
        error.__init__(message)
        raise error

    # --- LexicalHandler Interface --------------------------------------------

    @expat("CommentHandler")
    def comment(self, characters, start, length):
        return self._char_slice_to_unicode(characters, start, length)

    @expat("StartCdataSectionHandler")
    def startCDATA(self):
        return ()

    @expat("EndCdataSectionHandler")
    def endCDATA(self):
        return ()

    @expat("StartDoctypeDeclHandler", force=True)
    def startDTD(self, name, publicId, systemId):
        self.not_in_dtd = False
        has_internal_subset = 0 # don't know this ...
        return name, systemId, publicId, has_internal_subset

    @expat("EndDoctypeDeclHandler", force=True)
    def endDTD(self):
        self.not_in_dtd = True

    def startEntity(self, name):
        self._entity = {}
        self._entity["location"] = (self.parser._NextLineNumber,
                                    self.parser._NextColumnNumber)
        self._entity["name"] = name

    def endEntity(self, name):
        pass

    # --- DTDHandler Interface ------------------------------------------------

    @expat("NotationDeclHandler")
    def notationDecl(self, name, publicId, systemId):
        base = self.parser.GetBase()
        return name, base, systemId, publicId

    @expat("UnparsedEntityDeclHandler") # deprecated
    def unparsedEntityDecl(self, name, publicId, systemId, notationName):
        base = self.parser.GetBase()
        return name, base, systemId, publicId, notationName

    # --- DeclHandler Interface -----------------------------------------------

    @expat("AttlistDeclHandler")
    def attributeDecl(self, eName, aName, type, mode, value):
        # TODO: adapt mode, required, etc.
        required = False
        return eName, aName, type, value, required

    @expat("ElementDeclHandler")
    def elementDecl(self, name, model):
        return self._expat_content_model(name, model)

    @expat("EntityDeclHandler")
    def externalEntityDecl(self, name, publicId, systemId):
        base = self.parser.GetBase()
        value = None
        is_parameter_entity = None
        notation_name = None
        return (name, is_parameter_entity, value, base, systemId, publicId,
                notation_name)

    @expat("EntityDeclHandler")
    def internalEntityDecl(self, name, value):
        base = self.parser.GetBase()
        is_parameter_entity = None
        notation_name = None
        systemId, publicId = None, None
        return (name, is_parameter_entity, value, base, systemId, publicId,
                notation_name)


def _init_model():
    global model
    model = types.ModuleType("pyexpat.model")
    model.__doc__ = "Constants used to interpret content model information."
    quantifiers = "NONE, OPT, REP, PLUS"
    for i, quantifier in enumerate(quantifiers.split(", ")):
        setattr(model, "XML_CQUANT_" + quantifier, i)
    types_ = "EMPTY, ANY, MIXED, NAME, CHOICE, SEQ"
    for i, type_ in enumerate(types_.split(", ")):
        setattr(model, "XML_CTYPE_" + type_, i+1)

_init_model()
del _init_model


class ExpatError(Exception):
    pass


error = ExpatError


def _init_error_strings():
    global ErrorString
    error_strings = (
        None,
        "out of memory",
        "syntax error",
        "no element found",
        "not well-formed (invalid token)",
        "unclosed token",
        "partial character",
        "mismatched tag",
        "duplicate attribute",
        "junk after document element",
        "illegal parameter entity reference",
        "undefined entity",
        "recursive entity reference",
        "asynchronous entity",
        "reference to invalid character number",
        "reference to binary entity",
        "reference to external entity in attribute",
        "XML or text declaration not at start of entity",
        "unknown encoding",
        "encoding specified in XML declaration is incorrect",
        "unclosed CDATA section",
        "error in processing external entity reference",
        "document is not standalone",
        "unexpected parser state - please send a bug report",
        "entity declared in parameter entity",
        "requested feature requires XML_DTD support in Expat",
        "cannot change setting once parsing has begun",
        "unbound prefix",
        "must not undeclare prefix",
        "incomplete markup in parameter entity",
        "XML declaration not well-formed",
        "text declaration not well-formed",
        "illegal character(s) in public id",
        "parser suspended",
        "parser not suspended",
        "parsing aborted",
        "parsing finished",
        "cannot suspend in external parameter entity")
    def ErrorString(code):
        try:
            return error_strings[code]
        except IndexError:
            return None

_init_error_strings()
del _init_error_strings


def _init_errors():
    global errors

    errors = types.ModuleType("pyexpat.errors")
    errors.__doc__ = "Constants used to describe error conditions."

    error_names = """
    XML_ERROR_NONE
    XML_ERROR_NONE,
    XML_ERROR_NO_MEMORY,
    XML_ERROR_SYNTAX,
    XML_ERROR_NO_ELEMENTS,
    XML_ERROR_INVALID_TOKEN,
    XML_ERROR_UNCLOSED_TOKEN,
    XML_ERROR_PARTIAL_CHAR,
    XML_ERROR_TAG_MISMATCH,
    XML_ERROR_DUPLICATE_ATTRIBUTE,
    XML_ERROR_JUNK_AFTER_DOC_ELEMENT,
    XML_ERROR_PARAM_ENTITY_REF,
    XML_ERROR_UNDEFINED_ENTITY,
    XML_ERROR_RECURSIVE_ENTITY_REF,
    XML_ERROR_ASYNC_ENTITY,
    XML_ERROR_BAD_CHAR_REF,
    XML_ERROR_BINARY_ENTITY_REF,
    XML_ERROR_ATTRIBUTE_EXTERNAL_ENTITY_REF,
    XML_ERROR_MISPLACED_XML_PI,
    XML_ERROR_UNKNOWN_ENCODING,
    XML_ERROR_INCORRECT_ENCODING,
    XML_ERROR_UNCLOSED_CDATA_SECTION,
    XML_ERROR_EXTERNAL_ENTITY_HANDLING,
    XML_ERROR_NOT_STANDALONE,
    XML_ERROR_UNEXPECTED_STATE,
    XML_ERROR_ENTITY_DECLARED_IN_PE,
    XML_ERROR_FEATURE_REQUIRES_XML_DTD,
    XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING,
    XML_ERROR_UNBOUND_PREFIX,
    XML_ERROR_UNDECLARING_PREFIX,
    XML_ERROR_INCOMPLETE_PE,
    XML_ERROR_XML_DECL,
    XML_ERROR_TEXT_DECL,
    XML_ERROR_PUBLICID,
    XML_ERROR_SUSPENDED,
    XML_ERROR_NOT_SUSPENDED,
    XML_ERROR_ABORTED,
    XML_ERROR_FINISHED,
    XML_ERROR_SUSPEND_PE
    """
    error_names = [name.strip() for name in error_names.split(',')]
    for i, name in enumerate(error_names[1:]):
        setattr(errors, name, ErrorString(i+1))

_init_errors()
del _init_errors
