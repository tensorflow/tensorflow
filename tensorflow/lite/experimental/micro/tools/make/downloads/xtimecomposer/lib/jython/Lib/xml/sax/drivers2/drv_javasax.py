"""
SAX driver for the Java SAX parsers. Can only be used in Jython.

$Id: drv_javasax.py,v 1.5 2003/01/26 09:08:51 loewis Exp $
"""

# --- Initialization

version = "0.10"
revision = "$Revision: 1.5 $"

import string
from xml.sax import xmlreader, saxutils
from xml.sax.handler import feature_namespaces, feature_namespace_prefixes
from xml.sax import _exceptions

# we only work in jython
import sys
if sys.platform[:4] != "java":
    raise _exceptions.SAXReaderNotAvailable("drv_javasax not available in CPython", None)
del sys

# get the necessary Java SAX classes
try:
    from org.python.core import FilelikeInputStream
    from org.xml.sax.helpers import XMLReaderFactory
    from org.xml import sax as javasax
except ImportError:
    raise _exceptions.SAXReaderNotAvailable("SAX is not on the classpath", None)

# get some JAXP stuff
try:
    from javax.xml.parsers import SAXParserFactory, ParserConfigurationException
    factory = SAXParserFactory.newInstance()
    jaxp = 1
except ImportError:
    jaxp = 0

from java.lang import String


def _wrap_sax_exception(e):
    return _exceptions.SAXParseException(e.message,
                                         e.exception,
                                         SimpleLocator(e.columnNumber,
                                                              e.lineNumber,
                                                              e.publicId,
                                                              e.systemId))

class JyErrorHandlerWrapper(javasax.ErrorHandler):
    def __init__(self, err_handler):
        self._err_handler = err_handler

    def error(self, exc):
        self._err_handler.error(_wrap_sax_exception(exc))

    def fatalError(self, exc):
        self._err_handler.fatalError(_wrap_sax_exception(exc))

    def warning(self, exc):
        self._err_handler.warning(_wrap_sax_exception(exc))

class JyInputSourceWrapper(javasax.InputSource):
    def __init__(self, source):
        if isinstance(source, str):
            javasax.InputSource.__init__(self, source)
        elif hasattr(source, "read"):#file like object
            f = source
            javasax.InputSource.__init__(self, FilelikeInputStream(f))
            if hasattr(f, "name"):
                self.setSystemId(f.name)
        else:#xml.sax.xmlreader.InputSource object
            #Use byte stream constructor if possible so that Xerces won't attempt to open
            #the url at systemId unless it's really there
            if source.getByteStream():
                javasax.InputSource.__init__(self,
                                             FilelikeInputStream(source.getByteStream()))
            else:
                javasax.InputSource.__init__(self)
            if source.getSystemId():
                self.setSystemId(source.getSystemId())
            self.setPublicId(source.getPublicId())
            self.setEncoding(source.getEncoding())

class JyEntityResolverWrapper(javasax.EntityResolver):
    def __init__(self, entityResolver):
        self._resolver = entityResolver

    def resolveEntity(self, pubId, sysId):
        return JyInputSourceWrapper(self._resolver.resolveEntity(pubId, sysId))

class JyDTDHandlerWrapper(javasax.DTDHandler):
    def __init__(self, dtdHandler):
        self._handler = dtdHandler

    def notationDecl(self, name, publicId, systemId):
        self._handler.notationDecl(name, publicId, systemId)

    def unparsedEntityDecl(self, name, publicId, systemId, notationName):
        self._handler.unparsedEntityDecl(name, publicId, systemId, notationName)

class SimpleLocator(xmlreader.Locator):
    def __init__(self, colNum, lineNum, pubId, sysId):
        self.colNum = colNum
        self.lineNum = lineNum
        self.pubId = pubId
        self.sysId = sysId

    def getColumnNumber(self):
        return self.colNum

    def getLineNumber(self):
        return self.lineNum

    def getPublicId(self):
        return self.pubId

    def getSystemId(self):
        return self.sysId

# --- JavaSAXParser
class JavaSAXParser(xmlreader.XMLReader, javasax.ContentHandler):
    "SAX driver for the Java SAX parsers."

    def __init__(self, jdriver = None):
        xmlreader.XMLReader.__init__(self)
        self._parser = create_java_parser(jdriver)
        self._parser.setFeature(feature_namespaces, 0)
        self._parser.setFeature(feature_namespace_prefixes, 0)
        self._parser.setContentHandler(self)
        self._nsattrs = AttributesNSImpl()
        self._attrs = AttributesImpl()
        self.setEntityResolver(self.getEntityResolver())
        self.setErrorHandler(self.getErrorHandler())
        self.setDTDHandler(self.getDTDHandler())

    # XMLReader methods

    def parse(self, source):
        "Parse an XML document from a URL or an InputSource."
        self._parser.parse(JyInputSourceWrapper(source))

    def getFeature(self, name):
        return self._parser.getFeature(name)

    def setFeature(self, name, state):
        self._parser.setFeature(name, state)

    def getProperty(self, name):
        return self._parser.getProperty(name)

    def setProperty(self, name, value):
        self._parser.setProperty(name, value)

    def setEntityResolver(self, resolver):
        self._parser.entityResolver = JyEntityResolverWrapper(resolver)
        xmlreader.XMLReader.setEntityResolver(self, resolver)

    def setErrorHandler(self, err_handler):
        self._parser.errorHandler = JyErrorHandlerWrapper(err_handler)
        xmlreader.XMLReader.setErrorHandler(self, err_handler)

    def setDTDHandler(self, dtd_handler):
        self._parser.setDTDHandler(JyDTDHandlerWrapper(dtd_handler))
        xmlreader.XMLReader.setDTDHandler(self, dtd_handler)

    # ContentHandler methods
    def setDocumentLocator(self, locator):
        self._cont_handler.setDocumentLocator(locator)

    def startDocument(self):
        self._cont_handler.startDocument()
        self._namespaces = self._parser.getFeature(feature_namespaces)

    def startElement(self, uri, lname, qname, attrs):
        if self._namespaces:
            self._nsattrs._attrs = attrs
            self._cont_handler.startElementNS((uri or None, lname), qname,
                                              self._nsattrs)
        else:
            self._attrs._attrs = attrs
            self._cont_handler.startElement(qname, self._attrs)

    def startPrefixMapping(self, prefix, uri):
        self._cont_handler.startPrefixMapping(prefix, uri)

    def characters(self, char, start, len):
        self._cont_handler.characters(str(String(char, start, len)))

    def ignorableWhitespace(self, char, start, len):
        self._cont_handler.ignorableWhitespace(str(String(char, start, len)))

    def endElement(self, uri, lname, qname):
        if self._namespaces:
            self._cont_handler.endElementNS((uri or None, lname), qname)
        else:
            self._cont_handler.endElement(qname)

    def endPrefixMapping(self, prefix):
        self._cont_handler.endPrefixMapping(prefix)

    def endDocument(self):
        self._cont_handler.endDocument()

    def processingInstruction(self, target, data):
        self._cont_handler.processingInstruction(target, data)

class AttributesImpl:
    def __init__(self, attrs = None):
        self._attrs = attrs

    def getLength(self):
        return self._attrs.getLength()

    def getType(self, name):
        return self._attrs.getType(name)

    def getValue(self, name):
        value = self._attrs.getValue(name)
        if value == None:
            raise KeyError(name)
        return value

    def getNames(self):
        return [self._attrs.getQName(index) for index in range(len(self))]

    def getQNames(self):
        return [self._attrs.getQName(index) for index in range(len(self))]

    def getValueByQName(self, qname):
        idx = self._attrs.getIndex(qname)
        if idx == -1:
            raise KeyError, qname
        return self._attrs.getValue(idx)

    def getNameByQName(self, qname):
        idx = self._attrs.getIndex(qname)
        if idx == -1:
            raise KeyError, qname
        return qname

    def getQNameByName(self, name):
        idx = self._attrs.getIndex(name)
        if idx == -1:
            raise KeyError, name
        return name

    def __len__(self):
        return self._attrs.getLength()

    def __getitem__(self, name):
        return self.getValue(name)

    def keys(self):
        return self.getNames()

    def copy(self):
        return self.__class__(self._attrs)

    def items(self):
        return [(name, self[name]) for name in self.getNames()]

    def values(self):
        return map(self.getValue, self.getNames())

    def get(self, name, alt=None):
        try:
            return self.getValue(name)
        except KeyError:
            return alt

    def has_key(self, name):
        try:
            self.getValue(name)
            return True
        except KeyError:
            return False

# --- AttributesNSImpl

class AttributesNSImpl(AttributesImpl):

    def __init__(self, attrs=None):
        AttributesImpl.__init__(self, attrs)

    def getType(self, name):
        return self._attrs.getType(name[0], name[1])

    def getValue(self, name):
        value = self._attrs.getValue(name[0], name[1])
        if value == None:
            raise KeyError(name)
        return value

    def getNames(self):
        names = []
        for idx in range(len(self)):
            names.append((self._attrs.getURI(idx),
                          self._attrs.getLocalName(idx)))
        return names

    def getNameByQName(self, qname):
        idx = self._attrs.getIndex(qname)
        if idx == -1:
            raise KeyError, qname
        return (self._attrs.getURI(idx), self._attrs.getLocalName(idx))

    def getQNameByName(self, name):
        idx = self._attrs.getIndex(name[0], name[1])
        if idx == -1:
            raise KeyError, name
        return self._attrs.getQName(idx)

    def getQNames(self):
        return [self._attrs.getQName(idx) for idx in range(len(self))]

# ---

def create_java_parser(jdriver = None):
    try:
        if jdriver:
            return XMLReaderFactory.createXMLReader(jdriver)
        elif jaxp:
            return factory.newSAXParser().getXMLReader()
        else:
            return XMLReaderFactory.createXMLReader()
    except ParserConfigurationException, e:
        raise _exceptions.SAXReaderNotAvailable(e.getMessage())
    except javasax.SAXException, e:
        raise _exceptions.SAXReaderNotAvailable(e.getMessage())

def create_parser(jdriver = None):
    return JavaSAXParser(jdriver)
