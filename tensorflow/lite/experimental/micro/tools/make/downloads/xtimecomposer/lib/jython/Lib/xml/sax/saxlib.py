"""
This module contains the core classes of version 2.0 of SAX for Python.
This file provides only default classes with absolutely minimum
functionality, from which drivers and applications can be subclassed.

Many of these classes are empty and are included only as documentation
of the interfaces.

$Id: saxlib.py,v 1.12 2002/05/10 14:49:21 akuchling Exp $
"""

version = '2.0beta'

# A number of interfaces used to live in saxlib, but are now in
# various other modules for Python 2 compatibility. If nobody uses
# them here any longer, the references can be removed

from handler import ErrorHandler, ContentHandler, DTDHandler, EntityResolver
from xmlreader import XMLReader, InputSource, Locator, IncrementalParser
from _exceptions import *

from handler import \
     feature_namespaces,\
     feature_namespace_prefixes,\
     feature_string_interning,\
     feature_validation,\
     feature_external_ges,\
     feature_external_pes,\
     all_features,\
     property_lexical_handler,\
     property_declaration_handler,\
     property_dom_node,\
     property_xml_string,\
     all_properties

#============================================================================
#
# MAIN INTERFACES
#
#============================================================================

# ===== XMLFILTER =====

class XMLFilter(XMLReader):
    """Interface for a SAX2 parser filter.

    A parser filter is an XMLReader that gets its events from another
    XMLReader (which may in turn also be a filter) rather than from a
    primary source like a document or other non-SAX data source.
    Filters can modify a stream of events before passing it on to its
    handlers."""

    def __init__(self, parent = None):
        """Creates a filter instance, allowing applications to set the
        parent on instantiation."""
        XMLReader.__init__(self)
        self._parent = parent

    def setParent(self, parent):
        """Sets the parent XMLReader of this filter. The argument may
        not be None."""
        self._parent = parent

    def getParent(self):
        "Returns the parent of this filter."
        return self._parent

# ===== ATTRIBUTES =====

class Attributes:
    """Interface for a list of XML attributes.

    Contains a list of XML attributes, accessible by name."""

    def getLength(self):
        "Returns the number of attributes in the list."
        raise NotImplementedError("This method must be implemented!")

    def getType(self, name):
        "Returns the type of the attribute with the given name."
        raise NotImplementedError("This method must be implemented!")

    def getValue(self, name):
        "Returns the value of the attribute with the given name."
        raise NotImplementedError("This method must be implemented!")

    def getValueByQName(self, name):
        """Returns the value of the attribute with the given raw (or
        qualified) name."""
        raise NotImplementedError("This method must be implemented!")

    def getNameByQName(self, name):
        """Returns the namespace name of the attribute with the given
        raw (or qualified) name."""
        raise NotImplementedError("This method must be implemented!")

    def getNames(self):
        """Returns a list of the names of all attributes
        in the list."""
        raise NotImplementedError("This method must be implemented!")

    def getQNames(self):
        """Returns a list of the raw qualified names of all attributes
        in the list."""
        raise NotImplementedError("This method must be implemented!")

    def __len__(self):
        "Alias for getLength."
        raise NotImplementedError("This method must be implemented!")

    def __getitem__(self, name):
        "Alias for getValue."
        raise NotImplementedError("This method must be implemented!")

    def keys(self):
        "Returns a list of the attribute names in the list."
        raise NotImplementedError("This method must be implemented!")

    def has_key(self, name):
        "True if the attribute is in the list, false otherwise."
        raise NotImplementedError("This method must be implemented!")

    def get(self, name, alternative=None):
        """Return the value associated with attribute name; if it is not
        available, then return the alternative."""
        raise NotImplementedError("This method must be implemented!")

    def copy(self):
        "Return a copy of the Attributes object."
        raise NotImplementedError("This method must be implemented!")

    def items(self):
        "Return a list of (attribute_name, value) pairs."
        raise NotImplementedError("This method must be implemented!")

    def values(self):
        "Return a list of all attribute values."
        raise NotImplementedError("This method must be implemented!")


#============================================================================
#
# HANDLER INTERFACES
#
#============================================================================


# ===== DECLHANDLER =====

class DeclHandler:
    """Optional SAX2 handler for DTD declaration events.

    Note that some DTD declarations are already reported through the
    DTDHandler interface. All events reported to this handler will
    occur between the startDTD and endDTD events of the
    LexicalHandler.

    To set the DeclHandler for an XMLReader, use the setProperty method
    with the identifier http://xml.org/sax/handlers/DeclHandler."""

    def attributeDecl(self, elem_name, attr_name, type, value_def, value):
        """Report an attribute type declaration.

        Only the first declaration will be reported. The type will be
        one of the strings "CDATA", "ID", "IDREF", "IDREFS",
        "NMTOKEN", "NMTOKENS", "ENTITY", "ENTITIES", or "NOTATION", or
        a list of names (in the case of enumerated definitions).

        elem_name is the element type name, attr_name the attribute
        type name, type a string representing the attribute type,
        value_def a string representing the default declaration
        ('#IMPLIED', '#REQUIRED', '#FIXED' or None). value is a string
        representing the attribute's default value, or None if there
        is none."""

    def elementDecl(self, elem_name, content_model):
        """Report an element type declaration.

        Only the first declaration will be reported.

        content_model is the string 'EMPTY', the string 'ANY' or the content
        model structure represented as tuple (separator, tokens, modifier)
        where separator is the separator in the token list (that is, '|' or
        ','), tokens is the list of tokens (element type names or tuples
        representing parentheses) and modifier is the quantity modifier
        ('*', '?' or '+')."""

    def internalEntityDecl(self, name, value):
        """Report an internal entity declaration.

        Only the first declaration of an entity will be reported.

        name is the name of the entity. If it is a parameter entity,
        the name will begin with '%'. value is the replacement text of
        the entity."""

    def externalEntityDecl(self, name, public_id, system_id):
        """Report a parsed entity declaration. (Unparsed entities are
        reported to the DTDHandler.)

        Only the first declaration for each entity will be reported.

        name is the name of the entity. If it is a parameter entity,
        the name will begin with '%'. public_id and system_id are the
        public and system identifiers of the entity. public_id will be
        None if none were declared."""



# ===== LEXICALHANDLER =====

class LexicalHandler:
    """Optional SAX2 handler for lexical events.

    This handler is used to obtain lexical information about an XML
    document, that is, information about how the document was encoded
    (as opposed to what it contains, which is reported to the
    ContentHandler), such as comments and CDATA marked section
    boundaries.

    To set the LexicalHandler of an XMLReader, use the setProperty
    method with the property identifier
    'http://xml.org/sax/handlers/LexicalHandler'. There is no
    guarantee that the XMLReader will support or recognize this
    property."""

    def comment(self, content):
        """Reports a comment anywhere in the document (including the
        DTD and outside the document element).

        content is a string that holds the contents of the comment."""

    def startDTD(self, name, public_id, system_id):
        """Report the start of the DTD declarations, if the document
        has an associated DTD.

        A startEntity event will be reported before declaration events
        from the external DTD subset are reported, and this can be
        used to infer from which subset DTD declarations derive.

        name is the name of the document element type, public_id the
        public identifier of the DTD (or None if none were supplied)
        and system_id the system identfier of the external subset (or
        None if none were supplied)."""

    def endDTD(self):
        "Signals the end of DTD declarations."

    def startEntity(self, name):
        """Report the beginning of an entity.

        The start and end of the document entity is not reported. The
        start and end of the external DTD subset is reported with the
        pseudo-name '[dtd]'.

        Skipped entities will be reported through the skippedEntity
        event of the ContentHandler rather than through this event.

        name is the name of the entity. If it is a parameter entity,
        the name will begin with '%'."""

    def endEntity(self, name):
        """Reports the end of an entity. name is the name of the
        entity, and follows the same conventions as for
        startEntity."""

    def startCDATA(self):
        """Reports the beginning of a CDATA marked section.

        The contents of the CDATA marked section will be reported
        through the characters event."""

    def endCDATA(self):
        "Reports the end of a CDATA marked section."


#============================================================================
#
# SAX 1.0 COMPATIBILITY CLASSES
# Note that these are all deprecated.
#
#============================================================================

# ===== ATTRIBUTELIST =====

class AttributeList:
    """Interface for an attribute list. This interface provides
    information about a list of attributes for an element (only
    specified or defaulted attributes will be reported). Note that the
    information returned by this object will be valid only during the
    scope of the DocumentHandler.startElement callback, and the
    attributes will not necessarily be provided in the order declared
    or specified."""

    def getLength(self):
        "Return the number of attributes in list."

    def getName(self, i):
        "Return the name of an attribute in the list."

    def getType(self, i):
        """Return the type of an attribute in the list. (Parameter can be
        either integer index or attribute name.)"""

    def getValue(self, i):
        """Return the value of an attribute in the list. (Parameter can be
        either integer index or attribute name.)"""

    def __len__(self):
        "Alias for getLength."

    def __getitem__(self, key):
        "Alias for getName (if key is an integer) and getValue (if string)."

    def keys(self):
        "Returns a list of the attribute names."

    def has_key(self, key):
        "True if the attribute is in the list, false otherwise."

    def get(self, key, alternative=None):
        """Return the value associated with attribute name; if it is not
        available, then return the alternative."""

    def copy(self):
        "Return a copy of the AttributeList."

    def items(self):
        "Return a list of (attribute_name,value) pairs."

    def values(self):
        "Return a list of all attribute values."


# ===== DOCUMENTHANDLER =====

class DocumentHandler:
    """Handle general document events. This is the main client
    interface for SAX: it contains callbacks for the most important
    document events, such as the start and end of elements. You need
    to create an object that implements this interface, and then
    register it with the Parser. If you do not want to implement
    the entire interface, you can derive a class from HandlerBase,
    which implements the default functionality. You can find the
    location of any document event using the Locator interface
    supplied by setDocumentLocator()."""

    def characters(self, ch, start, length):
        "Handle a character data event."

    def endDocument(self):
        "Handle an event for the end of a document."

    def endElement(self, name):
        "Handle an event for the end of an element."

    def ignorableWhitespace(self, ch, start, length):
        "Handle an event for ignorable whitespace in element content."

    def processingInstruction(self, target, data):
        "Handle a processing instruction event."

    def setDocumentLocator(self, locator):
        "Receive an object for locating the origin of SAX document events."

    def startDocument(self):
        "Handle an event for the beginning of a document."

    def startElement(self, name, atts):
        "Handle an event for the beginning of an element."


# ===== HANDLERBASE =====

class HandlerBase(EntityResolver, DTDHandler, DocumentHandler,\
                     ErrorHandler):
    """Default base class for handlers. This class implements the
    default behaviour for four SAX interfaces: EntityResolver,
    DTDHandler, DocumentHandler, and ErrorHandler: rather
    than implementing those full interfaces, you may simply extend
    this class and override the methods that you need. Note that the
    use of this class is optional (you are free to implement the
    interfaces directly if you wish)."""


# ===== PARSER =====

class Parser:
    """Basic interface for SAX (Simple API for XML) parsers. All SAX
    parsers must implement this basic interface: it allows users to
    register handlers for different types of events and to initiate a
    parse from a URI, a character stream, or a byte stream. SAX
    parsers should also implement a zero-argument constructor."""

    def __init__(self):
        self.doc_handler = DocumentHandler()
        self.dtd_handler = DTDHandler()
        self.ent_handler = EntityResolver()
        self.err_handler = ErrorHandler()

    def parse(self, systemId):
        "Parse an XML document from a system identifier."

    def parseFile(self, fileobj):
        "Parse an XML document from a file-like object."

    def setDocumentHandler(self, handler):
        "Register an object to receive basic document-related events."
        self.doc_handler=handler

    def setDTDHandler(self, handler):
        "Register an object to receive basic DTD-related events."
        self.dtd_handler=handler

    def setEntityResolver(self, resolver):
        "Register an object to resolve external entities."
        self.ent_handler=resolver

    def setErrorHandler(self, handler):
        "Register an object to receive error-message events."
        self.err_handler=handler

    def setLocale(self, locale):
        """Allow an application to set the locale for errors and warnings.

        SAX parsers are not required to provide localisation for errors
        and warnings; if they cannot support the requested locale,
        however, they must throw a SAX exception. Applications may
        request a locale change in the middle of a parse."""
        raise SAXNotSupportedException("Locale support not implemented")
