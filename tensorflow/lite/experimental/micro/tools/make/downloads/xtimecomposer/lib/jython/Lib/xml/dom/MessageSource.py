# DOMException
from xml.dom import INDEX_SIZE_ERR, DOMSTRING_SIZE_ERR , HIERARCHY_REQUEST_ERR
from xml.dom import WRONG_DOCUMENT_ERR, INVALID_CHARACTER_ERR, NO_DATA_ALLOWED_ERR
from xml.dom import NO_MODIFICATION_ALLOWED_ERR, NOT_FOUND_ERR, NOT_SUPPORTED_ERR
from xml.dom import INUSE_ATTRIBUTE_ERR, INVALID_STATE_ERR, SYNTAX_ERR
from xml.dom import INVALID_MODIFICATION_ERR, NAMESPACE_ERR, INVALID_ACCESS_ERR
from xml.dom import VALIDATION_ERR

# EventException
from xml.dom import UNSPECIFIED_EVENT_TYPE_ERR

#Range Exceptions
from xml.dom import BAD_BOUNDARYPOINTS_ERR
from xml.dom import INVALID_NODE_TYPE_ERR

# Fourthought Exceptions
from xml.dom import XML_PARSE_ERR

from xml.FtCore import get_translator

_ = get_translator("dom")


DOMExceptionStrings = {
    INDEX_SIZE_ERR: _("Index error accessing NodeList or NamedNodeMap"),
    DOMSTRING_SIZE_ERR: _("DOMString exceeds maximum size"),
    HIERARCHY_REQUEST_ERR: _("Node manipulation results in invalid parent/child relationship."),
    WRONG_DOCUMENT_ERR: _("Node is from a different document"),
    INVALID_CHARACTER_ERR: _("Invalid or illegal character"),
    NO_DATA_ALLOWED_ERR: _("Node does not support data"),
    NO_MODIFICATION_ALLOWED_ERR: _("Attempt to modify a read-only object"),
    NOT_FOUND_ERR: _("Node does not exist in this context"),
    NOT_SUPPORTED_ERR: _("Object or operation not supported"),
    INUSE_ATTRIBUTE_ERR: _("Attribute already in use by an element"),
    INVALID_STATE_ERR: _("Object is not, or is no longer, usable"),
    SYNTAX_ERR: _("Specified string is invalid or illegal"),
    INVALID_MODIFICATION_ERR: _("Attempt to modify the type of a node"),
    NAMESPACE_ERR: _("Invalid or illegal namespace operation"),
    INVALID_ACCESS_ERR: _("Object does not support this operation or parameter"),
    VALIDATION_ERR: _("Operation would invalidate partial validity constraint"),
    }

EventExceptionStrings = {
    UNSPECIFIED_EVENT_TYPE_ERR : _("Uninitialized type in Event object"),
    }

FtExceptionStrings = {
    XML_PARSE_ERR : _("XML parse error at line %d, column %d: %s"),
    }

RangeExceptionStrings = {
    BAD_BOUNDARYPOINTS_ERR : _("Invalid Boundary Points specified for Range"),
    INVALID_NODE_TYPE_ERR : _("Invalid Container Node")
    }
