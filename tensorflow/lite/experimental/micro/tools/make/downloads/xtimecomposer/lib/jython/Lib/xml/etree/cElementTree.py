# make an exact copy of ElementTree's namespace here to support even
# private API usage
from xml.etree.ElementTree import (
    Comment, Element, ElementPath, ElementTree, PI, ProcessingInstruction,
    QName, SubElement, TreeBuilder, VERSION, XML, XMLID, XMLParser,
    XMLTreeBuilder, _Element, _ElementInterface, _SimpleElementPath,
    __all__, __doc__, __file__, __name__, _encode, _encode_entity,
    _escape, _escape_attrib, _escape_cdata, _escape_map, _namespace_map,
    _raise_serialization_error, dump, fixtag, fromstring, iselement,
    iterparse, parse, re, string, sys, tostring)
