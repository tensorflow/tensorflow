from __future__ import absolute_import, division, unicode_literals

from xml.dom import Node
from ..constants import namespaces, voidElements, spaceCharacters

__all__ = ["DOCUMENT", "DOCTYPE", "TEXT", "ELEMENT", "COMMENT", "ENTITY", "UNKNOWN",
           "TreeWalker", "NonRecursiveTreeWalker"]

DOCUMENT = Node.DOCUMENT_NODE
DOCTYPE = Node.DOCUMENT_TYPE_NODE
TEXT = Node.TEXT_NODE
ELEMENT = Node.ELEMENT_NODE
COMMENT = Node.COMMENT_NODE
ENTITY = Node.ENTITY_NODE
UNKNOWN = "<#UNKNOWN#>"

spaceCharacters = "".join(spaceCharacters)


class TreeWalker(object):
    def __init__(self, tree):
        self.tree = tree

    def __iter__(self):
        raise NotImplementedError

    def error(self, msg):
        return {"type": "SerializeError", "data": msg}

    def emptyTag(self, namespace, name, attrs, hasChildren=False):
        yield {"type": "EmptyTag", "name": name,
               "namespace": namespace,
               "data": attrs}
        if hasChildren:
            yield self.error("Void element has children")

    def startTag(self, namespace, name, attrs):
        return {"type": "StartTag",
                "name": name,
                "namespace": namespace,
                "data": attrs}

    def endTag(self, namespace, name):
        return {"type": "EndTag",
                "name": name,
                "namespace": namespace}

    def text(self, data):
        data = data
        middle = data.lstrip(spaceCharacters)
        left = data[:len(data) - len(middle)]
        if left:
            yield {"type": "SpaceCharacters", "data": left}
        data = middle
        middle = data.rstrip(spaceCharacters)
        right = data[len(middle):]
        if middle:
            yield {"type": "Characters", "data": middle}
        if right:
            yield {"type": "SpaceCharacters", "data": right}

    def comment(self, data):
        return {"type": "Comment", "data": data}

    def doctype(self, name, publicId=None, systemId=None):
        return {"type": "Doctype",
                "name": name,
                "publicId": publicId,
                "systemId": systemId}

    def entity(self, name):
        return {"type": "Entity", "name": name}

    def unknown(self, nodeType):
        return self.error("Unknown node type: " + nodeType)


class NonRecursiveTreeWalker(TreeWalker):
    def getNodeDetails(self, node):
        raise NotImplementedError

    def getFirstChild(self, node):
        raise NotImplementedError

    def getNextSibling(self, node):
        raise NotImplementedError

    def getParentNode(self, node):
        raise NotImplementedError

    def __iter__(self):
        currentNode = self.tree
        while currentNode is not None:
            details = self.getNodeDetails(currentNode)
            type, details = details[0], details[1:]
            hasChildren = False

            if type == DOCTYPE:
                yield self.doctype(*details)

            elif type == TEXT:
                for token in self.text(*details):
                    yield token

            elif type == ELEMENT:
                namespace, name, attributes, hasChildren = details
                if (not namespace or namespace == namespaces["html"]) and name in voidElements:
                    for token in self.emptyTag(namespace, name, attributes,
                                               hasChildren):
                        yield token
                    hasChildren = False
                else:
                    yield self.startTag(namespace, name, attributes)

            elif type == COMMENT:
                yield self.comment(details[0])

            elif type == ENTITY:
                yield self.entity(details[0])

            elif type == DOCUMENT:
                hasChildren = True

            else:
                yield self.unknown(details[0])

            if hasChildren:
                firstChild = self.getFirstChild(currentNode)
            else:
                firstChild = None

            if firstChild is not None:
                currentNode = firstChild
            else:
                while currentNode is not None:
                    details = self.getNodeDetails(currentNode)
                    type, details = details[0], details[1:]
                    if type == ELEMENT:
                        namespace, name, attributes, hasChildren = details
                        if (namespace and namespace != namespaces["html"]) or name not in voidElements:
                            yield self.endTag(namespace, name)
                    if self.tree is currentNode:
                        currentNode = None
                        break
                    nextSibling = self.getNextSibling(currentNode)
                    if nextSibling is not None:
                        currentNode = nextSibling
                        break
                    else:
                        currentNode = self.getParentNode(currentNode)
