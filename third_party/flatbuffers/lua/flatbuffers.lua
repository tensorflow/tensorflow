local m = {}

m.Builder = require("flatbuffers.builder").New
m.N = require("flatbuffers.numTypes")
m.view = require("flatbuffers.view")
m.binaryArray = require("flatbuffers.binaryarray")

return m