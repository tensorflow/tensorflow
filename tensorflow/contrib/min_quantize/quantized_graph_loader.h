//
// by afpro.
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <istream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/contrib/min_quantize/quantized.pb.h"

// graph data class
class GraphData {
public:
  virtual ~GraphData() {}
  virtual void append(std::vector<std::pair<std::string, tensorflow::Tensor> > &inputs) = 0;
};

// map file name generator
class MapFileNameGenerator {
public:
  virtual ~MapFileNameGenerator() {}
};

// methods
tensorflow::Status loadQuantizedGraph(tensorflow::Session *session,
                                      GraphData **pGD,
                                      const QuantizedGraph &quantizedGraph);

tensorflow::Status loadQuantizedGraph(tensorflow::Session *session,
                                      GraphData **pGD,
                                      std::istream *quantizedGraphData);

tensorflow::Status loadQuantizedGraph(tensorflow::Session *session,
                                      GraphData **pGD,
                                      void *quantizedGraphData,
                                      int quantizedGraphDataSize);

