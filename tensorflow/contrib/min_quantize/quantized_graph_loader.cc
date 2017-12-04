//
// by afpro.
//

#include "quantized_graph_loader.h"
#include "tensorflow/core/framework/types.h"

#include <fstream>
#include <sstream>
#include <array>
#include <map>
#include <stdexcept>

#include <unistd.h>
#include <errno.h>
#include <string.h>

namespace {
    struct GraphDataItem {
        std::string name;
        tensorflow::Tensor tensor;

        GraphDataItem(const std::string &name, const tensorflow::DataType &type, const tensorflow::TensorShape &shape)
            : name(name), tensor(type, shape) {}
    };

    class GraphDataImpl : public GraphData {
    public:
        ~GraphDataImpl() override {
        }

        void append(std::vector<std::pair<std::string, tensorflow::Tensor> > &inputs) override {
          inputs.reserve(inputs.size() + items.size());
          for (const auto &item : items) {
            inputs.push_back(std::make_pair(item->name, item->tensor));
          }
        }

        tensorflow::Tensor &addNormal(const std::string &name,
                                      tensorflow::DataType type, const tensorflow::TensorShape &shape) {
          auto item = new GraphDataItem(name, type, shape);
          items.push_back(std::unique_ptr<GraphDataItem>(item));
          return item->tensor;
        }

    private:
        std::vector<std::unique_ptr<GraphDataItem>> items;
    };

    template<typename TTensor>
    struct RestoreUtils {
        template<typename TRaw>
        inline static void restoreRaw(TRaw &raw, float base,
                                      typename tensorflow::TTypes<TTensor>::Flat flat, int size) {
          for (int i = 0; i < size; i++) {
            flat(i) = static_cast<TTensor>(raw[i] + base);
          }
        };

        template<typename TIndex>
        inline static void restoreSimple(TIndex &index, float base, float step,
                                         typename tensorflow::TTypes<TTensor>::Flat flat, int size) {
          for (int i = 0; i < size; i++) {
            flat(i) = index[i] == 0 ? 0 : static_cast<TTensor>(index[i] * step + base);
          }
        };

        template<typename TTable, typename TIndex>
        inline static void restoreTable(TTable &table, TIndex &index,
                                        typename tensorflow::TTypes<TTensor>::Flat flat, int size) {
          for (int i = 0; i < size; i++) {
            flat(i) = static_cast<TTensor>(table[index[i]]);
          }
        };
    };
}

tensorflow::Status loadQuantizedGraph(tensorflow::Session *session,
                                      GraphData **pGD,
                                      const QuantizedGraph &graph) {
  if (!pGD) {
    return tensorflow::errors::Unknown("pGD is null");
  }

  tensorflow::Status createSessionStatus = session->Extend(graph.graph());
  if (!createSessionStatus.ok()) {
    return createSessionStatus;
  }

  GraphDataImpl *pGDImpl = new GraphDataImpl();

#define FAIL_LOAD_QUANTIZED_GRAPH(...) {\
    delete pGDImpl;\
    return tensorflow::errors::Unknown(__VA_ARGS__);\
  }

  for (const QuantizedItem &item : graph.items()) {
    // get shape
    tensorflow::TensorShape shape;
    for (int dim : item.shape()) {
      shape.AddDim(dim);
    }

    // prepare data
    auto &tensor = pGDImpl->addNormal(item.name(), item.dtype(), shape);

    // restore data
#define RESTORE_QUANTIZED_ITEM(_DTYPE, _TABLE, _RAW) \
    if (item.dtype() == _DTYPE) {\
      switch (item.vtype()) { \
        case ValueType::RAW: \
          RestoreUtils<tensorflow::EnumToDataType<_DTYPE>::Type>::restoreRaw(_RAW,\
                     item.base(),\
                     tensor.flat<tensorflow::EnumToDataType<_DTYPE>::Type>(),\
                     static_cast<int>(shape.num_elements()));\
          break;\
        case ValueType::SIMPLE:\
          RestoreUtils<tensorflow::EnumToDataType<_DTYPE>::Type>::restoreSimple(item.index(), item.base(), item.step(),\
                     tensor.flat<tensorflow::EnumToDataType<_DTYPE>::Type>(),\
                     static_cast<int>(shape.num_elements()));\
          break;\
        case ValueType::TABLE:\
          RestoreUtils<tensorflow::EnumToDataType<_DTYPE>::Type>::restoreTable(_TABLE, item.index(),\
                     tensor.flat<tensorflow::EnumToDataType<_DTYPE>::Type>(),\
                     static_cast<int>(shape.num_elements()));\
          break;\
        default:\
          FAIL_LOAD_QUANTIZED_GRAPH("invalid item.dtype()");\
          break;\
      }\
    }

    RESTORE_QUANTIZED_ITEM(tensorflow::DT_FLOAT, item.float_table(), item.float_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_DOUBLE, item.float_table(), item.float_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_INT8, item.int_table(), item.int_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_INT16, item.int_table(), item.int_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_INT32, item.int_table(), item.int_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_INT64, item.int_table(), item.int_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_UINT8, item.int_table(), item.int_raw())
    else RESTORE_QUANTIZED_ITEM(tensorflow::DT_UINT16, item.int_table(), item.int_raw())
  }

  *pGD = pGDImpl;
  return tensorflow::Status::OK();
}

tensorflow::Status loadQuantizedGraph(tensorflow::Session *session,
                                      GraphData **pGD,
                                      std::istream *quantizedGraphData) {
  if (!pGD) {
    return tensorflow::errors::Unknown("pGD is null");
  }

  QuantizedGraph graph;
  if (!graph.ParseFromIstream(quantizedGraphData)) {
    return tensorflow::errors::Unknown("parse quantized graph failed");
  }

  return loadQuantizedGraph(session, pGD, graph);
}

tensorflow::Status loadQuantizedGraph(tensorflow::Session *session,
                                      GraphData **pGD,
                                      void *quantizedGraphData,
                                      int quantizedGraphDataSize) {
  if (!pGD) {
    return tensorflow::errors::Unknown("pGD is null");
  }

  QuantizedGraph graph;
  if (!graph.ParseFromArray(quantizedGraphData, quantizedGraphDataSize)) {
    return tensorflow::errors::Unknown("parse quantized graph failed");
  }

  return loadQuantizedGraph(session, pGD, graph);
}
