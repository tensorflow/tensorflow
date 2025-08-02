/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <any>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/python/model_utils_core/pybind_api_internal.h"

namespace py = pybind11;

namespace {

class MuContainer {
 public:
  template <typename T>
  explicit MuContainer(T&& d)
      : data_(std::any(std::forward<T>(d))),
        type_hash_(typeid(T).hash_code()),
        template_type_name_(typeid(T).name()) {}

  MuContainer() = delete;
  MuContainer(MuContainer&&) = default;
  MuContainer& operator=(MuContainer&&) = default;

  template <typename T, typename... Args>
  static MuContainer create(Args&&... args) {
    return MuContainer(std::make_any<T>(std::forward<Args>(args)...));
  }

  template <typename T>
  inline T* get() {
    return std::any_cast<T>(&data_);
  }

  size_t type_hash() const { return type_hash_; }

  std::string debuginfo() const {
    return absl::StrCat("template_type_name=", template_type_name_, ", ",
                        "type_hash=", type_hash_);
  }

 protected:
  std::any data_;
  size_t type_hash_ = 0;
  const char* template_type_name_ = "";
};

template <typename T>
class MuMlirObject : public MuContainer {
 public:
  explicit MuMlirObject(T data) : MuContainer(std::move(data)) {}

  T* get() { return MuContainer::get<T>(); }

  std::string to_string() {
    std::string s;
    llvm::raw_string_ostream ostream(s);
    T* obj = get();
    if constexpr (std::is_pointer_v<T>) {
      (*obj)->print(ostream);
    } else {
      obj->print(ostream);
    }
    return s;
  }
};

class MuMlirOperation : public MuMlirObject<mlir::Operation*> {
 public:
  explicit MuMlirOperation(mlir::Operation* op) : MuMlirObject(op) {}
  explicit MuMlirOperation(mlir::ModuleOp module_op)
      : MuMlirObject(module_op.getOperation()) {}
};

class MuMlirAttribute : public MuMlirObject<mlir::Attribute> {
 public:
  explicit MuMlirAttribute(mlir::Attribute attr) : MuMlirObject(attr) {
    type_hash_ = mlir::hash_value(attr.getTypeID());
  }
};

class MuMlirType : public MuMlirObject<mlir::Type> {
 public:
  explicit MuMlirType(mlir::Type type) : MuMlirObject(type) {
    type_hash_ = mlir::hash_value(type.getTypeID());
  }
};

using MuMlirLocation = MuMlirObject<mlir::Location>;

class MlirTransformRegistry {
 public:
  template <typename T>
  void register_transform(py::object cls) {
    transforms_[mlir::hash_value(T::getTypeID())] = cls;
  }
  void register_op_cls(absl::string_view name, py::object cls) {
    op_clss_[name] = cls;
  }

  py::object attribute_from_mlir(MuMlirAttribute& attr) {
    auto cls = transforms_.find(attr.type_hash());
    if (cls == transforms_.end()) {
      return base_attribute_cls_(attr);
    }
    return cls->second(attr);
  }

  py::object type_from_mlir(MuMlirType& type) {
    auto cls = transforms_.find(type.type_hash());
    if (cls == transforms_.end()) {
      return base_type_cls_(type);
    }
    return cls->second(type);
  }

  py::object register_base_attribute_cls(py::object cls) {
    base_attribute_cls_ = cls;
    return cls;
  }
  py::object register_base_type_cls(py::object cls) {
    base_type_cls_ = cls;
    return cls;
  }

  py::object register_base_op_cls(py::object cls) {
    base_op_cls_ = cls;
    return cls;
  }
  py::object base_op_cls() { return base_op_cls_; }
  std::optional<py::object> get_op_cls(std::string name) {
    auto cls = op_clss_.find(name);
    if (cls == op_clss_.end()) {
      return std::nullopt;
    }
    return cls->second(name);
  }

 private:
  absl::flat_hash_map<size_t, py::object> transforms_;
  absl::flat_hash_map<std::string, py::object> op_clss_;
  py::object base_attribute_cls_ = py::none();
  py::object base_type_cls_ = py::none();
  py::object base_op_cls_ = py::none();
};  // namespace

static mlir::MLIRContext* g_ir_context = nullptr;

inline void assert_ir_context_set() {
  if (g_ir_context == nullptr) {
    throw std::runtime_error("MLIRContext is not set.");
  }
}

py::bool_ WrappedMlirVerify(MuMlirOperation& op_) {
  mlir::Operation* op = *op_.get();
  absl::Status status = mlir::TFL::model_utils::MlirVerify(op);
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
  return py::bool_(true);
}

}  // namespace

void PopulateModelUtilsCoreApis(py::module& m_) {
  auto mu = m_.def_submodule("model_utils_core_api");
  py::class_<MuContainer>(mu, "MuContainer")
      .def("_cpp_debuginfo",
           [](MuContainer& self) { return self.debuginfo(); });

#define DEFINE_PY_MLIR_OBJECT(name)                                \
  py::class_<name>(mu, #name)                                      \
      .def("__str__", [](name& self) { return self.to_string(); }) \
      .def("_cpp_debuginfo", [](name& self) { return self.debuginfo(); })

  DEFINE_PY_MLIR_OBJECT(MuMlirAttribute);
  DEFINE_PY_MLIR_OBJECT(MuMlirType);
  DEFINE_PY_MLIR_OBJECT(MuMlirLocation);
  DEFINE_PY_MLIR_OBJECT(MuMlirOperation).def("verify", &WrappedMlirVerify);

#undef DEFINE_PY_MLIR_OBJECT

  py::class_<MlirTransformRegistry>(mu, "MlirTransformRegistry")
      .def(py::init<>())
      .def("attribute_from_mlir",
           [](MlirTransformRegistry& self, MuMlirAttribute& attr_) {
             return self.attribute_from_mlir(attr_);
           })
      .def("type_from_mlir",
           [](MlirTransformRegistry& self, MuMlirType& type_) {
             return self.type_from_mlir(type_);
           })
      .def("register_base_attribute_cls",
           [](MlirTransformRegistry& self, py::object cls) {
             return self.register_base_attribute_cls(cls);
           })
      .def("register_base_type_cls",
           [](MlirTransformRegistry& self, py::object cls) {
             return self.register_base_type_cls(cls);
           })
      .def("register_base_op_cls",
           [](MlirTransformRegistry& self, py::object cls) {
             return self.register_base_op_cls(cls);
           })
      .def("register_StringAttr",
           [](MlirTransformRegistry& self, py::class_<py::object> cls) {
             self.register_transform<mlir::StringAttr>(cls);

             cls.def("to_mlir", [](py::handle self) {
               assert_ir_context_set();
               py::str py_str = self.attr("data");
               mlir::StringAttr attr = mlir::StringAttr::get(
                   g_ir_context, py_str.cast<std::string>());
               return MuMlirAttribute(std::move(attr));
             });
             cls.def_static("from_mlir", [cls](MuMlirAttribute* attr_) {
               auto attr = mlir::dyn_cast<mlir::StringAttr>(*attr_->get());
               return py::str(attr.strref().str());
             });
             return cls;
           });

  mu.def("create_ir_context", []() {
    std::unique_ptr<mlir::MLIRContext> context =
        mlir::TFL::model_utils::CreateIRContext();
    return MuContainer(context.release());
  });

  mu.def("set_ir_context", [](MuContainer& context_) {
    mlir::MLIRContext* context = *context_.get<mlir::MLIRContext*>();
    g_ir_context = context;
  });
  mu.def("clear_ir_context", []() { g_ir_context = nullptr; });

  mu.def("flatbuffer_to_mlir", [](py::bytes buffer) {
    assert_ir_context_set();

    mlir::ModuleOp module_op =
        tflite::FlatBufferToMlir(buffer, g_ir_context,
                                 mlir::UnknownLoc::get(g_ir_context))
            .release();

    return MuMlirOperation(std::move(module_op));
  });

  mu.def("mlir_to_flatbuffer", [](MuMlirOperation& module_op_) {
    mlir::Operation* module_op = *module_op_.get();

    tflite::FlatbufferExportOptions options;
    std::string result;
    tflite::MlirToFlatBufferTranslateFunction(
        mlir::dyn_cast<mlir::ModuleOp>(module_op), options, &result, true);
    return py::bytes(result);
  });

  mu.def("mlir_verify", &WrappedMlirVerify);

  mu.def("op_python_to_mlir", [](py::object py_op) {
    assert_ir_context_set();

    absl::AnyInvocable<mlir::Operation*(
        py::handle, absl::flat_hash_map<PyObject*, mlir::Value>&) const>
        build_op;

    auto build_region = [&](py::handle py_region, mlir::Region& region) {
      py::list py_blocks(py_region.attr("blocks"));
      for (py::handle py_block : py_blocks) {
        llvm::SmallVector<mlir::Type> arg_types;
        py::list py_block_args(py_block.attr("args"));
        for (py::handle py_arg : py_block_args) {
          py::object py_type = py_arg.attr("type").attr("to_mlir")();
          auto wrapped_type = py_type.cast<MuMlirType*>();
          arg_types.push_back(*wrapped_type->get());
        }

        mlir::Block& block = region.emplaceBlock();
        for (mlir::Type& type : arg_types) {
          block.addArgument(type, mlir::UnknownLoc::get(g_ir_context));
        }

        absl::flat_hash_map<PyObject*, mlir::Value> mapping;
        for (int i = 0; i < block.getNumArguments(); ++i) {
          mapping[py_block_args[i].ptr()] = block.getArgument(i);
        }

        for (py::handle py_op : py_block.attr("ops")) {
          mlir::Operation* op = build_op(py_op, mapping);
          block.push_back(op);
        }
      }
    };

    build_op = [&](py::handle py_op,
                   absl::flat_hash_map<PyObject*, mlir::Value>& mapping) {
      llvm::SmallVector<mlir::Value> operands;
      py::list py_operands(py_op.attr("operands"));
      for (int i = 0; i < py_operands.size(); ++i) {
        auto it = mapping.find(py_operands[i].ptr());
        if (it == mapping.end()) {
          throw std::runtime_error(
              "Failed to MLIR operation: operand not found.");
        }
        operands.push_back(it->second);
      }

      llvm::SmallVector<mlir::Type> result_types;
      py::list py_results(py_op.attr("results"));
      for (int i = 0; i < py_results.size(); ++i) {
        py::object py_wrapped_type =
            py_results[i].attr("type").attr("to_mlir")();
        auto wrapped_type = py_wrapped_type.cast<MuMlirType*>();
        result_types.push_back(*wrapped_type->get());
      }

      llvm::SmallVector<mlir::NamedAttribute> attributes;
      if (py::hasattr(py_op, "attributes") &&
          py::isinstance<py::dict>(py_op.attr("attributes"))) {
        py::dict py_attributes = py_op.attr("attributes");
        for (auto [py_name, py_attr] : py_attributes) {
          py::object py_wrapped_attr = py_attr.attr("to_mlir")();
          auto wrapped_attr = py_wrapped_attr.cast<MuMlirAttribute*>();

          auto name = py::str(py_name).cast<std::string>();
          attributes.push_back(
              mlir::NamedAttribute(name, *wrapped_attr->get()));
        }
      }

      mlir::Location loc = mlir::UnknownLoc::get(g_ir_context);
      if (py::hasattr(py_op, "location") &&
          py::isinstance<MuMlirLocation>(py_op.attr("location"))) {
        auto wrapped_loc = py_op.attr("location").cast<MuMlirLocation*>();
        if (wrapped_loc->type_hash() == typeid(mlir::Location).hash_code()) {
          loc = *wrapped_loc->get();
        }
      }

      std::string op_name = py::str(py_op.attr("name"));
      mlir::OperationState op_state(loc, op_name);
      op_state.addOperands(operands);
      op_state.addTypes(result_types);
      op_state.addAttributes(attributes);

      llvm::SmallVector<mlir::Region> regions;
      if (py::hasattr(py_op, "regions") && !py_op.attr("regions").is_none()) {
        py::list py_regions(py_op.attr("regions"));
        for (int i = 0; i < py_regions.size(); ++i) {
          py::handle py_region = py_regions[i];
          build_region(py_region, *op_state.addRegion());
        }
      }

      mlir::Operation* op = mlir::Operation::create(op_state);

      for (int i = 0; i < op->getNumResults(); ++i) {
        mapping[py_results[i].ptr()] = op->getResult(i);
      }
      return op;
    };

    absl::flat_hash_map<PyObject*, mlir::Value> mapping;
    mlir::Operation* op = build_op(py_op, mapping);
    return MuMlirOperation(op);
  });

  mu.def("op_mlir_to_python", [](MlirTransformRegistry& mlir_transforms,
                                 py::object region_cls, py::object block_cls,
                                 MuMlirOperation& op_) {
    assert_ir_context_set();
    mlir::Operation* op = *op_.get();

    absl::AnyInvocable<py::object(mlir::Region&) const> build_region;

    auto build_op = [&](mlir::Operation& op,
                        llvm::DenseMap<mlir::Value, py::object>& mapping) {
      py::list py_operands;
      for (mlir::Value operand : op.getOperands()) {
        auto it = mapping.find(operand);
        if (it == mapping.end()) {
          throw std::runtime_error(
              "Failed to build Python operation: operand not found.");
        }
        py::object py_operand = it->second;
        py_operands.append(py_operand);
      }

      py::list py_result_types;
      for (mlir::Type type : op.getResultTypes()) {
        MuMlirType wrapped_type(std::move(type));
        py_result_types.append(mlir_transforms.type_from_mlir(wrapped_type));
      }

      py::list py_regions;
      for (mlir::Region& region : op.getRegions()) {
        py_regions.append(build_region(region));
      }

      py::dict py_attributes;
      for (mlir::NamedAttribute named_attr : op.getAttrDictionary()) {
        mlir::Attribute attr = named_attr.getValue();
        MuMlirAttribute wrapped_attr(std::move(attr));
        py_attributes[py::str(named_attr.getName().str())] =
            mlir_transforms.attribute_from_mlir(wrapped_attr);
      }

      py::object py_op;
      std::optional<py::object> op_cls =
          mlir_transforms.get_op_cls(op.getName().getStringRef().str());
      if (op_cls.has_value()) {
        py_op = op_cls->attr("build")(
            py::arg("operands") = py_operands,
            py::arg("result_types") = py_result_types,
            py::arg("attributes") = py_attributes,
            py::arg("regions") = py_regions,
            py::arg("location") = MuMlirLocation(op.getLoc()));
      } else {
        py_op = mlir_transforms.base_op_cls()(
            py::arg("name") = py::str(op.getName().getStringRef().str()),
            py::arg("operands") = py_operands,
            py::arg("result_types") = py_result_types,
            py::arg("attributes") = py_attributes,
            py::arg("regions") = py_regions,
            py::arg("location") = MuMlirLocation(op.getLoc()));
      }

      py::list py_results(py_op.attr("results"));
      for (int i = 0; i < op.getNumResults(); ++i) {
        mapping.insert({op.getResult(i), py_results[i]});
      }
      return py_op;
    };

    auto build_block = [&](mlir::Block& block) {
      py::list py_arg_types;
      for (mlir::Type type : block.getArgumentTypes()) {
        MuMlirType wrapped_type(std::move(type));
        py_arg_types.append(mlir_transforms.type_from_mlir(wrapped_type));
      }
      py::object py_block = block_cls(py::arg("arg_types") = py_arg_types);

      llvm::DenseMap<mlir::Value, py::object> mapping;
      py::list py_block_args = py_block.attr("args");
      for (int i = 0; i < block.getNumArguments(); ++i) {
        mapping.insert({block.getArgument(i), py_block_args[i]});
      }

      py::list py_block_ops;
      for (mlir::Operation& op : block.getOperations()) {
        py::object py_op = build_op(op, mapping);
        py_block_ops.append(py_op);
      }
      py_block.attr("add_ops")(py_block_ops);
      return py_block;
    };

    build_region = [&](mlir::Region& region) {
      py::list py_blocks;
      for (mlir::Block& block : region.getBlocks()) {
        py_blocks.append(build_block(block));
      }
      return region_cls(py_blocks);
    };

    llvm::DenseMap<mlir::Value, py::object> mapping;
    return build_op(*op, mapping);
  });
}
