// TODO(fengliuai): automatically generate this file
// TODO(fengliuai): add all the xla_hlo ops

static std::unique_ptr<quant::OpQuantSpec> GetOpQuantSpec(mlir::Operation *op) {
  auto spec = absl::make_unique<quant::OpQuantSpec>();
  return spec;
}
