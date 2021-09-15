const path = require("path");

module.exports = {
  entry: { alchemyWeb3: "./dist/esm/index.js" },
  mode: "production",
  output: {
    filename: "[name].min.js",
    library: "AlchemyWeb3",
    libraryTarget: "var",
    path: path.resolve(__dirname, "dist"),
  },
};
