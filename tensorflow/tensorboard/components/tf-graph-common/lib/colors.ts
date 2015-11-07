module tf {

/**
 * Mapping from color palette name to color pallette, which contains
 * exact colors for multiple states of a single color pallette.
 */
export let COLORS = [
  {
    "name": "Google Blue",
    "color": "#4184f3",
    "active": "#3a53c5",
    "disabled": "#cad8fc"
  },
  {
    "name": "Google Red",
    "color": "#db4437",
    "active": "#8f2a0c",
    "disabled": "#e8c6c1"
  },
  {
    "name": "Google Yellow",
    "color": "#f4b400",
    "active": "#db9200",
    "disabled": "#f7e8b0"
  },
  {
    "name": "Google Green",
    "color": "#0f9d58",
    "active": "#488046",
    "disabled": "#c2e1cc"
  },
  {
    "name": "Purple",
    "color": "#aa46bb",
    "active": "#5c1398",
    "disabled": "#d7bce6"
  },
  {
    "name": "Teal",
    "color": "#00abc0",
    "active": "#47828e",
    "disabled": "#c2eaf2"
  },
  {
    "name": "Deep Orange",
    "color": "#ff6f42",
    "active": "#ca4a06",
    "disabled": "#f2cbba"
  },
  {
    "name": "Lime",
    "color": "#9d9c23",
    "active": "#7f771d",
    "disabled": "#f1f4c2"
  },
  {
    "name": "Indigo",
    "color": "#5b6abf",
    "active": "#3e47a9",
    "disabled": "#c5c8e8"
  },
  {
    "name": "Pink",
    "color": "#ef6191",
    "active": "#ca1c60",
    "disabled": "#e9b9ce"
  },
  {
    "name": "Deep Teal",
    "color": "#00786a",
    "active": "#2b4f43",
    "disabled": "#bededa"
  },
  {
    "name": "Deep Pink",
    "color": "#c1175a",
    "active": "#75084f",
    "disabled": "#de8cae"
  },
  {
    "name": "Gray",
    "color": "#9E9E9E", // 500
    "active": "#424242", // 800
    "disabled": "F5F5F5" // 100
  }
].reduce((m, c) => {
  m[c.name] = c;
  return m;
}, {});

/**
 * Mapping from op category to color palette name
 * e.g.,  OP_GROUP_COLORS["state_ops"] = "Google Blue";
 */
export let OP_GROUP_COLORS = [
  {
    color: "Google Red",
    groups: ["gen_legacy_ops", "legacy_ops", "legacy_flogs_input",
      "legacy_image_input", "legacy_input_example_input",
      "legacy_sequence_input", "legacy_seti_input_input"]
  }, {
    color: "Deep Orange",
    groups: ["constant_ops"]
  }, {
    color: "Indigo",
    groups: ["state_ops"]
  }, {
    color: "Purple",
    groups: ["nn_ops", "nn"]
  }, {
    color: "Google Green",
    groups: ["math_ops"]
  }, {
    color: "Lime",
    groups: ["array_ops"]
  }, {
    color: "Teal",
    groups: ["control_flow_ops", "data_flow_ops"]
  }, {
    color: "Pink",
    groups: ["summary_ops"]
  }, {
    color: "Deep Pink",
    groups: ["io_ops"]
  }
].reduce((m, c) => {
  c.groups.forEach(function(group) {
    m[group] = c.color;
  });
  return m;
}, {});

}
