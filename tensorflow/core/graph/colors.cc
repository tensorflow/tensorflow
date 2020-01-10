#include "tensorflow/core/graph/colors.h"

#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Color palette
// http://www.mulinblog.com/a-color-palette-optimized-for-data-visualization/
static const char* kColors[] = {
    "#F15854",  // red
    "#5DA5DA",  // blue
    "#FAA43A",  // orange
    "#60BD68",  // green
    "#F17CB0",  // pink
    "#B2912F",  // brown
    "#B276B2",  // purple
    "#DECF3F",  // yellow
    "#4D4D4D",  // gray
};

const char* ColorFor(int dindex) {
  return kColors[dindex % TF_ARRAYSIZE(kColors)];
}

}  // namespace tensorflow
