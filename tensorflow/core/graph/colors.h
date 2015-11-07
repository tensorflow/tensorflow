#ifndef TENSORFLOW_GRAPH_COLORS_H_
#define TENSORFLOW_GRAPH_COLORS_H_

namespace tensorflow {

// Return a color drawn from a palette to represent an entity
// identified by "i".  The return value has the form "#RRGGBB" Note
// that the palette has a limited set of colors and therefore colors
// will be reused eventually.
const char* ColorFor(int dindex);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_COLORS_H_
