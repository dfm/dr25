#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "quad.h"

namespace py = pybind11;

PYBIND11_MODULE(quad, m) {
  m.def("quad", py::vectorize(batman::quad));
}
