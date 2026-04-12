#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nb_classifier.h"

namespace py = pybind11;
using namespace kmer;

PYBIND11_MODULE(fastnb_cpp, m) {
    m.doc() = "fastnb: High-performance C++ Naive Bayes classifier for 16S rRNA taxonomy";

    py::class_<NbConfig>(m, "NbConfig")
        .def(py::init<>())
        .def_readwrite("num_threads", &NbConfig::num_threads)
        .def_readwrite("confidence_threshold", &NbConfig::confidence_threshold);

    py::class_<NbResult>(m, "NbResult")
        .def_readonly("taxonomy", &NbResult::taxonomy)
        .def_readonly("confidence", &NbResult::confidence);

    py::class_<NbClassifier>(m, "NbClassifier")
        .def(py::init<>())
        .def("load", &NbClassifier::load, py::arg("params_dir"))
        .def("classify", &NbClassifier::classify,
             py::arg("sequences"), py::arg("config") = NbConfig{});
}
