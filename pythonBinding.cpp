#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include <vector>
#include <iostream>

#include "functions.h"

namespace py = pybind11;

/**
 * Requires that the dtype of x is F
 */
template<typename F>
py::array_t<F> log1exp_template(const py::array &x) {
    py::buffer_info buffer_inf = x.request();
    if(buffer_inf.ndim != 1) {
        std::cerr << "log1exp expects a 1 dimensional array" << std::endl;
        exit(1);
    }

    if(buffer_inf.strides[0] != sizeof(F)) {
        std::cerr << "log1exp expects a densely packed array" << std::endl;
        exit(1);
    }

    size_t L = buffer_inf.size;


    auto *output = new F[L];
    py::capsule free_when_done(output, [](void *f) {
        auto foo = reinterpret_cast<F*>(f);
        delete[] foo;
    });
    auto ptr = reinterpret_cast<F*>(buffer_inf.ptr);
    log1exp_scalar(ptr, output, L);
    return py::array_t<F>({L}, {sizeof(F)}, output, free_when_done);
}

py::array log1exp(const py::array &x) {
    if(x.dtype().is(pybind11::dtype::of<double>())) {
        return log1exp_template<double>(x);
    } else if(x.dtype().is(pybind11::dtype::of<float>())) {
        return log1exp_template<float>(x);
    } else {
        std::cerr << "log1exp expects either 4byte float or 8 byte float as dtype" << std::endl;
        exit(1);
    }
}


PYBIND11_MODULE(simdefy, m) {
m.doc() = "simdefy module"; // optional module docstring

m.def("log1exp", &log1exp, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
}