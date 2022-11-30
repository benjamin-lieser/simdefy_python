#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include <vector>
#include <iostream>


bool precalc = false;

const unsigned int N = 2000;

float pre1[N];
float pre2[N];

const float eps = 0.0075;

template<typename F>
void precalc() {
	for(unsigned int i = 0; i < N; i++) {
		F x_i = std::exp(eps * i);
		F G_i = std::lgamma(x_i);
		//Can be optimized, but this is just one time per loading of the library
		F x_i1 = std::exp(eps * (i+1));
		F G_i1 = std::lgamma(x_i1);
		
		pre1[i] = G_i - ((G_i1 - G_i)*x_i / (x_i1 - x_i));
		pre2[i] = (G_i1 - G_i) / (x_i1 - x_i)
	}
	precalc = true;
}

template<typename F>
void log_gamma(F *input, F *output, unsigned int L) {
	if(!precalc) {
		precalc();
	}
	for(unsigned j = 0; j < L; j++) {
		F x = input[i];
		unsigned long long i = std::floor(std::log(x) / eps);
		output[i] = pre1[i] + x * pre2[i];
	}
}


namespace py = pybind11;

/**
 * Requires that the dtype of x is F
 */
template<typename F>
py::array_t<F> log1exp_template(const py::array_t<F, py::array::c_style> &x) {
    py::buffer_info buffer_inf = x.request();

    size_t L = buffer_inf.size;

    auto *output = allocAligned<F>(L);
    py::capsule free_when_done(output, [](void *f) {
        auto foo = reinterpret_cast<F*>(f);
        freeAligned(foo);
    });
    auto ptr = reinterpret_cast<F*>(buffer_inf.ptr);
    log_gamma(ptr, output, L);
    return py::array_t<F>(buffer_inf.shape, buffer_inf.strides, output, free_when_done);
}

py::array log1exp_non_dense(const py::array&) {
    std::cerr << "simdefy only works with dense array with dtype single or double" << std::endl;
    exit(1);
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

m.def("log_gamma", &log1exp_template<double>, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
m.def("log_gamma", &log1exp_template<float>, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
m.def("log_gamma", &log1exp_non_dense, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
}
