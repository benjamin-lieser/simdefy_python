#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include <vector>
#include <iostream>
#include <atomic>

#include "functions.h"


const unsigned int N = 2000;

float pre1[N];
float pre2[N];

const float eps = 0.01;
const float rcp_eps = 1.0f / eps;

template<typename F>
void calc_precalc() {
	for(unsigned int i = 0; i < N; i++) {
		F x_i = std::exp(eps * (float)i);
		F G_i = std::lgamma(x_i);
		//Can be optimized, but this is just one time per loading of the library
		F x_i1 = std::exp(eps * (float)(i+1));
		F G_i1 = std::lgamma(x_i1);
		
		pre1[i] = G_i - ((G_i1 - G_i)*x_i / (x_i1 - x_i));
		pre2[i] = (G_i1 - G_i) / (x_i1 - x_i);
	}
}

template<typename F>
void log_gamma(F *input, F *output, unsigned int L) {
	for(unsigned j = 0; j < L; j++) {
		F x = input[j];
		auto i = (unsigned long long) std::floor(std::log(x) / eps);
		output[j] = pre1[i] + x * pre2[i];
	}
}

template<typename F>
typename SIMD<F>::type log_gamma_simd_register(typename SIMD<F>::type data) {
    auto log = SIMD<F>::log(data);
    auto rcp_eps_reg = SIMD<F>::set(rcp_eps);
    auto log_div_eps = SIMD<F>::mul(log, rcp_eps_reg);
    auto index = SIMD<F>::floor(log_div_eps);
    auto int_index = SIMD<F>::toInt_32(index);
    auto pre1_reg = simde_mm256_i32gather_ps(pre1, int_index, 4);
    auto pre2_reg = simde_mm256_i32gather_ps(pre2, int_index, 4);
    auto mul = SIMD<F>::mul(pre2_reg, data);
    return SIMD<F>::add(pre1_reg, mul);
}

template<typename F>
void log_gamma_simd(F *input, F *output, size_t L) {
    size_t simd_blocks = L / SIMD<F>::count;

    size_t index = 0;
    for(; index < simd_blocks; index++) {
        auto data = SIMD<F>::loadU(input + index * SIMD<F>::count);
        auto result = log_gamma_simd_register<F>(data);
        SIMD<F>::store(output + index * SIMD<F>::count, result);
    }

    //Do the last few elements

    size_t remainder_elements = L % SIMD<F>::count;
    auto data = SIMD<F>::set(1.0);
    std::memcpy(&data, input + index * SIMD<F>::count, remainder_elements * sizeof(F));
    data = log_gamma_simd_register<F>(data);
    std::memcpy(output + index * SIMD<F>::count, &data, remainder_elements * sizeof(F));
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

template<typename F>
py::array_t<F> log_gamma_template(const py::array_t<F, py::array::c_style> &x) {
    py::buffer_info buffer_inf = x.request();

    size_t L = buffer_inf.size;

    auto *output = allocAligned<F>(L);
    py::capsule free_when_done(output, [](void *f) {
        auto foo = reinterpret_cast<F*>(f);
        freeAligned(foo);
    });
    auto ptr = reinterpret_cast<F*>(buffer_inf.ptr);
    log_gamma_simd(ptr, output, L);
    return py::array_t<F>(buffer_inf.shape, buffer_inf.strides, output, free_when_done);
}

py::array log1exp_non_dense(const py::array&) {
    throw std::invalid_argument("Arguemnt has to be a dense float32 numpy array");
}


PYBIND11_MODULE(simdefy, m) {
m.doc() = "simdefy module"; // optional module docstring

//m.def("log_gamma", &log1exp_template<double>, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
m.def("log_gamma", &log1exp_template<float>, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
m.def("log_gamma_avx2", &log_gamma_template<float>, "", py::arg("x"));
m.def("log_gamma", &log1exp_non_dense, "calculates log(1+exp) for an numpy array, returns a new array", py::arg("x"));
m.def("init", &calc_precalc<float>, "Init the lookup table");
}
