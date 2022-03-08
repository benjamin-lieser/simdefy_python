#ifndef SIMD_H
#define SIMD_H

#include "simde/simde/x86/avx2.h"
#include <cfloat>
#include <iostream>

template<typename F>
struct SIMD;

template<>
struct SIMD<float> {
	//Type of a float simd register
	using type = simde__m256;
    //Type of the integer register with the same size, in this case 32bit int
	using intType = simde__m256i;

	//Number of floats in a float register
	static const unsigned int count = 8;

	static type floor(type a) {
		return simde_mm256_floor_ps(a);
	}

	static intType toInt_32(type a) {
		return simde_mm256_cvtps_epi32(a);
	}

	static intType toIntBitcast(type a) {
		return simde_mm256_castps_si256(a);
	}

	static type toFloatBitcast(intType a) {
		return simde_mm256_castsi256_ps(a);
	}

	static type toFloat(intType a) {
		return simde_mm256_cvtepi32_ps(a);
	}

	static type comp_less(type a, type b) {
		return simde_mm256_cmp_ps(a, b, SIMDE_CMP_LT_OQ);
	}

	static type comp_greater(type a, type b) {
		return simde_mm256_cmp_ps(a,b, SIMDE_CMP_GT_OQ);
	}

	static intType shift_left_32(intType a, unsigned int shift) {
		return simde_mm256_slli_epi32(a, shift);
	}

	static type bit_and(type a, type b) {
		return simde_mm256_and_ps(a, b);
	}

    static type bit_and_not(type a, type b) {
        return simde_mm256_andnot_ps(a, b);
    }

    static type bit_xor(type a, type b) {
        return simde_mm256_xor_ps(a, b);
    }

    static type flip_sign(type a) {
        return bit_and_not(a, set(-0.0f));
    }

    static type abs(type a) {
        return bit_and_not(a, set(-0.0f));
    }

	static type blendv(type a, type b, type mask) {
		return simde_mm256_blendv_ps(a, b, mask);
	}

	static intType add_32(intType a, intType b) {
		return simde_mm256_add_epi32(a, b);
	}

	static type exp2(type X) {
		/////////////////////////////////////////////////////////////////////////////////////
		// SIMD 2^x for four floats
		// Calculate float of 2pow(x) for four floats in parallel with SSE2
		// Relative deviation < 4.6E-6  (< 2.3E-7 with 5'th order polynomial)
		//
		// Internal representation of float number according to IEEE 754 (__m128 --> 4x):
		//   1bit sign, 8 bits exponent, 23 bits mantissa: seee eeee emmm mmmm mmmm mmmm mmmm mmmm
		//                                    0x4b400000 = 0100 1011 0100 0000 0000 0000 0000 0000
		//   In summary: x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeee-127)
		/////////////////////////////////////////////////////////////////////////////////////

		const type CONST32_1f        = set(1.0f);
		const type CONST32_FLTMAXEXP = set(std::numeric_limits<float>::max_exponent - 1);
		const type CONST32_FLTMAX    = set(std::numeric_limits<float>::infinity());
		const type CONST32_FLTMINEXP = set(std::numeric_limits<float>::min_exponent - 1);

		// fifth order
		const type CONST32_A = set(0.00187682f);
		const type CONST32_B = set(0.00898898f);
		const type CONST32_C = set(0.0558282f);
		const type CONST32_D = set(0.240153f);
		const type CONST32_E = set(0.693153f);

		const type maskedMax = comp_greater(X, CONST32_FLTMAXEXP); //Values greater than 127 will be considered inf
		const type maskedMin = comp_greater(X, CONST32_FLTMINEXP); //Values less or equal to -126 will have a zero mask

		const type tx = floor(X);
		const type dx = sub(X, tx);
		const intType lx = toInt_32(tx);

		X = mul(dx, CONST32_A);
		X = add(CONST32_B, X);  // add constant B
		X = mul(dx, X);
		X = add(CONST32_C, X);  // add constant C
		X = mul(dx, X);
		X = add(CONST32_D, X);  // add constant D
		X = mul(dx, X);
		X = add(CONST32_E, X);  // add constant E
		X = mul(dx, X);
		X = add(X, CONST32_1f); // add 1.0f

		const intType lxExp = shift_left_32(lx, 23); // shift the exponent into the correct position

		const intType X_as_int = toIntBitcast(X);

		X = toFloatBitcast(add_32(X_as_int, lxExp));

		// Set all values to zero which were less or equal to -126
		X = bit_and(X, maskedMin);
		// Add MAX_FLT values where entry values were > FLT_MAX_EXP
		return blendv(X, CONST32_FLTMAX, maskedMax);
	}

    static type exp(type x) {
        auto log2e = set(1.44269504089f);
        return exp2(mul(x, log2e));
    }

	static type log2(type X) {

		// Fast SIMD log2 for eight floats
		// Calculate integer of log2 for four floats in parallel with SSE2
		// Maximum deviation: +/- 2.1E-5
		// For a negative argument, nonsense is returned. Otherwise, when <1E-38, a value
		// close to -126 is returned and when >1.7E38, +128 is returned.
		// The function makes use of the representation of 4-byte floating point numbers:
		// seee eeee emmm mmmm mmmm mmmm mmmm mmmm
		// s is the sign, eee eee e gives the exponent + 127 (in hex: 0x7f).
		// The following 23 bits give the mantisse, the binary digits after the decimal
		// point:  x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeeee-127)
		// Therefore,  log2(x) = eeeeeeee-127 + log2(1.mmmmmm...)
		//                     = eeeeeeee-127 + log2(1+y),  where y = 0.mmmmmm...
		//                     ~ eeeeeeee-127 + ((a*y+b)*y+c)*y
		// The coefficients a, b  were determined by a least squares fit, and c=1-a-b to get 1 at y=1.
		// Lower/higher order polynomials may be used for faster or more precise calculation:
		// Order 1: log2(1+y) ~ y
		// Order 2: log2(1+y) = (a*y + 1-a)*y, a=-0.3427
		//  => max dev = +/- 8E-3
		// Order 3: log2(1+y) = ((a*y+b)*y + 1-a-b)*y, a=0.1564, b=-0.5773
		//  => max dev = +/- 1E-3
		// Order 4: log2(1+y) = (((a*y+b)*y+c)*y + 1-a-b-c)*y, a=-0.0803 b=0.3170 c=-0.6748
		//  => max dev = +/- 1.4E-4
		// Order 5: log2(1+y) = ((((a*y+b)*y+c)*y+d)*y + 1-a-b-c-d)*y, a=0.0440047 b=-0.1903190 c=0.4123442 d=-0.7077702
		//  => max dev = +/- 2.1E-5

		const simde__m256i CONST32_0x7f = simde_mm256_set1_epi32(0x7f);
		const simde__m256i CONST32_0x7fffff = simde_mm256_set1_epi32(0x7fffff);
		const simde__m256i CONST32_0x3f800000 = simde_mm256_set1_epi32(0x3f800000);

		const simde__m256 CONST32_1f = simde_mm256_set1_ps(1.0);

		// const float a=0.1564, b=-0.5773, c=1.0-a-b;  // third order
		const float a=0.0440047f, b=-0.1903190f, c=0.4123442f, d=-0.7077702f, e=1.0f-a-b-c-d; // fifth order
		const simde__m256  CONST32_A = simde_mm256_set1_ps(a);
		const simde__m256  CONST32_B = simde_mm256_set1_ps(b);
		const simde__m256  CONST32_C = simde_mm256_set1_ps(c);
		const simde__m256  CONST32_D = simde_mm256_set1_ps(d);
		const simde__m256  CONST32_E = simde_mm256_set1_ps(e);
		simde__m256i E; // exponents of X
		simde__m256 R; //  result

		E = simde_mm256_srli_epi32((simde__m256i) X, 23);    // shift right by 23 bits to obtain exponent+127, sign bit is zero, because this only works with positive numbers
		E = simde_mm256_sub_epi32(E, CONST32_0x7f);     // subtract 127 = 0x7f
		X = (simde__m256) simde_mm256_and_si256((simde__m256i) X, CONST32_0x7fffff);  // mask out exponent => mantisse
		X = (simde__m256) simde_mm256_or_si256 ((simde__m256i) X, CONST32_0x3f800000); // set exponent to 127 (i.e., 0)
		X = simde_mm256_sub_ps(X, CONST32_1f);          // subtract one from mantisse
		R = simde_mm256_mul_ps(X, CONST32_A);           // R = a*X
		R = simde_mm256_add_ps(R, CONST32_B);           // R = a*X+b
		R = simde_mm256_mul_ps(R, X);                   // R = (a*X+b)*X
		R = simde_mm256_add_ps(R, CONST32_C);           // R = (a*X+b)*X+c
		R = simde_mm256_mul_ps(R, X);                   // R = ((a*X+b)*X+c)*X
		R = simde_mm256_add_ps(R, CONST32_D);           // R = ((a*X+b)*X+c)*X+d
		R = simde_mm256_mul_ps(R, X);                   // R = (((a*X+b)*X+c)*X+d)*X
		R = simde_mm256_add_ps(R, CONST32_E);           // R = (((a*X+b)*X+c)*X+d)*X+e
		R = simde_mm256_mul_ps(R, X);                   // R = ((((a*X+b)*X+c)*X+d)*X+e)*X ~ log2(1+X) !!
		R = simde_mm256_add_ps(R, simde_mm256_cvtepi32_ps(E));  // convert integer exponent to float and add to mantisse
		return R;
	}

    static type log(type x) {
        auto log2_const = set(0.69314718056f);
        return mul(log2(x), log2_const);
    }

	static type add(type a, type b) {
		return simde_mm256_add_ps(a,b);
	}

	static type mul(type a, type b) {
		return simde_mm256_mul_ps(a,b);
	}

	static type sub(type a, type b) {
		return simde_mm256_sub_ps(a, b);
	}

    static type div(type a, type b) {
        return simde_mm256_div_ps(a, b);
    }

	static type max(type a, type b) {
		return simde_mm256_max_ps(a,b);
	}

	static type load(const float *ptr) {
		return simde_mm256_load_ps(ptr);
	}

    static type loadU(const float *ptr) {
        return simde_mm256_loadu_ps(ptr);
    }

	static void store(float *ptr, type a) {
		simde_mm256_store_ps(ptr, a);
	}

    static void storeU(float *ptr, type a) {
        simde_mm256_storeu_ps(ptr, a);
    }
	
	static type set(float a) {
		return simde_mm256_set1_ps(a);
	}
};

template<>
struct SIMD<double> {
	//Type of a double register
	using type = simde__m256d;

	using intType = simde__m256i;

	//Number of doubles in a double register
	static const unsigned int count = 4;

	static type floor(type a) {
		return simde_mm256_floor_pd(a);
	}

	//https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
	//  Only works for inputs in the range: [-2^51, 2^51]
	static intType toInt_fast(type x){
		x = simde_mm256_add_pd(x, simde_mm256_set1_pd(0x0018000000000000));
		return simde_mm256_sub_epi64(
				simde_mm256_castpd_si256(x),
				simde_mm256_castpd_si256(simde_mm256_set1_pd(0x0018000000000000))
		);
	}

	static intType toIntBitcast(type a) {
		return simde_mm256_castpd_si256(a);
	}

	static type toFloatBitcast(intType a) {
		return simde_mm256_castsi256_pd(a);
	}

	/*
	 * Works only int the interval [-2^31, 2^31-1]
	 */
	static type toFloat_fast(intType a) {

		intType idx = simde_mm256_set_epi32(7,5,3,1,6,4,2,0);

		intType e = simde_mm256_permutevar8x32_epi32(a, idx);

		simde__m128i ee = simde_mm256_castsi256_si128(e);

		return simde_mm256_cvtepi32_pd(ee);  // convert to double
	}

	static type comp_less(type a, type b) {
		return simde_mm256_cmp_pd(a, b, SIMDE_CMP_LT_OQ);
	}

	static type comp_greater(type a, type b) {
		return simde_mm256_cmp_pd(a,b, SIMDE_CMP_GT_OQ);
	}

	static intType shift_left(intType a, unsigned int shift) {
		return simde_mm256_slli_epi64(a, shift);
	}

	static intType shift_right_logical(intType a, unsigned int shift) {
		return simde_mm256_srli_epi64(a, shift);
	}

	static type bit_and(type a, type b) {
		return simde_mm256_and_pd(a, b);
	}

	static intType bit_and(intType a, intType b) {
		return simde_mm256_and_si256(a, b);
	}

	static type bit_or(type a, type b) {
		return simde_mm256_or_pd(a,b);
	}

	static intType bit_or(intType a, intType b) {
		return simde_mm256_or_si256(a, b);
	}

    static type bit_and_not(type a, type b) {
        return simde_mm256_andnot_pd(a, b);
    }

    static type bit_xor(type a, type b) {
        return simde_mm256_xor_pd(a, b);
    }

    static type flip_sign(type a) {
        return bit_and_not(a, set(-0.0));
    }

    static type abs(type a) {
        return bit_and_not(a, set(-0.0));
    }

	static type blendv(type a, type b, type mask) {
		return simde_mm256_blendv_pd(a, b, mask);
	}

	static type exp2(type x) {

		/*
		 * Approximates the pow2(x) of double precision numbers
		 *
		 * decomposes x:= y1 + y2, where y1 := floor(x) and y2 = 0.zzzzzzz.
		 * Uses a polynomial approximation of f := pow2(y2) f: [0, 1[ -> [1, 2[ gives the mantisse of the result
		 * Note: f(0) = 1 and f(1) = 2, therefore h = 1 and g = 2-a-b-c-d-e-f-g-1
		 * Maximum deviation: 4.2e-9
		 */

		// mask out edge cases
		// if x > DBL_MAX_EXP 2^x -> inf, if x < DBL_MIN_EXP 2^x -> 0
		const type c_max_exp = set(DBL_MAX_EXP);
		const type c_min_exp = set(DBL_MIN_EXP);
		const type c_max = set(INFINITY);
		const type c_max_mask = comp_greater(x, c_max_exp);
		const type c_min = set(0);
		const type c_min_mask = comp_less(x, c_min_exp);

		const type c_1d = set(1);
		const intType c_1023 = setInt(1023); // 1023 is the offset of the exponent in double representation
		const intType c_mantissa_mask = setInt(0xfffffffffffff); // the 52 bits of the double mantissa set to 1

		// 6th order polynomial coefficients
		const type poly_a = set(0.0002187767014305746);
		const type poly_b = set(0.0012388813954882880);
		const type poly_c = set(0.0096843277474313091);
		const type poly_d = set(0.0554806806423937746);
		const type poly_e = set(0.2402303737183841825);
		const type poly_f = set(0.6931469597948718420);

		// decompose x = y1 + y2, where y1 := floor(x) and y2 := 1.zzzzzzzzz

		const type floor = simde_mm256_floor_pd(x);
		const intType y1 = toInt_fast(floor);
		const type y2 = sub(x, floor);

		// calculate the polynomial approximation f(y2) ~ 2^y.
		// mant := f(y2) = ((((((a x y2 + b)*y2 + c)*y2 + d)*y2 + e)*y2 + f)*y2 + g)*y2 + h
		type mant;
		mant = mul(y2, poly_a);
		mant = add(poly_b, mant);
		mant = mul(y2, mant);
		mant = add(poly_c, mant);
		mant = mul(y2, mant);
		mant = add(poly_d, mant);
		mant = mul(y2, mant);
		mant = add(poly_e, mant);
		mant = mul(y2, mant);
		mant = add(poly_f, mant);
		mant = mul(y2, mant);
		mant = add(mant, c_1d);

		// assemble the double number by putting together mantissa and exponent
		const intType mantissa_long = bit_and(toIntBitcast(mant), c_mantissa_mask); // zero out everything but the mantissa digits

		const intType exp_i64 = add(y1, c_1023); // double exponent is stored with an offset of 1023

		const intType shifted_exp = shift_left(exp_i64, 52); // double mantissa has 52 bits
		x = toFloatBitcast(bit_or(mantissa_long, shifted_exp)); // join mantisse and exponent and obtain the final result 2^x


		x = blendv(x, c_max, c_max_mask);
		x = blendv(x, c_min, c_min_mask);

		return x;
	}

    static type exp(type x) {
        auto log2e = set(1.44269504089);
        return exp2(mul(x, log2e));
    }

	static type log2(type x) {

		/*
		 * Approximates the log2(x) of double precision numbers
		 *
		 * Based on: log2[2^e * 1.m] = e + log2[1.m]
		 * Uses a polynomial approximation of f := log2(x+1) the interval of [1;2] for the log2 of the mantissa 1.m
		 * Note: f(0) = 0 and f(1) = 1, therefore j = 0 and i = 1-a-b-c-d-e-f-g-h
		 * Maximum deviation: 1.3e-8
		 */

		// define coefficients for fitted polynomial of x+1.
		const type poly_a = set(0.00539574483271335);
		const type poly_b = set(-0.033134075405641866);
		const type poly_c = set(0.09571929135783046);
		const type poly_d = set(-0.18043327446159182);
		const type poly_e = set(0.26625227022774905);
		const type poly_f = set(-0.3553426744739997);
		const type poly_g = set(0.4801415033950581);
		const type poly_h = set(-0.7212923532638644);
		const type poly_i = set(1.4426935677917467);

		const intType c_1023 = setInt(0x3ff);
		const intType c_mantissa_mask = setInt(0xfffffffffffff);
		const intType c_exp_1023 = setInt(0x3ff0000000000000);

		const type const_1d = set(1);

		type R;
		intType e;

		// can be done with AVX2
		e = shift_right_logical(toIntBitcast(x), 52);
		e = sub(e, c_1023);

		x = bit_and(x, toFloatBitcast(c_mantissa_mask));  // zero out exponent
		x = bit_or(x, toFloatBitcast(c_exp_1023));         // set exponent to 1023 (1023 - 1023 = 0)

		x = sub(x, const_1d);         // subtract one from mantisse
		R = mul(x, poly_a);           // R = a*X
		R = add(R, poly_b);           // R = a*X+b
		R = mul(R, x);                // R = (a*X+b)*X
		R = add(R, poly_c);           // R = (a*X+b)*X+c
		R = mul(R, x);                // R = ((a*X+b)*X+c)*X
		R = add(R, poly_d);           // R = ((a*X+b)*X+c)*X+d
		R = mul(R, x);                // R = (((a*X+b)*X+c)*X+d)*X
		R = add(R, poly_e);           // R = (((a*X+b)*X+c)*X+d)*X+e
		R = mul(R, x);                // R = ((((a*X+b)*X+c)*X+d)*X+e)*X
		R = add(R, poly_f);           // R = ((((a*X+b)*X+c)*X+d)*X+e)*X+f
		R = mul(R, x);                // R = (((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X
		R = add(R, poly_g);           // R = (((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g
		R = mul(R, x);                // R = ((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X
		R = add(R, poly_h);           // R = (((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)
		R = mul(R, x);                // R = ((((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)*X)
		R = add(R, poly_i);           // R = ((((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)*X)+i
		R = mul(R, x);                // R = (((((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)*X)+i)*X ~ log2(1+X) !!

		return add(R, toFloat_fast(e));
	}

    static type log(type x) {
        auto log2_const = set(0.69314718056);
        return mul(log2(x), log2_const);
    }

	static type add(type a, type b) {
		return simde_mm256_add_pd(a,b);
	}

	static intType add(intType a, intType b) {
		return simde_mm256_add_epi64(a, b);
	}

	static type mul(type a, type b) {
		return simde_mm256_mul_pd(a,b);
	}

	static type sub(type a, type b) {
		return simde_mm256_sub_pd(a, b);
	}

	static intType sub(intType a, intType b) {
		return simde_mm256_sub_epi64(a, b);
	}

	static type max(type a, type b) {
		return simde_mm256_max_pd(a,b);
	}

	static type load(const double *ptr) {
		return simde_mm256_load_pd(ptr);
	}

    static type loadU(const double *ptr) {
        return simde_mm256_loadu_pd(ptr);
    }

	static void store(double *ptr, type a) {
		simde_mm256_store_pd(ptr, a);
	}

    static void storeU(double *ptr, type a) {
        simde_mm256_storeu_pd(ptr, a);
    }

	static type set(double a) {
		return simde_mm256_set1_pd(a);
	}

	static intType setInt(int64_t a) {
		return simde_mm256_set1_epi64x(a);
	}
};


//Returns a number bigger or equal to size which divides SIMD<F>::count i.e the number of floating points in a register
template<typename F>
unsigned int getSimdSize(unsigned int size) {
	return ((size + SIMD<F>::count - 1) / SIMD<F>::count) * SIMD<F>::count;
}

template<typename F>
F *allocAligned(unsigned int numberOfFloatingPoints) {
	return (F *) std::aligned_alloc(32, getSimdSize<F>(numberOfFloatingPoints) * sizeof(F));
}

template<typename F>
F *allocAlignedRaw(unsigned int numberOfFloatingPoints) {
	return (F *) std::aligned_alloc(32, numberOfFloatingPoints * sizeof(F));
}

template<typename F>
void freeAligned(F *ptr) {
	std::free(ptr);
}

template<typename F>
void avx_logSumExp(F *dest, const F *AiAjEdges, unsigned int AiAj, unsigned int edges) {
	assert(AiAj % SIMD<F>::count == 0);
	for(unsigned int i = 0; i < AiAj; i+=SIMD<F>::count) {
		const F *begin = AiAjEdges + i;
		auto max = SIMD<F>::load(begin);
		for(unsigned int e = 1; e < edges; e++) {
			max = SIMD<F>::max(max, SIMD<F>::load(begin + AiAj*e));
		}
		auto sum = SIMD<F>::set(0.0);
		for(unsigned int e = 0; e <edges; e++) {
			auto temp = SIMD<F>::load(begin + AiAj*e);

			auto sub = SIMD<F>::sub(temp, max);

			auto exp = SIMD<F>::exp2(sub);

			sum = SIMD<F>::add(sum, exp);
		}
		auto log = SIMD<F>::log2(sum);
		SIMD<F>::store(dest + i, SIMD<F>::add(max, log));
	}
}

#endif
