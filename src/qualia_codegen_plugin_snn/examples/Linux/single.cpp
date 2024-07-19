// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <type_traits>

//#include "output/number.h"
#include "NeuralNetwork.h"

int main(int argc, const char*argv[]) {
	if (argc < 2) {
		printf("Usage: %s <test vector>\n", argv[0]);
		return 1;
	}

	static float input[MODEL_INPUT_DIMS] = {0};
	static output_t output;

	for (size_t i = 0; i < MODEL_INPUT_DIMS; i++) {
		input[i] = strtof(argv[i + 1], NULL);
	}

#ifdef MODEL_INPUT_TIMESTEPS
	output_t tmp_output;
	for (size_t i = 0; i < MODEL_OUTPUT_SAMPLES; i++){
		output[i] = 0;
	}

	for (int t = 0; t < MODEL_INPUT_TIMESTEPS; t++) {
		neuralNetworkRun(input, tmp_output);

		for (size_t i = 0; i < MODEL_OUTPUT_SAMPLES; i++){
			output[i] += tmp_output[i];
		}
	}

	reset();

	for (size_t i = 0; i < MODEL_OUTPUT_SAMPLES; i++){
		output[i] = output[i] / MODEL_INPUT_TIMESTEPS;
	}
#else
	neuralNetworkRun(input, output);
#endif

	for (int i = 0; i < MODEL_OUTPUT_SAMPLES; i++) {
		if constexpr(std::is_integral_v<MODEL_INPUT_NUMBER_T>) {
			printf("%d\n", output[i]);
		} else {
			printf("%f\n", (double)output[i]);
		}
	}

	return 0;
}
