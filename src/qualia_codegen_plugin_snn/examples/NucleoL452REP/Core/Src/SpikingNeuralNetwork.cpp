#include "SpikingNeuralNetwork.h"

#include "NeuralNetwork.h"

extern "C" {
#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif
#include "model.h"

struct NNResult spikingNeuralNetworkInfer(const float input[]) {
	static output_t outputs;

	neuralNetworkRun(input, outputs);

#ifdef MODEL_INPUT_TIMESTEPS
	static output_t new_outputs;
	
	for (int t = 1; t < MODEL_INPUT_TIMESTEPS; t++) {
#ifdef MODEL_INPUT_TIMESTEP_MODE_DUPLICATE
		neuralNetworkRun(input, new_outputs);
#elif defined(MODEL_INPUT_TIMESTEP_MODE_ITERATE)
		neuralNetworkRun(input + t * MODEL_INPUT_DIMS, new_outputs);
#else
	#error "Unknown input timestep mode"
#endif

		for (size_t j = 0; j < MODEL_OUTPUT_SAMPLES; j++) {
			outputs[j] += new_outputs[j];
		}
	}

	// Average over timesteps
	for (size_t j = 0; j < MODEL_OUTPUT_SAMPLES; j++){
		outputs[j] = outputs[j] / MODEL_INPUT_TIMESTEPS;
	}
#endif

	// Some models have an internal state that must be reset between each sample
	reset();

	// Get output class
	unsigned int label = 0;
	float max_val = outputs[0];
	for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
		if (max_val < outputs[i]) {
			max_val = outputs[i];
			label = i;
		}
	}

	inference_count++;

	return {inference_count, label, max_val};
}

}
