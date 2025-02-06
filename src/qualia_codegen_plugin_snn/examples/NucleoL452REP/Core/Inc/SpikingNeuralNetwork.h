#ifndef _SPIKINGNEURALNETWORK_H_
#define _SPIKINGNEURALNETWORK_H_

#include "NeuralNetwork.h"

#ifdef __cplusplus
extern "C" {
#endif//__cplusplus
struct NNResult spikingNeuralNetworkInfer(const float input[]);
#ifdef __cplusplus
}

#include <algorithm>
#include <array>
#include <cstdlib>

template<size_t NMetrics = 0, typename MetricT = float, std::size_t MetricN = 0>
class SpikingNeuralNetwork : public NeuralNetwork<NMetrics, MetricT, MetricN> {
public:
  SpikingNeuralNetwork() : NeuralNetwork<NMetrics, MetricT, MetricN>() {
  }

  SpikingNeuralNetwork(std::array<Metric<MetricT, MetricN>*, NMetrics> metrics) : NeuralNetwork<NMetrics, MetricT, MetricN>(metrics) {
  }

  virtual std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> evaluate(
    const std::array<float, MODEL_INPUT_DIMS> input,
    const std::array<float, MODEL_OUTPUT_SAMPLES> targets) {
    auto preds = this->run(input);
#ifdef MODEL_INPUT_TIMESTEPS
    // Accumulate over timesteps
		for (int t = 1; t < MODEL_INPUT_TIMESTEPS; t++) {
      std::transform(preds.begin(),
                     preds.end(),
                     this->run(input).begin(),
                     preds.begin(),
                     std::plus<std::remove_all_extents<output_t>::type>());
    }

		// Average over timesteps
		for (size_t j = 0; j < MODEL_OUTPUT_SAMPLES; j++){
			preds[j] = preds[j] / MODEL_INPUT_TIMESTEPS;
		}
#endif
		// Some models have an internal state that must be reset between each sample
    reset();

    // Quantize targets to match outputs
    std::array<MODEL_INPUT_NUMBER_T, MODEL_OUTPUT_SAMPLES> q_targets{};
    std::transform(targets.begin(),
                   targets.end(),
                   q_targets.begin(),
                   [](float v) {
                    return clamp_to(MODEL_OUTPUT_NUMBER_T, (MODEL_OUTPUT_LONG_NUMBER_T)round_with_mode(v * (1 << MODEL_OUTPUT_SCALE_FACTOR), MODEL_OUTPUT_ROUND_MODE));
                   });

    for (auto &metric: this->metrics) {
      metric->update(preds, q_targets);
    }

    return preds;
  }

#ifdef MODEL_INPUT_TIMESTEP_MODE_ITERATE
  std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> evaluate_timesteps(
    const std::array<float, MODEL_INPUT_TIMESTEPS * MODEL_INPUT_DIMS> input,
    const std::array<float, MODEL_OUTPUT_SAMPLES> targets) {
    std::array<std::array<float, MODEL_INPUT_DIMS>, MODEL_INPUT_TIMESTEPS> input_timesteps;
    for (size_t i = 0; i < MODEL_INPUT_TIMESTEPS; i++) {
      for (size_t j = 0; j < MODEL_INPUT_DIMS; j++) {
        input_timesteps[i][j] = input[i * MODEL_INPUT_DIMS + j];
      }
    }

    auto preds = this->run(input_timesteps.at(0));
    // Accumulate over timesteps
		for (int t = 1; t < MODEL_INPUT_TIMESTEPS; t++) {
      std::transform(preds.begin(),
                     preds.end(),
                     this->run(input_timesteps.at(t)).begin(),
                     preds.begin(),
                     std::plus<std::remove_all_extents<output_t>::type>());
    }

    reset();

    // Quantize targets to match outputs
    std::array<MODEL_INPUT_NUMBER_T, MODEL_OUTPUT_SAMPLES> q_targets{};
    std::transform(targets.begin(),
                   targets.end(),
                   q_targets.begin(),
                   [](float v) {
                    return clamp_to(MODEL_OUTPUT_NUMBER_T, (MODEL_OUTPUT_LONG_NUMBER_T)round_with_mode(v * (1 << MODEL_OUTPUT_SCALE_FACTOR), MODEL_OUTPUT_ROUND_MODE));
                   });

    for (auto &metric: this->metrics) {
      metric->update(preds, q_targets);
    }

    return preds;
  }
#endif
};
#endif//__cplusplus

#endif//_SPIKINGNEURALNETWORK_H_
