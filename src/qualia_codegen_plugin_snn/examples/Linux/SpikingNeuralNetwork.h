#ifndef _SPIKINGNEURALNETWORK_H_
#define _SPIKINGNEURALNETWORK_H_

#include "NeuralNetwork.h"

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
    // Use LONG_NUMBER_T to avoid overflow when accumulating over timesteps
    std::array<MODEL_OUTPUT_LONG_NUMBER_T, MODEL_OUTPUT_SAMPLES> preds{};
    // Averaged over timesteps with original NUMBER_T
    std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> preds_avg{};

#ifdef MODEL_INPUT_TIMESTEPS
    // Accumulate over timesteps
		for (int t = 0; t < MODEL_INPUT_TIMESTEPS; t++) {
      std::transform(preds.begin(),
                     preds.end(),
                     this->run(input).begin(),
                     preds.begin(),
                     std::plus<std::remove_all_extents<output_t>::type>());
    }

		// Average over timesteps
		for (size_t j = 0; j < MODEL_OUTPUT_SAMPLES; j++){
			preds_avg[j] = preds[j] / MODEL_INPUT_TIMESTEPS;
		}
#endif
		// Some models have an internal state that must be reset between each sample
    reset();

    // De-quantize predictions to match targets for metrics computation
    std::array<metric_return_t, MODEL_OUTPUT_SAMPLES> deqpreds{};
    std::transform(preds.begin(),
                   preds.end(),
                   deqpreds.begin(),
                   [](MODEL_OUTPUT_NUMBER_T v) {
                    return static_cast<metric_return_t>(v) / (1 << MODEL_OUTPUT_SCALE_FACTOR);
                   });

    for (auto &metric: this->metrics) {
      metric->update(deqpreds, targets);
    }

    return preds_avg;
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
    // Use LONG_NUMBER_T to avoid overflow when accumulating over timesteps
    std::array<MODEL_OUTPUT_LONG_NUMBER_T, MODEL_OUTPUT_SAMPLES> preds{};
    // Averaged over timesteps with original NUMBER_T
    std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> preds_avg{};

    // Accumulate over timesteps
		for (int t = 0; t < MODEL_INPUT_TIMESTEPS; t++) {
      std::transform(preds.begin(),
                     preds.end(),
                     this->run(input_timesteps.at(t)).begin(),
                     preds.begin(),
                     std::plus<MODEL_OUTPUT_LONG_NUMBER_T>());
    }

    // Average over timesteps
    for (size_t j = 0; j < MODEL_OUTPUT_SAMPLES; j++){
      preds_avg[j] = preds[j] / MODEL_INPUT_TIMESTEPS;
    }

    reset();

    // De-quantize predictions to match targets for metrics computation
    std::array<metric_return_t, MODEL_OUTPUT_SAMPLES> deqpreds{};
    std::transform(preds.begin(),
                   preds.end(),
                   deqpreds.begin(),
                   [](MODEL_OUTPUT_NUMBER_T v) {
                    return static_cast<metric_return_t>(v) / (1 << MODEL_OUTPUT_SCALE_FACTOR);
                   });

    for (auto &metric: this->metrics) {
      metric->update(deqpreds, targets);
    }

    return preds_avg;
  }
#endif
  
};

#endif//_SPIKINGNEURALNETWORK_H_
