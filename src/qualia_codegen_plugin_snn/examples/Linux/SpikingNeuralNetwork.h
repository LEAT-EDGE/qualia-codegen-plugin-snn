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

    for (auto &metric: this->metrics) {
      metric->update(preds, targets);
    }

    return preds;
  }
};

#endif//_SPIKINGNEURALNETWORK_H_
