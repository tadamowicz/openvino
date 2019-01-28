//
// Copyright 2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <list>
#include <unordered_set>

#include <quantization/quantized_layer_params.hpp>
#include "gna_plugin.hpp"
#include "gna_layer_info.hpp"


using namespace InferenceEngine;
using namespace GNAPluginNS;

void GNAPlugin::insertDiagonalLayer(std::vector<CNNLayerPtr> & layers) {
    int numOfDiagLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayer(l);
        if (LayerInfo(l).isActivation()) {
            if (LayerInfo(prevLayer).has32BOutput())
                continue;
        } else {
            auto eltwise = dynamic_cast<InferenceEngine::EltwiseLayer *>(l.get());
            if (!eltwise) {
                continue;
            }
            // in case of eltwise sum one of input would be 4 bytes one - 2
            // in case of eltwise mull one of input would be 2 bytes one - 2
            // for e sum if we have 4-4 inputs we will handle that by inserting identity activation
            // for e sum if we have 4-2 - OK
            // for e sum if we have 2-2 inputs we need to insert diagonal -- handling here
            // for e mul if we have 2-2 - OK
            // for e mul if we have 2-4 - inputs we need to insert identity to put 4 bytes input into weights
            // for e mul if we have 4-4 - inputs we need to insert 2 identities to put both 4 bytes input into weights

            if (eltwise->_operation != EltwiseLayer::Sum)
                continue;

            auto prevLayer1 = CNNNetPrevLayer(l, 1);
            if (!LayerInfo(prevLayer).has16BOutput() || !LayerInfo(prevLayer1).has16BOutput())
                continue;
        }

#ifdef PLOT
        std::cout << "Inserted Diagonal Layer between: " << prevLayer->name << " and " << l->name << "\n" << std::flush;
#endif
        // actual insertion
        auto diagName = std::string("SyntheticScaleShift_") + std::to_string(numOfDiagLayers++);
        auto diagLayer = std::make_shared<ScaleShiftLayer>(LayerParams({diagName, "ScaleShift", Precision::FP32}));

        // TODO: diagonal size
        std::vector<float> arrayOf1(l->outData[0]->dims[0], 1.f);
        diagLayer->_weights = make_shared_blob<float>(l->outData[0]->precision, Layout::C, arrayOf1);
        auto newDims = l->outData[0]->dims;
        auto dataPtr = std::make_shared<Data>(diagName,
                                              newDims,
                                              l->outData[0]->precision,
                                              l->outData[0]->layout);

        auto diagonalWithQuant = quantized ?
                            InferenceEngine::injectData<QuantizedLayerParams>(diagLayer) :
                                                                                    diagLayer;

        dataPtr->creatorLayer = diagonalWithQuant;
        diagonalWithQuant->outData.push_back(dataPtr);
        CNNNetworkInsertLayer(prevLayer, l, diagonalWithQuant);
    }
}

void GNAPlugin::reorderMaxPool(std::vector<InferenceEngine::CNNLayerPtr> & layers) {
    // detecting following pattern
    // conv->relu->maxpooling
    // changing it to conv->mxpooling->relu
    for (auto & l : layers) {
        auto pool = LayerInfo(l);
        if (!pool.isMaxPooling()) continue;

        // checking prev layer type
        auto activation = LayerInfo(CNNNetPrevLayer(l));
        if (!activation.isActivation()) continue;

        // if activation came from convolution
        auto convolution = LayerInfo(CNNNetPrevLayer(static_cast<InferenceEngine::CNNLayer*>(activation)));
        if (!convolution.isConvolution()) continue;

        gnalog() << "MaxPooling: " << pool << ", reordered with activation: " << activation << "\n";

        CNNNetSwapLayers(activation, pool);
    }
}

std::vector<CNNLayerPtr> GNAPlugin::getCandidatesForIdentityInsertion(const CNNLayerPtr l) {
    std::vector<CNNLayerPtr> prevLayers;

    // skipping memory inputs and true inputs layers
    if (l->insData.empty()) return {};

    auto eltwise = dynamic_cast<InferenceEngine::EltwiseLayer *>(l.get());
    auto concat = dynamic_cast<InferenceEngine::ConcatLayer *>(l.get());

    // eltwise
    if (eltwise != nullptr) {
        // eltwise layer has 2 inputs, so depends on situation identity should or should not be inserted

        // for  sum if we have 4-4 inputs we will handle that by inserting identity activation case (1)
        // for  sum if we have 4-2 - OK
        // for  sum if we have 2-2 inputs we need to insert diagonal

        // for  mul if we have 2-2 - OK
        // for  mul if we have 2-4 - inputs we need to insert identity activation to make 2 bytes input
        // for  mul if we have 4-4 - inputs we need to insert 2 identities activations  to put 2 bytes input and weights
        auto prev0 = CNNNetPrevLayer(l, 0);
        auto prev1 = CNNNetPrevLayer(l, 1);
        switch (eltwise->_operation) {
            case EltwiseLayer::Sum:
                if (!LayerInfo(prev0).has32BOutput() || !LayerInfo(prev1).has32BOutput()) {
                    return prevLayers;
                }
                // TODO: wether there - are possibility to select what layer to quantize
                prevLayers.push_back(prev0);
                break;
            case EltwiseLayer::Prod:
                if (LayerInfo(prev0).has16BOutput() && LayerInfo(prev1).has16BOutput()) {
                    return prevLayers;
                }

                if (LayerInfo(prev0).has32BOutput()) {
                    prevLayers.push_back(prev0);
                }

                if (LayerInfo(prev1).has32BOutput()) {
                    prevLayers.push_back(prev1);
                }

                break;
            default :
                THROW_GNA_EXCEPTION << "Eltwise Layer of type: " << eltwise->_operation << " not supported";
        }
    } else if (concat != nullptr) {
        for (int i = 0; CNNNetHasPrevLayer(l.get(), i); ++i) {
            auto prev = CNNNetPrevLayer(l, i);
            if (LayerInfo(prev).has32BOutput()) {
                prevLayers.push_back(prev);
            }
        }
    } else {  // not eltwise or concat
        // other layers has 1 inputs - situation is easier
        // ex. activation or pooling - no need to insert identity activation.
        if (LayerInfo(l).has32BInput())
            return prevLayers;

        auto prevLayer = CNNNetPrevLayer(l);
        if (!LayerInfo(prevLayer).has32BOutput())
            return prevLayers;

        prevLayers.push_back(prevLayer);
    }
    return prevLayers;
}

void GNAPlugin::substitutePRelu(std::vector<InferenceEngine::CNNLayerPtr> &layers) {
    auto getScale = [](CNNLayer* layer) {
        auto powerCandidate = LayerInfo(layer);
        if (!powerCandidate.isPower()) return 0.0f;
        auto power = powerCandidate.as<PowerLayer*>();

        return power->power == 1 && power->offset == 0.0f ? power->scale : 0.0f;
    };

    auto isScale = [getScale](CNNLayer* layer) {
        return getScale(layer) != 0.0f;
    };

    auto isNegate = [getScale](CNNLayer* layer) {
        return getScale(layer) == -1.0f;
    };

    auto getNext = [](CNNLayer* layer) {
        CNNLayer* next = nullptr;
        if (layer == nullptr) return next;
        if (layer->outData.size() != 1) return next;
        return layer->outData[0]->inputTo.begin()->second.get();
    };

    // TODO: unit tests for bad cases
    for (auto & l : layers) {
        // assume l is starting layer, that is followed by eltwise_sum(relu, negate/relu/scale/negate)
        if (l->outData.size() != 1) continue;
        auto &outputLayers = l->outData[0]->inputTo;
        if (outputLayers.size() != 2) continue;

        // one of followed layers need to be generic relu
        auto first = LayerInfo(outputLayers.begin()->second);
        auto second = LayerInfo((++outputLayers.begin())->second);

        auto relu1 = outputLayers.begin()->second;
        auto neg1 = (++outputLayers.begin())->second;
        if (second.isRelu()) {
            std::swap(first, second);
            std::swap(relu1, neg1);
        }
        if (!first.isRelu()) continue;
        // now we have relu as first layer, lets check second
        // negate
        if (!isNegate(neg1.get())) continue;

        // relu
        auto relu2 = getNext(second);
        if (!LayerInfo(relu2).isRelu()) continue;

        // scale
        auto scale = getNext(relu2);
        if (!isScale(scale)) continue;

        // negate2
        auto negate = getNext(scale);
        if (!isNegate(negate)) continue;

        // sum
        auto sum = getNext(negate);
        if (!LayerInfo(sum).isEltwiseSum()) continue;
        if (sum->insData.size() != 2) continue;

        auto s1 = sum->insData[0].lock()->creatorLayer.lock().get();
        auto s2 = sum->insData[1].lock()->creatorLayer.lock().get();

        if (s1 != static_cast<InferenceEngine::CNNLayer *>(first) &&
            s2 != static_cast<InferenceEngine::CNNLayer *>(first)) {
            continue;
        }

        // hurray we found parametric relu group - dont know what to do with it though
        gnalog() << "PRelu with negative slope of " << -LayerInfo(scale).as<PowerLayer*>()->scale << " found" << std::endl;

        // removing all layers references except of relu layer
        outputLayers.clear();
        outputLayers[relu1->name] = relu1;
        // pointing relu to output of eltwise_summ
        relu1->outData = sum->outData;
        // changing creator layer
        relu1->outData[0]->creatorLayer = relu1;
        // pointing back to relu if any
        if (!relu1->outData[0]->inputTo.empty()) {
            auto summOutputLayer = relu1->outData[0]->inputTo.begin()->second;
            summOutputLayer->insData.clear();
            summOutputLayer->insData.push_back(relu1->outData[0]);
        }

        // changing negative slope
        first.as<ReLULayer*>()->negative_slope = LayerInfo(scale).as<PowerLayer*>()->scale;
    }
}

void GNAPlugin::reversePermutations(std::vector<CNNLayerPtr> &layers) {
    std::function<CNNLayerPtr(CNNLayerPtr, std::function<bool(CNNLayerPtr)>)> prevLayerSkipCertain
        = [&prevLayerSkipCertain](CNNLayerPtr layer, std::function<bool(CNNLayerPtr)> shouldSkip) -> CNNLayerPtr {
        if (CNNNetHasPrevLayer(layer.get())) {
            return nullptr;
        }
        auto prev = CNNNetPrevLayer(layer);

        if (!shouldSkip(prev)) return prevLayerSkipCertain(prev, shouldSkip);

        return prev;
    };

    auto prevLayerSkipReshape = [&prevLayerSkipCertain](CNNLayerPtr layer) -> CNNLayerPtr {
        return prevLayerSkipCertain(layer, [] (CNNLayerPtr l2) {
            return LayerInfo(l2).isReshape();
        });
    };


    std::function<CNNLayerPtr(CNNLayerPtr)> nextLayerSkipReshape = [&nextLayerSkipReshape](CNNLayerPtr layer) -> CNNLayerPtr {
        if (layer->outData.empty()) {
            return nullptr;
        }
        if (layer->outData.front()->inputTo.size() != 1) {
            return nullptr;
        }
        auto next = layer->outData.front()->inputTo.begin()->second;

        if (LayerInfo(next).isReshape()) return nextLayerSkipReshape(next);

        return next;
    };

    auto prevConv = [&prevLayerSkipCertain](CNNLayerPtr layer) -> CNNLayerPtr {
        return prevLayerSkipCertain(layer, [] (CNNLayerPtr l2) {
            return
                LayerInfo(l2).isReshape() ||
                LayerInfo(l2).isPooling() ||
                LayerInfo(l2).isActivation();
        });
    };

    std::unordered_set<std::string> affineWithPermutedWeights;
    std::list<CNNLayerPtr> permutationstoRemove;

    for (auto & l : layers) {
        if (!LayerInfo(l).isPermute()) {
            continue;
        }

        auto layerOrder = l->GetParamAsInts("order");

        if (layerOrder != std::vector<int>({0, 3, 2, 1})) {
            THROW_GNA_EXCEPTION << "Unsupported permute layer: " << l->name << ", order: was " << l->GetParamAsString("order") <<
                               ", but support order is 0,3,2,1";
        }

        // search for it's input convolution
        auto prev = prevConv(l);

        // pooling no used in speech models without convolution
        if (!prev) {
            THROW_GNA_EXCEPTION << "Unsupported permute layer: " << l->name << " no valid input to that layer";
        }

        // we can remove that permutation if it is input to ScaleShift or FC layer
        auto next = nextLayerSkipReshape(l);
        if (!next || !LayerInfo(next).isFullyConnected()) {
            THROW_GNA_EXCEPTION << "Unsupported permute layer: " << l->name << " no valid output of that layer";
        }

        permutationstoRemove.push_back(l);

        // removing that permutation layer and saving information about affine
        affineWithPermutedWeights.insert(next->name);
    }

    for (auto && toRemove : permutationstoRemove) {
        CNNNetworkRemoveLayer(toRemove);
    }

    // search for conv->affine sequences
    for (auto & l : layers) {
        if (!LayerInfo(l).isFullyConnected() || 0 != affineWithPermutedWeights.count(l->name)) {
            continue;
        }
        // found an affine layer that not involved in permutations removing
        // searching whether it has direct input from convolution
        auto prevConvLayer = prevConv(l);
        if (!prevConvLayer) continue;

        auto directPrev = CNNNetPrevLayer(l);

        // TODO : make new permute
        CNNNetworkInsertLayer(l, directPrev, CNNLayerPtr(nullptr));
    }
}

void GNAPlugin::insertIdentityLayer(std::vector<CNNLayerPtr> &layers) {
    int numOfIdentityLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        for (auto && prev : getCandidatesForIdentityInsertion(l)) {
            // actual insertion
            auto activationName = std::string("identity_") + std::to_string(numOfIdentityLayers++);

            gnalog() << "Inserted "<< activationName << " between: " << prev->name << " and " << l->name << "\n" << std::flush;

            CNNLayerPtr activationLayer =
                std::make_shared<GenericLayer>(LayerParams({activationName, "identity", Precision::FP32}));
            auto inputData = l->insData[0].lock();
            auto newDims = inputData->dims;
            std::reverse(begin(newDims), end(newDims));

            auto dataPtr = std::make_shared<Data>("FullyConnected",
                                                  TensorDesc(inputData->precision,
                                                             newDims,
                                                             inputData->layout));
            auto activationLayerWithQuant = quantized ?
                                    InferenceEngine::injectData<QuantizedLayerParams>(activationLayer) :
                                                                                            activationLayer;
            dataPtr->creatorLayer = activationLayerWithQuant;
            activationLayerWithQuant->outData.push_back(dataPtr);
            // wether 1 identity or all outputs TODO possible grouping here, need to implement special groupped inserter
            bool notAll = false;
            for (auto && nextData  : prev->outData) {
                for (auto && nextLayer : nextData->inputTo) {
                    if (nextLayer.second.get() == l.get())
                        continue;
                    if (getCandidatesForIdentityInsertion(nextLayer.second).empty()) {
                        notAll = true;
                    }
                }
            }

            CNNNetworkInsertLayer(prev, notAll ? l : CNNLayerPtr(nullptr), activationLayerWithQuant);
        }
    }
}

void GNAPlugin::insertCopyLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers) {
    int numCopyLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayer(l);
        if ((LayerInfo(l).isMemory() && LayerInfo(prevLayer).isConcat()) ||
            (LayerInfo(l).isConcat() && LayerInfo(prevLayer).isCrop())) {
            if (LayerInfo(prevLayer).isCrop()) {
                auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (prevLayer.get());
                size_t cropOffset = cropLayer->offset.back() * cropLayer->precision.size();
                if (ALIGN(cropOffset, 8) != cropOffset) {
                    // The crop will be replced by affine.
                    // Copy layer insertion is not required
                    continue;
                }
            }
            std::string copyName = std::string("copy_") + std::to_string(numCopyLayers++);
            gnalog() << "Inserted "<< copyName << " between: " << l->name << " and " << prevLayer->name << "\n" << std::flush;

            CNNLayerPtr copyLayer =
            std::make_shared<GenericLayer>(LayerParams({copyName, "Copy", Precision::FP32}));

            auto inputData = l->insData[0].lock();
            auto newDims = inputData->dims;

            std::reverse(begin(newDims), end(newDims));

            auto dataPtr = std::make_shared<Data>(copyName,
                                                  TensorDesc(inputData->precision,
                                                             newDims,
                                                             inputData->layout));

            auto copyWithQuant = quantized ?
                                    InferenceEngine::injectData<QuantizedLayerParams>(copyLayer) :
                                                                                            copyLayer;
            dataPtr->creatorLayer = copyWithQuant;
            copyWithQuant->outData.push_back(dataPtr);
            CNNNetworkInsertLayer(prevLayer, l, copyWithQuant);
        }
    }
}

void GNAPlugin::insertAligningFilterLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers) {
    int numOfFilterLayers = 0;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layers.front());
    for (auto & l : layers) {
        if (l->insData.empty()) continue;
        auto prevLayer = CNNNetPrevLayer(l);
        if (LayerInfo(prevLayer).isSplit() || LayerInfo(prevLayer).isSlice()) {
            // if the input is a predecessor layer, we don't need to insert a filter
            auto splitInput = prevLayer->insData.begin()->lock();
            auto ptrSplitLayerInput = splitInput->creatorLayer.lock();

            // find this split and check aligning of each output layer
            size_t layers_size = 0;
            size_t aligned_layers_size = 0;
            size_t filter_num_rows_in = 0;
            size_t data_index = 0;

            // Special case of eltwise, because it can join two parts of the same split
            if (LayerInfo(l).isEltwise()) {
                auto input1 = l->insData[0].lock();
                auto input2 = l->insData[1].lock();
                uint8_t precision = quantized ? 2 : prevLayer->outData[data_index]->precision.size();
                if (input1->name.substr(0, input1->name.find(".")) ==
                                input2->name.substr(0, input2->name.find("."))) {
                    auto& dataOutput = prevLayer->outData[data_index];
                    auto split_1st_output_size = InferenceEngine::details::product(begin(dataOutput->dims),
                                                             end(dataOutput->dims)) * precision;

                    if (split_1st_output_size != ALIGN64(split_1st_output_size)) {
                        size_t aligned64_prev_offset = std::max(0, static_cast<int>(ALIGN64(split_1st_output_size) - 64));
                            filter_num_rows_in =
                                            (split_1st_output_size - aligned64_prev_offset) / precision
                                                    + InferenceEngine::details::product(begin(dataOutput->dims),
                                                                                        end(dataOutput->dims));
                        data_index = prevLayer->outData.size();
                    } else {
                        return;
                    }
                }
            }

            for (; data_index < prevLayer->outData.size(); ++data_index) {
                auto& dataOutput = prevLayer->outData[data_index];

                if (!dataOutput) {
                    THROW_GNA_EXCEPTION << "Output layer pointer for split/slice is unexpectedly absent";
                }
                for (auto&& ptrSplitLayerOutputPair : dataOutput->getInputTo()) {
                    auto& ptrSplitLayerOutput = ptrSplitLayerOutputPair.second;
                    if (!ptrSplitLayerOutput) {
                        THROW_GNA_EXCEPTION << "Output layer for split/slice is unexpectedly absent";
                    }
                    if (ptrSplitLayerOutput->name == l->name) {
                        if (layers_size != aligned_layers_size) {
                            size_t aligned64_prev_offset = std::max(0, static_cast<int>(aligned_layers_size - 64));
                            filter_num_rows_in =
                                            (layers_size - aligned64_prev_offset) / dataOutput->precision.size()
                                                    + InferenceEngine::details::product(begin(dataOutput->dims),
                                                                                        end(dataOutput->dims));
                            break;
                        }
                        return;
                    } else if (data_index == 0) {
                        // This is the first layer in this split output port
                        auto split_1st_output_size = InferenceEngine::details::product(begin(dataOutput->dims),
                                                             end(dataOutput->dims)) * dataOutput->precision.size();
                        layers_size += split_1st_output_size;
                        aligned_layers_size += ALIGN64(split_1st_output_size);
                    }
                }
            }

#ifdef PLOT
        std::cout << "Inserted Affine Filter Layer between: " << prevLayer->name << " and " << l->name << "\n" << std::flush;
#endif
            // insert the filter
            auto filterName = std::string("SyntheticAffineFilter_") + std::to_string(numOfFilterLayers++);
            auto filterLayer = std::make_shared<WeightableLayer>(LayerParams({filterName, "AffineFilter", Precision::FP32}));
            auto inputData = l->insData[0].lock();
            auto newDims = inputData->dims;

            auto& num_rows_out = inputData->dims[0];
            std::vector<float> filterWeights(ALIGN(filter_num_rows_in, 8) * inputData->dims[0], 0.f);
            std::vector<float> filterBiases(inputData->dims[0], 0.f);
            auto offset = filter_num_rows_in - num_rows_out;
            int out = 0;
            if (filter_num_rows_in + offset > filterWeights.size()) {
                THROW_GNA_EXCEPTION << "Weights size not correctly calculated";
            }
            for (int input = offset; input < filter_num_rows_in; ++input, ++out) {
                auto filterWeightsIt = filterWeights.begin();
                std::advance(filterWeightsIt, input + out * filter_num_rows_in);
                *filterWeightsIt = 1.0f;
            }

            filterLayer->_weights = make_shared_blob<float>(inputData->precision, Layout::C, filterWeights);

            auto& dataOutput = prevLayer->outData[1];

            std::reverse(begin(newDims), end(newDims));
            auto dataPtr = std::make_shared<Data>(filterName,
                                                TensorDesc(inputData->precision,
                                                            newDims,
                                                            inputData->layout));

            auto filterWithQuant = quantized ?
                                    InferenceEngine::injectData<QuantizedLayerParams>(filterLayer) :
                                                                                            filterLayer;
            dataPtr->creatorLayer = filterWithQuant;
            filterWithQuant->outData.push_back(dataPtr);
            CNNNetworkInsertLayer(prevLayer, l, filterWithQuant, 1);

            dataOutput->setDims({dataOutput->dims[1], filter_num_rows_in});
        }
    }
}
