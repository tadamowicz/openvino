//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#pragma once

#include <string>
#include <memory>
#include "inference_engine.hpp"
#include "details/caseless.hpp"
#include "ie_algorithm.hpp"


namespace GNAPluginNS {

/**
 * @brief detecting of const pointer for dynamic cast operations
 * @tparam T
 */
template <class T>
struct is_const_pointer : public std::false_type{
};

template <class T>
struct is_const_pointer<const T *> : public std::true_type{
};


/**
 * similar to type traits determined in standard library this trait provides details per layer type, with some attributes specific for GNA
 * we don't need to have compile time performance for this yet
 */
class LayerInfo {
    InferenceEngine::CNNLayer * layer;

#define IS_VALID() if (nullptr == layer) return false

 public:
    explicit LayerInfo(InferenceEngine::CNNLayer & layer)
        : LayerInfo(&layer) {
    }
    explicit LayerInfo(const InferenceEngine::CNNLayerPtr & layer)
        : LayerInfo(layer.get()) {
    }
    explicit LayerInfo(InferenceEngine::CNNLayer * layer)
        : layer(layer) {
    }
    bool has16BOutput() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> layersWith16BOutputs = {"memory", "input", "split", "slice", "concat", "copy"};
        return layersWith16BOutputs.find(layer->type) != layersWith16BOutputs.end() ||
                                                                        isActivation() ||
                                                            (isCrop() && !isCropAffined());
    }
    bool has32BOutput() const noexcept {
        IS_VALID();
        static  InferenceEngine::details::caseless_set<std::string> layersWith32BOutputs =
                {"FullyConnected", "InnerProduct", "Eltwise", "ScaleShift", "Convolution", "Pooling"};
        return (layersWith32BOutputs.find(layer->type) != layersWith32BOutputs.end()) ||
                                                            (isCrop() && isCropAffined());
    }
    static bool isBatchSizeConstrained(const std::string name) {
        static InferenceEngine::details::caseless_set<std::string> layersWithConstrains = {"memory", "convolution"};
        return layersWithConstrains.find(name) != layersWithConstrains.end();
    }
    bool isActivation() const noexcept {
        IS_VALID();
        static InferenceEngine::details::caseless_set<std::string> activations = {"clamp", "sigmoid", "identity", "relu", "leakyrelu", "tanh", "prelu"};
        return activations.find(layer->type) != activations.end();
    }
    bool isRelu() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "relu");
    }
    bool isConvolution() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "convolution");
    }
    bool isPower() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "power");
    }
    bool has32BInput() const noexcept {
        IS_VALID();
        return isActivation() || isPooling();
    }
    bool isInput() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "input");
    }
    bool isEltwise() const noexcept {
        IS_VALID();
        return nullptr != as<const InferenceEngine::EltwiseLayer*>();
    }
    bool isEltwiseSum() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        return dynamic_cast<const InferenceEngine::EltwiseLayer*>(layer)->_operation ==
            InferenceEngine::EltwiseLayer::Sum;
    }
    bool isEltwiseMul() const noexcept {
        IS_VALID();
        if (!isEltwise()) return false;
        return dynamic_cast<const InferenceEngine::EltwiseLayer*>(layer)->_operation ==
            InferenceEngine::EltwiseLayer::Prod;
    }
    bool isIdentity() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "identity");
    }
    bool isFullyConnected() const noexcept {
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "FullyConnected") ||
                InferenceEngine::details::CaselessEq<std::string>()(layer->type, "InnerProduct");
    }
    bool isConvolutional() const noexcept {
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "Convolution");
    }
    bool isSplit() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "split");
    }
    bool isSlice() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "slice");
    }
    bool isConcat() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "concat");
    }
    bool isReshape() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "reshape");
    }
    bool isPermute() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "permute");
    }
    bool isPooling() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "Pooling");
    }
    bool isMaxPooling() const noexcept {
        IS_VALID();
        if (!isPooling()) return false;
        return as<const InferenceEngine::PoolingLayer*>()->_type == InferenceEngine::PoolingLayer::MAX;
    }
    bool isMemory() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "memory");
    }
    bool isCrop() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "crop");
    }
    bool isCropAffined() const noexcept {
        auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (layer);
        size_t cropOffset = cropLayer->offset.back() * cropLayer->precision.size();
        return (ALIGN64(cropOffset) != cropOffset);
    }
    bool isCopy() const noexcept {
        IS_VALID();
        return InferenceEngine::details::CaselessEq<std::string>()(layer->type, "copy");
    }
    size_t paddingSize() const noexcept {
        static InferenceEngine::details::caseless_set<std::string> layersWithPossiblePadding = {"FullyConnected",
                                                                        "InnerProduct",
                                                                             "Pooling",
                                                                         "Convolution"};
        if (layersWithPossiblePadding.find(layer->type) != layersWithPossiblePadding.end()) {
            size_t size_without_padding = 0;
            auto inputs = layer->insData.begin()->lock();
            if (inputs) {
                size_without_padding = InferenceEngine::details::product(begin(inputs->dims),
                                                                   end(inputs->dims));
            }
            return ALIGN(size_without_padding, 8) - size_without_padding;
        }
        return 0;
    }
    template <class T>
    typename std::enable_if<!is_const_pointer<T>::value, T>::type as() noexcept {
        return dynamic_cast<T>(layer);
    }
    template <class T>
    typename std::enable_if<is_const_pointer<T>::value, T>::type as() const noexcept {
        return dynamic_cast<T>(layer);
    }
    operator InferenceEngine::CNNLayer *() noexcept {
        return layer;
    }
    operator const InferenceEngine::CNNLayer *() const noexcept {
        return layer;
    }
    operator InferenceEngine::CNNLayerPtr () const noexcept {
        return std::shared_ptr<InferenceEngine::CNNLayer>(layer, [] (InferenceEngine::CNNLayer * p) {});
    }

    #undef IS_VALID
};

inline std::ostream & operator <<(std::ostream &os, const LayerInfo & info) {
    os << static_cast<const InferenceEngine::CNNLayer*>(info)->name;
    return os;
}

}  // namespace GNAPluginNS
