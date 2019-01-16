// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <ie_common.h>
namespace GNAPluginNS {

class PassManager {
    Policy policy;

 public:
    explicit PassManager(Policy policy) noexcept : policy(policy) {}

    /**
    * @brief GNA affine layers are always have activation atached, while IR not
    */
    void insertIdentityLayer(std::vector<InferenceEngine::CNNLayerPtr> &layers);

    /**
     * @brief GNA cannot support broadcast - so we will tile weights and biases for scaleshift layer
     */
    void substituteScaleShiftBroadCast(std::vector<InferenceEngine::CNNLayerPtr> &layers);

    /**
     * @brief GNA convolution layers have deinterleaved layout, while affine one doesn't
     * so between convolution and affine layers permute layers need to be inserted,
     * current MO approach is to insert such permutations
     * since GNA-HW already support conv->affine in permuted for, this pass inverses MO behavior
     * so its remove permutations of certain form conv->conv, and between conv->affine
     * and insert permutation between conv->affine if they are missed in IR
     * @param layers
     */
    void reversePermutations(std::vector<InferenceEngine::CNNLayerPtr> &layers);

    /**
     * brief @search for specific patter in the graph (6 layers are replaced by single one)
     * @param layers
     */
    void substitutePRelu(std::vector<InferenceEngine::CNNLayerPtr> &layers);

    /**
     * diagonal layer insertion required in cases where activation followed by split layers, or any other
     * topology changing layers
     */
    void insertDiagonalLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers);

    /**
     * @brief MaxPool can be reordered with activation, on GNA there is a strategy to have conv->maxpool->activation
     * it means maxpool receives 4 bytes, and produces 4 bytes
     */
    void reorderMaxPool(std::vector<InferenceEngine::CNNLayerPtr> & layers);
    /**
     * @brief GNA doen't support multiple activations fused with functional layer
     * currently for n activations for the layer X, it will be 1 PWL identity inserted, and n diagonal layers.
     * if one of activations is already identity, n-1 diagonal layers will be inserted
     */
    void handleMultipleActivationsForTheLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers);

    /**
     * @brief copy layer insertion required in cases where input layer does not have output memory
     */
    void insertCopyLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers);

    /**
     * @brief aligned filter layer insertion required in cases when split/slice have output connections on not aligned addresses
     */
    void insertAligningFilterLayer(std::vector<InferenceEngine::CNNLayerPtr> & layers);

 protected:
    /**
     * helper injections of diagonal layer with certain value
     */
    int syntheticDiagonalLayersNum = 0;
    void insertDiagonalLayerBetween(InferenceEngine::CNNLayerPtr l1, InferenceEngine::CNNLayerPtr l2, float fillValue);
    std::vector<InferenceEngine::CNNLayerPtr> getCandidatesForIdentityInsertion(const InferenceEngine::CNNLayerPtr layer);
};

}  // namespace GNAPluginNS