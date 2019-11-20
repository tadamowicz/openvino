// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "inference_engine.hpp"

namespace GNAPluginNS {
class GNAConcatLayer {
    InferenceEngine::CNNLayerPtr concatLayer;

public:
    explicit GNAConcatLayer(InferenceEngine::CNNLayerPtr layer) :
        concatLayer(layer)
    {}

    InferenceEngine::CNNLayerPtr getConcat() { return concatLayer; }
    /**
     * pointer to gna memory request
     */
    void *gna_ptr = nullptr;
    /**
     * gna memory of this size is reserved for concat
     */
    size_t reserved_size = 0;
    bool output_allocation_flag = false;
    bool input_allocated = false;
    /**
     * gna memory of this offset from gna_ptr
     */
    struct ConcatConnectedLayerInfo {
        ConcatConnectedLayerInfo(const std::string& n,
            size_t o) :
            name(n),
            offset(o) {}
        std::string name = "";
        size_t offset = 0;
    };

    std::vector<ConcatConnectedLayerInfo> concatInputLayers;
};
}  // namespace GNAPluginNS
