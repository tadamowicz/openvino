// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

#include "inference_engine.hpp"
#include "gna_plugin.hpp"

namespace GNAPluginNS {
class LayersBuilder {
    using CreatorFnc = std::function<void(GNAPlugin*, CNNLayerPtr)>;

public:
    LayersBuilder(const std::vector<std::string> &types, CreatorFnc callback) {
        for (auto && str : types) {
            getStorage()[str] = callback;
        }
    }
    static caseless_unordered_map<std::string, CreatorFnc> &getStorage() {
        static caseless_unordered_map<std::string, CreatorFnc> LayerBuilder;
        return LayerBuilder;
    }
};
}  // namespace GNAPluginNS
