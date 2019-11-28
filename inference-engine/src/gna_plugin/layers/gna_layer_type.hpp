// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include "dnn.h"
#include "details/caseless.hpp"

namespace GNAPluginNS {
enum LayerType {
    Input,
    Convolution,
    ReLU,
    LeakyReLU,
    Sigmoid,
    TanH,
    Activation,
    Pooling,
    FullyConnected,
    InnerProduct,
    Reshape,
    Split,
    Slice,
    Eltwise,
    ScaleShift,
    Clamp,
    Concat,
    Const,
    Copy,
    Permute,
    Memory,
    Power,
    Crop,
    LSTMCell,
    TensorIterator,
    NO_TYPE
};

GNAPluginNS::LayerType LayerTypeFromStr(const std::string &str) {
    static const InferenceEngine::details::caseless_map<std::string, GNAPluginNS::LayerType> LayerNameToType = {
            { "Input" , Input },
            { "Convolution" , Convolution },
            { "ReLU" , ReLU },
            { "Sigmoid" , Sigmoid },
            { "TanH" , TanH },
            { "Pooling" , Pooling },
            { "FullyConnected" , FullyConnected },
            { "InnerProduct" , InnerProduct},
            { "Split" , Split },
            { "Slice" , Slice },
            { "Eltwise" , Eltwise },
            { "Const" , Const },
            { "Reshape" , Reshape },
            { "ScaleShift" , ScaleShift },
            { "Clamp" , Clamp },
            { "Concat" , Concat },
            { "Copy", Copy },
            { "Permute" , Permute },
            { "Power" , Power},
            { "Memory" , Memory },
            { "Crop" , Crop },
            { "LSTMCell", LSTMCell },
            { "TensorIterator", TensorIterator }
    };
    auto it = LayerNameToType.find(str);
    if (it != LayerNameToType.end())
        return it->second;
    else
        return NO_TYPE;
}
}  // namespace GNAPluginNS
