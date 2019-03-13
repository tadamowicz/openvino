//
// INTEL CONFIDENTIAL
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <gna-api-types-xnn.h>
#include "gna_plugin_log.hpp"
namespace GNAPluginNS {

/**
 * represent wrapper that capable to exception save pass c-objects
 * @tparam T
 */
template <class T>
class CPPWrapper {
};

template <>
class CPPWrapper<intel_nnet_type_t> {
 public:
    intel_nnet_type_t obj;

    CPPWrapper() {
        obj.nLayers = 0;
        obj.pLayers = nullptr;
        obj.nGroup = 0;
    }

    /**
     * creates nnet structure of n layers
     * @param n - number  of layers
     */
    explicit CPPWrapper(size_t n) {
        if (n == 0) {
            THROW_GNA_EXCEPTION << "Can't allocate array of intel_nnet_layer_t objects of zero length";
        }
        obj.pLayers = reinterpret_cast<intel_nnet_layer_t *>(_mm_malloc(n * sizeof(intel_nnet_layer_t), 64));
        if (obj.pLayers == nullptr) {
            THROW_GNA_EXCEPTION << "out of memory in while allocating "<< n << " GNA layers";
        }
        obj.nLayers = n;
        for (int i = 0; i < obj.nLayers; i++) {
            obj.pLayers[i].pLayerStruct = nullptr;
        }
    }
    ~CPPWrapper() {
        for (int i = 0; i < obj.nLayers; i++) {
            if (obj.pLayers[i].pLayerStruct != nullptr) {
                _mm_free(obj.pLayers[i].pLayerStruct);
            }
        }
        _mm_free(obj.pLayers);
    }
    intel_nnet_type_t * operator ->() {
        return &obj;
    }
    intel_nnet_type_t * operator *() {
        return &obj;
    }
    operator  intel_nnet_type_t &() {
        return *this;
    }
};

}  // namespace GNAPluginNS