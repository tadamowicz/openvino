//
// Copyright 2016-2018 Intel Corporation.
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

#pragma once

#include <memory>
#include <cpp_interfaces/impl/ie_memory_state_internal.hpp>
#include "gna_plugin.hpp"

namespace  GNAPluginNS {

class GNAMemoryState : public InferenceEngine::MemoryStateInternal {
    std::shared_ptr<GNAPlugin> plg;
 public:
    using Ptr = InferenceEngine::MemoryStateInternal::Ptr;

    explicit GNAMemoryState(std::shared_ptr<GNAPlugin> plg)
        : InferenceEngine::MemoryStateInternal("GNAResetState"), plg(plg) {}
    void Reset() override {
        plg->Reset();
    }
};

}  // namespace GNAPluginNS