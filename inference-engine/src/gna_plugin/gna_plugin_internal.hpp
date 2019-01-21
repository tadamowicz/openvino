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
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include "gna_executable_network.hpp"

namespace GNAPluginNS {

class GNAPluginInternal  : public InferenceEngine::InferencePluginInternal {
 public:
    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
                                                InferenceEngine::ICNNNetwork &network,
                                                const std::map<std::string, std::string> &config) override {
        return std::make_shared<GNAExecutableNetwork>(network, config);
    }
    void SetConfig(const std::map<std::string, std::string> &config) override {}
    InferenceEngine::IExecutableNetwork::Ptr  ImportNetwork(
                                                const std::string &modelFileName,
                                                const std::map<std::string, std::string> &config) override {
        return make_executable_network(std::make_shared<GNAExecutableNetwork>(modelFileName, config));
    }

    /**
     * @depricated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      InferenceEngine::QueryNetworkResult& res) const override {
        auto plg = std::make_shared<GNAPlugin>();
        plg->QueryNetwork(network, {}, res);
    }
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config,
                      InferenceEngine::QueryNetworkResult& res) const override {
        auto plg = std::make_shared<GNAPlugin>(config);
        plg->QueryNetwork(network, config, res);
    }
};

}  // namespace GNAPluginNS
