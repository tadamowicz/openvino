// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <utility>

namespace GNAPluginNS {
namespace backend {
class DNN_Dump {
private:
    int N;
    static bool instanceFlag;
    static std::unique_ptr<DNN_Dump> instancePtr;
    DNN_Dump();

public:
    static std::unique_ptr<GNAPluginNS::backend::DNN_Dump> getInstance() {
        if (!instanceFlag) {
            instancePtr = std::unique_ptr<GNAPluginNS::backend::DNN_Dump>(new GNAPluginNS::backend::DNN_Dump());
            instanceFlag = true;
            return std::move(instancePtr);
        } else {
            return std::move(instancePtr);
        }
    }

    int & getDumpFolderId();
    std::string getDumpFilePrefixGNA();
    std::string getDumpFolderName();
    std::string getRefFolderName();
};
}  // namespace backend
}  // namespace GNAPluginNS
