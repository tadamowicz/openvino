// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>

#include "dnn_dump.hpp"

bool GNAPluginNS::backend::DNN_Dump::instanceFlag;
std::unique_ptr<GNAPluginNS::backend::DNN_Dump> GNAPluginNS::backend::DNN_Dump::instancePtr;

GNAPluginNS::backend::DNN_Dump::DNN_Dump() {
    N = 0;
}

int & GNAPluginNS::backend::DNN_Dump::getDumpFolderId() {
    return N;
}

std::string GNAPluginNS::backend::DNN_Dump::getDumpFilePrefixGNA() {
    return std::string("./gna_layers/") + std::to_string(getDumpFolderId() - 1) + "/";
}

std::string GNAPluginNS::backend::DNN_Dump::getDumpFolderName() {
    return std::string("./layers/") + std::to_string(getDumpFolderId() - 1) + "/";
}

std::string GNAPluginNS::backend::DNN_Dump::getRefFolderName() {
    return std::string("./ref_layers/") + std::to_string(getDumpFolderId() - 1) + "/";
}
