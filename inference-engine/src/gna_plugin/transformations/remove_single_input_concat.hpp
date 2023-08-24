// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief remove concat layers with single input
 */
class RemoveSingleInputConcat : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  RemoveSingleInputConcat();
};

} // namespace GNAPluginNS
