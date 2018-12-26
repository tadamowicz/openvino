//
// Copyright 2012-2018 Intel Corporation.
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
// lstm.cpp : GNA LSTM macro layer definition
//

#include "lstm.hpp"

const char *intel_lstm_projected_layer_name[NUM_LSTM_LAYERS] = {
    "combined input transform",
    "combined recurrent transform",
    "input gate",
    "forget gate",
    "cell gate input part 1",
    "cell gate input part 2",
    "cell gate output part 1",
    "cell gate output part 2",
    "output gate",
    "hidden gated output",
    "projected output"
};

const char *intel_lstm_projected_layer_g4_name[NUM_LSTM_G4_LAYERS] = {
    "combined input transform",
    "deinterleave",
    "interleave 1",
    "interleave 2",
    "interleave 3",
    "interleave 4",
    "combined recurrent transform - 1",
    "input gate - 1",
    "forget gate - 1",
    "cell gate input part 1 - 1",
    "cell gate input part 2 - 1",
    "cell gate output part 1 - 1",
    "cell gate output part 2 - 1",
    "output gate - 1",
    "hidden gated output - 1",
    "projected output - 1",
    "combined recurrent transform - 2",
    "input gate - 2",
    "forget gate - 2",
    "cell gate input part 1 - 2",
    "cell gate input part 2 - 2",
    "cell gate output part 1 - 2",
    "cell gate output part 2 - 2",
    "output gate - 2",
    "hidden gated output - 2",
    "projected output - 2",
    "combined recurrent transform - 3",
    "input gate - 3",
    "forget gate - 3",
    "cell gate input part 1 - 3",
    "cell gate input part 2 - 3",
    "cell gate output part 1 - 3",
    "cell gate output part 2 - 3",
    "output gate - 3",
    "hidden gated output - 3",
    "projected output - 3",
    "combined recurrent transform - 4",
    "input gate - 4",
    "forget gate - 4",
    "cell gate input part 1 - 4",
    "cell gate input part 2 - 4",
    "cell gate output part 1 - 4",
    "cell gate output part 2 - 4",
    "output gate - 4",
    "hidden gated output - 4",
    "projected output - 4",
    "interleave"
};