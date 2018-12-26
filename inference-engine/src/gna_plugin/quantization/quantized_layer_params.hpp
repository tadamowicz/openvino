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

namespace GNAPluginNS {

struct Quantization {
    float scale = 1.0f;
    float offset = 0.0f;
    int shift = 0.0f;
};

struct QuantizedLayerParams {
    Quantization _src_quant;
    Quantization _dst_quant;
    Quantization _weights_quant;
    Quantization _bias_quant;
    float _o_shift = 0.0f;
    float _b_shift = 0.0f;
};

}  // namespace GNAPluginNS