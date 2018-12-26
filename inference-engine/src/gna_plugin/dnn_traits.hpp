//
// Copyright 2017-2018 Intel Corporation.
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
// dnn_traits.hpp : c++ trait approach to  define dnn objects
//

#pragma once

#include "dnn.h"

template<intel_dnn_operation_t layer>
struct DnnTrait {};

template<>
struct DnnTrait<kDnnDiagonalOp> {
    using Type = intel_affine_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.affine;
    }
};

template<>
struct DnnTrait<kDnnPiecewiselinearOp> {
    using Type = intel_piecewiselinear_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.pwl;
    }
};

template<>
struct DnnTrait<kDnnAffineOp> {
    using Type = intel_affine_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.affine;
    }
};

template<>
struct DnnTrait<kDnnConvolutional1dOp> {
    using Type = intel_convolutionalD_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.conv1D;
    }
};

template<>
struct DnnTrait<kDnnMaxPoolOp> {
    using Type = intel_maxpool_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.maxpool;
    }
};

template<>
struct DnnTrait<kDnnRecurrentOp> {
    using Type = intel_recurrent_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.recurrent;
    }
};

template<>
struct DnnTrait<kDnnInterleaveOp> {
    using Type = intel_interleave_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.interleave;
    }
};

template<>
struct DnnTrait<kDnnDeinterleaveOp> {
    using Type = intel_deinterleave_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.deinterleave;
    }
};

template<>
struct DnnTrait<kDnnCopyOp> {
    using Type = intel_copy_t;
    static Type *getLayer(intel_dnn_component_t &component) {
        return &component.op.copy;
    }
};

template<>
struct DnnTrait<kDnnNullOp> {
    using Type = void;
    static Type *getLayer(intel_dnn_component_t &component) {
        return nullptr;
    }
};
