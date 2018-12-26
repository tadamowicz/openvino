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

#include "ie_precision.hpp"

namespace InferenceEngine {

/**
 * @brief reverse trait for getting some precision from it's underlined memory type
 * this might not work for certain precisions : for Q78, U16
 * @tparam T
 */
template<class T>
struct precision_from_media {
    static const Precision::ePrecision type = Precision::CUSTOM;
};

template<>
struct precision_from_media<float> {
    static const Precision::ePrecision type = Precision::FP32;
};

template<>
struct precision_from_media<uint16_t> {
    static const Precision::ePrecision type = Precision::FP16;
};

template<>
struct precision_from_media<int16_t> {
    static const Precision::ePrecision type = Precision::I16;
};

template<>
struct precision_from_media<uint8_t> {
    static const Precision::ePrecision type = Precision::U8;
};

template<>
struct precision_from_media<int8_t> {
    static const Precision::ePrecision type = Precision::I8;
};

template<>
struct precision_from_media<int32_t> {
    static const Precision::ePrecision type = Precision::I32;
};

/**
 * @brief container for storing both precision and it's underlined media type
 * @tparam TMedia
 */
template <class TMedia>
class TPrecision : public Precision {
 public:
    typedef TMedia MediaType;
    TPrecision() : Precision(precision_from_media<TMedia>::type) {}
    explicit TPrecision(const Precision & that) : Precision(that) {}
    TPrecision & operator = (const Precision & that) {
        Precision::operator=(that);
        return *this;
    }
    explicit TPrecision(const Precision::ePrecision  value) : Precision(value) {}
};

template <class T> TPrecision<T> createTPrecision() {
    TPrecision<T> cnt(InferenceEngine::Precision::fromType<T>());
    return cnt;
}

template <InferenceEngine::Precision::ePrecision T>
TPrecision<typename InferenceEngine::PrecisionTrait<T>::value_type> createTPrecision() {
    TPrecision<typename InferenceEngine::PrecisionTrait<T>::value_type> cnt(T);
    return cnt;
}


// special case for Mixed, or undefined precisions
template <>
class TPrecision<void> : public Precision {
 public:
    typedef void MediaType;
    TPrecision() = default;
    explicit TPrecision(const Precision & that) : Precision(that) {}
    TPrecision & operator = (const Precision & that) {
        Precision::operator=(that);
        return *this;
    }
    explicit TPrecision(const Precision::ePrecision  value) : Precision(value) {}
};


}  // namespace InferenceEngine