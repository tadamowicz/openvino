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

#include <functional>
#include "gna_device.hpp"
#include "polymorh_allocator.hpp"

/**
 * wrap GNA interface into c++ allocator friendly one
 */
class GNAAllocator {
    std::reference_wrapper<GNADeviceHelper> _device;

 public:
    typedef uint8_t value_type;

    explicit GNAAllocator(GNADeviceHelper &device) : _device(device) {
    }
    uint8_t *allocate(std::size_t n) {
        uint32_t granted = 0;
        auto result = _device.get().alloc(n, &granted);
        if (result == nullptr || granted == 0) {
            throw std::bad_alloc();
        }
        return result;
    }
    void deallocate(uint8_t *p, std::size_t n) {
        _device.get().free();
    }
};
