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
#include <utility>

/**
 * @brief c++17 concept simulation
 */

template<class T>
class IPolymorhAllocator {
 public:
    virtual T *allocate(std::size_t n)  = 0;
    virtual void deallocate(T *p, std::size_t n)  = 0;
};

template<class T>
class allocator_polymorph;

template<class T>
class PolymorphAllocator {
    std::shared_ptr<IPolymorhAllocator<T>> _impl;
 public:
    explicit PolymorphAllocator(const std::shared_ptr<IPolymorhAllocator<T>> &impl) : _impl(impl) {}

    T *allocate(std::size_t n) {
        return _impl->allocate(n);
    }

    void deallocate(T *p, std::size_t n) {
        _impl->deallocate(p, n);
    }
};

/**
 * transform any allocator into polymorph type
 * @tparam origin
 */

template<class origin>
class polymorph_adapter : public IPolymorhAllocator<typename origin::value_type> {
    origin _impl;
    using T = typename origin::value_type;

 public:
    template<class ...Args>
    explicit polymorph_adapter(Args &&... args)
        :_impl(std::forward<Args>(args)...) {
    }
    T *allocate(std::size_t n) override {
        return _impl.allocate(n);
    }
    void deallocate(T *p, std::size_t n) override {
        _impl.deallocate(p, n);
    }
};

template<class T, class ...Args>
inline PolymorphAllocator<typename T::value_type> make_polymorph(Args &&... args) {
    auto sp = std::make_shared<polymorph_adapter<T>>(std::forward<Args>(args)...);
    auto ipoly = std::static_pointer_cast<IPolymorhAllocator<typename T::value_type>>(sp);

    return PolymorphAllocator<typename T::value_type>(ipoly);
}