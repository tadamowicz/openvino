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
// dnn_memory.hpp : memory manipulation routines

#pragma once

#include <cstdint>
extern void MemoryAssign(void **ptr_dest,
                         void **ptr_memory,
                         uint32_t num_bytes_needed,
                         uint32_t *ptr_num_bytes_used,
                         uint32_t num_memory_bytes,
                         const char *name);
