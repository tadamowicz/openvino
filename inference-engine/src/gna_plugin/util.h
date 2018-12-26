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

#pragma once

void *AllocateMemory(uint32_t num_memory_bytes, const char *ptr_name);
void FreeMemory(void *ptr_memory);
int32_t MemoryOffset(void *ptr_target, void *ptr_base);
