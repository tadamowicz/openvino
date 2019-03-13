//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include "gna-api-dumper.h"
#include "gna-api-instrumentation.h"
#include "ie_common.h"
#include <memory>
#include <string>
#include <map>
#include <thread>

/**
 * holds gna - style handle in RAII way
 */
class GNADeviceHelper {
    intel_gna_status_t nGNAStatus = GNA_NOERROR;
    intel_gna_handle_t nGNAHandle = 0;
    intel_gna_proc_t nGNAProcType = GNA_AUTO;
    intel_gna_perf_t nGNAPerfResults;
    intel_gna_perf_t nGNAPerfResultsTotal;
    const uint32_t GNA_TIMEOUT = MAX_TIMEOUT;
    bool isPerformanceMeasuring;

 public:
    explicit GNADeviceHelper(intel_gna_proc_t proc_type = GNA_AUTO,
                            uint8_t lib_async_n_threads = 1,
                            bool use_openmp = false,
                            bool isPerformanceMeasuring = false) :
                                    nGNAProcType(proc_type),
                                    isPerformanceMeasuring(isPerformanceMeasuring) {
        initGnaPerfCounters();
        open(lib_async_n_threads);

        if (use_openmp) {
            uint8_t num_cores = std::thread::hardware_concurrency();
            setOMPThreads((num_cores != 0) ? num_cores : 1);
        }
    }

    ~GNADeviceHelper() {
        close();
    }

    uint8_t *alloc(uint32_t size_requested, uint32_t *size_granted);

    void propagateSync(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices);

    uint32_t propagate(const intel_nnet_type_t *pNeuralNetwork,
                       const uint32_t *pActiveIndices,
                       uint32_t nActiveIndices);

    void wait(uint32_t id);


    struct DumpResult {
        intel_gna_model_header header;
        std::shared_ptr<void> model;
    };

    DumpResult dumpXnn(const intel_nnet_type_t *pNeuralNetwork,
                 const uint32_t *pActiveIndices,
                 uint32_t nActiveIndices);


    void free() {
        GNAFree(nGNAHandle);
    }
    void updateGnaPerfCounters();
    void getGnaPerfCounters(std::map<std::string,
                        InferenceEngine::InferenceEngineProfileInfo>& retPerfCounters);

 private:
    void open(uint8_t const n_threads);

    void close();

    void checkStatus() const;

    void setOMPThreads(uint8_t const n_threads);

    void initGnaPerfCounters() {
        nGNAPerfResults = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
        nGNAPerfResultsTotal = {{0, 0, 0, 0, 0, 0, 0}, {0, 0}, {0, 0, 0}, {0, 0}};
    }
};

