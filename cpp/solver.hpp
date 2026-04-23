#pragma once
#include <vector>
#include <cstdint>

/// Result struct for fit_impl
struct FitResult {
    bool disjunction;
    std::vector<int64_t> nodes;
    std::vector<uint8_t> polarities; // 0 or 1
    std::vector<uint8_t> pred;       // 0 or 1 (final remaining mask)
};

/// Perform fitting. The arrays nodes_start, nodes_stop have length n_nodes,
/// and is_target has length n_assemblies. Dtypes must match exactly.
FitResult fit_impl(
    const uint64_t* nodes_start,
    const uint64_t* nodes_stop,
    size_t n_nodes,
    const uint16_t* kmers_assembly_idx,
    const uint8_t* is_target,
    size_t n_assemblies,
    int max_rules,
    double p,
    bool disjunction
);
