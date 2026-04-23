#include "scm_core.hpp"
#include <limits>

static const bool PRESENCE = true;
static const bool ABSENCE = false;

void find_best_rule(
    const uint64_t* node_start,
    const uint64_t* node_stop,
    size_t n_nodes,
    const uint16_t* kmer_assembly_idx,
    const uint8_t* y,
    const uint8_t* remaining,
    int n_remaining_pos,
    int n_remaining_neg,
    double p,
    int& out_node_idx,
    bool& out_polarity,
    int& out_n_removed_pos,
    int& out_n_removed_neg
) {
    double best_utility = -std::numeric_limits<double>::infinity();
    int best_net_gain = std::numeric_limits<int>::min();
    out_node_idx = -1;
    out_polarity = PRESENCE;
    out_n_removed_pos = 0;
    out_n_removed_neg = 0;

    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        uint64_t start = node_start[node_idx];
        uint64_t stop  = node_stop[node_idx];
        int n_present_pos = 0;
        int n_present_neg = 0;
        int prev_asm = -1;
        // Count unique assemblies in this node slice
        for (uint64_t i = start; i < stop; ++i) {
            int asm_i = static_cast<int>(kmer_assembly_idx[i]);
            if (asm_i != prev_asm) {
                prev_asm = asm_i;
                if (remaining[asm_i]) {
                    int val = y[asm_i];
                    n_present_pos += val;
                    n_present_neg += (1 - val);
                }
            }
        }
        // Presence rule
        int n_removed_pos = n_remaining_pos - n_present_pos;
        int n_removed_neg = n_remaining_neg - n_present_neg;
        double utility = -p * static_cast<double>(n_removed_pos) + static_cast<double>(n_removed_neg);
        int net_gain = -n_removed_pos + n_removed_neg;
        if (utility > best_utility || (utility == best_utility && net_gain > best_net_gain)) {
            out_node_idx = static_cast<int>(node_idx);
            out_polarity = PRESENCE;
            out_n_removed_pos = n_removed_pos;
            out_n_removed_neg = n_removed_neg;
            best_utility = utility;
            best_net_gain = net_gain;
        }
        // Absence rule
        n_removed_pos = n_present_pos;
        n_removed_neg = n_present_neg;
        utility = -p * static_cast<double>(n_removed_pos) + static_cast<double>(n_removed_neg);
        net_gain = -n_removed_pos + n_removed_neg;
        if (utility > best_utility || (utility == best_utility && net_gain > best_net_gain)) {
            out_node_idx = static_cast<int>(node_idx);
            out_polarity = ABSENCE;
            out_n_removed_pos = n_removed_pos;
            out_n_removed_neg = n_removed_neg;
            best_utility = utility;
            best_net_gain = net_gain;
        }
    }
}

void apply_best_rule(
    int node_idx,
    bool polarity,
    const uint64_t* node_start,
    const uint64_t* node_stop,
    const uint16_t* kmer_assembly_idx,
    uint8_t* remaining,
    int* seen_stamp,
    int stamp,
    int n_assemblies
) {
    uint64_t start = node_start[node_idx];
    uint64_t stop  = node_stop[node_idx];

    if (polarity == PRESENCE) {
        // Mark assemblies seen in this node slice
        int prev_asm = -1;
        for (uint64_t i = start; i < stop; ++i) {
            int asm_i = static_cast<int>(kmer_assembly_idx[i]);
            if (asm_i != prev_asm) {
                prev_asm = asm_i;
                if (remaining[asm_i]) {
                    seen_stamp[asm_i] = stamp;
                }
            }
        }
        // Remove those not seen
        for (int asm_i = 0; asm_i < n_assemblies; ++asm_i) {
            if (remaining[asm_i] && seen_stamp[asm_i] != stamp) {
                remaining[asm_i] = 0;
            }
        }
    } else {
        // ABSENCE: remove those seen in this node slice
        int prev_asm = -1;
        for (uint64_t i = start; i < stop; ++i) {
            int asm_i = static_cast<int>(kmer_assembly_idx[i]);
            if (asm_i != prev_asm) {
                prev_asm = asm_i;
                if (remaining[asm_i]) {
                    remaining[asm_i] = 0;
                }
            }
        }
    }
}

FitResult fit_impl(
    const uint64_t* node_start,
    const uint64_t* node_stop,
    size_t n_nodes,
    const uint16_t* kmer_assembly_idx,
    const bool* is_target,
    size_t n_assemblies,
    int max_rules,
    double p,
    bool disjunction
) {
    // Prepare label array y (uint8 0/1)
    std::vector<uint8_t> y(n_assemblies);
    if (disjunction) {
        for (size_t i = 0; i < n_assemblies; ++i) {
            y[i] = static_cast<uint8_t>(1 - static_cast<int>(is_target[i]));
        }
    } else {
        for (size_t i = 0; i < n_assemblies; ++i) {
            y[i] = static_cast<uint8_t>(is_target[i] ? 1 : 0);
        }
    }
    // Initialize remaining mask and seen_stamp
    std::vector<uint8_t> remaining(n_assemblies, 1);
    std::vector<int> seen_stamp(n_assemblies, 0);
    int n_remaining_pos = 0;
    for (auto v : y) n_remaining_pos += v;
    int n_remaining_neg = static_cast<int>(n_assemblies) - n_remaining_pos;

    // Store rules
    std::vector<int64_t> rule_nodes;
    std::vector<uint8_t> rule_polarities;
    rule_nodes.reserve(max_rules);
    rule_polarities.reserve(max_rules);

    int stamp = 1;
    while (n_remaining_neg > 0 && (int)rule_nodes.size() < max_rules) {
        int node_idx;
        bool polarity;
        int n_removed_pos, n_removed_neg;
        find_best_rule(node_start, node_stop, n_nodes,
                       kmer_assembly_idx, y.data(), remaining.data(),
                       n_remaining_pos, n_remaining_neg, p,
                       node_idx, polarity, n_removed_pos, n_removed_neg);
        apply_best_rule(node_idx, polarity, node_start, node_stop,
                        kmer_assembly_idx, remaining.data(),
                        seen_stamp.data(), stamp, static_cast<int>(n_assemblies));

        rule_nodes.push_back(node_idx);
        rule_polarities.push_back(polarity ? 1 : 0);
        n_remaining_pos -= n_removed_pos;
        n_remaining_neg -= n_removed_neg;
        stamp += 1;
    }

    // Post-process disjunction: invert polarity and remaining
    if (disjunction) {
        for (auto &pol : rule_polarities) {
            pol = 1 - pol;
        }
        for (auto &r : remaining) {
            r = 1 - r;
        }
    }

    FitResult res;
    res.disjunction = disjunction;
    res.nodes = std::move(rule_nodes);
    res.polarities = std::move(rule_polarities);
    res.pred = std::move(remaining);
    return res;
}
