#include "solver.hpp"
#include <algorithm>
#include <limits>
#include <queue>

static const uint8_t PRESENCE = 0;
static const uint8_t ABSENCE = 1;

struct RuleCandidate {
    int node_idx;
    uint8_t polarity;
    int n_removed_pos;
    int n_removed_neg;
    double utility;
    int net_gain;
    int scan_order;
};

struct SearchState {
    std::vector<uint8_t> remaining;
    std::vector<int64_t> rule_nodes;
    std::vector<uint8_t> rule_polarities;
    int n_remaining_pos;
    int n_remaining_neg;
    double risk;
    double cumulative_utility;
};

static bool is_better_candidate(const RuleCandidate& a, const RuleCandidate& b) {
    if (a.utility != b.utility) return a.utility > b.utility;
    if (a.net_gain != b.net_gain) return a.net_gain > b.net_gain;
    return a.scan_order < b.scan_order;
}

static bool is_worse_candidate(const RuleCandidate& a, const RuleCandidate& b) {
    return is_better_candidate(b, a);
}

std::vector<RuleCandidate> find_top_rules(
    const uint64_t* nodes_start,
    const uint64_t* nodes_stop,
    size_t n_nodes,
    const uint16_t* kmers_assembly_idx,
    const uint8_t* y,
    const uint8_t* remaining,
    int n_remaining_pos,
    int n_remaining_neg,
    double p,
    int branch_width
) {
    struct WorseFirst {
        bool operator()(const RuleCandidate& a, const RuleCandidate& b) const {
            return is_better_candidate(a, b);
        }
    };

    std::priority_queue<RuleCandidate, std::vector<RuleCandidate>, WorseFirst> heap;

    for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
        uint64_t start = nodes_start[node_idx];
        uint64_t stop = nodes_stop[node_idx];
        int n_present_pos = 0;
        int n_present_neg = 0;
        int prev_asm = -1;
        for (uint64_t i = start; i < stop; ++i) {
            int asm_i = static_cast<int>(kmers_assembly_idx[i]);
            if (asm_i != prev_asm) {
                prev_asm = asm_i;
                if (remaining[asm_i]) {
                    int val = y[asm_i];
                    n_present_pos += val;
                    n_present_neg += (1 - val);
                }
            }
        }

        RuleCandidate presence;
        presence.node_idx = static_cast<int>(node_idx);
        presence.polarity = PRESENCE;
        presence.n_removed_pos = n_remaining_pos - n_present_pos;
        presence.n_removed_neg = n_remaining_neg - n_present_neg;
        presence.utility = -p * static_cast<double>(presence.n_removed_pos) + static_cast<double>(presence.n_removed_neg);
        presence.net_gain = -presence.n_removed_pos + presence.n_removed_neg;
        presence.scan_order = 2 * static_cast<int>(node_idx);

        if (static_cast<int>(heap.size()) < branch_width) {
            heap.push(presence);
        } else if (is_better_candidate(presence, heap.top())) {
            heap.pop();
            heap.push(presence);
        }

        RuleCandidate absence;
        absence.node_idx = static_cast<int>(node_idx);
        absence.polarity = ABSENCE;
        absence.n_removed_pos = n_present_pos;
        absence.n_removed_neg = n_present_neg;
        absence.utility = -p * static_cast<double>(absence.n_removed_pos) + static_cast<double>(absence.n_removed_neg);
        absence.net_gain = -absence.n_removed_pos + absence.n_removed_neg;
        absence.scan_order = 2 * static_cast<int>(node_idx) + 1;

        if (static_cast<int>(heap.size()) < branch_width) {
            heap.push(absence);
        } else if (is_better_candidate(absence, heap.top())) {
            heap.pop();
            heap.push(absence);
        }
    }

    std::vector<RuleCandidate> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::stable_sort(out.begin(), out.end(), is_better_candidate);
    return out;
}

void apply_rule(
    int node_idx,
    uint8_t polarity,
    const uint64_t* nodes_start,
    const uint64_t* nodes_stop,
    const uint16_t* kmers_assembly_idx,
    uint8_t* remaining,
    int* seen_stamp,
    int stamp,
    int n_assemblies
) {
    uint64_t start = nodes_start[node_idx];
    uint64_t stop = nodes_stop[node_idx];

    if (polarity == PRESENCE) {
        int prev_asm = -1;
        for (uint64_t i = start; i < stop; ++i) {
            int asm_i = static_cast<int>(kmers_assembly_idx[i]);
            if (asm_i != prev_asm) {
                prev_asm = asm_i;
                if (remaining[asm_i]) {
                    seen_stamp[asm_i] = stamp;
                }
            }
        }
        for (int asm_i = 0; asm_i < n_assemblies; ++asm_i) {
            if (remaining[asm_i] && seen_stamp[asm_i] != stamp) {
                remaining[asm_i] = 0;
            }
        }
    } else {
        int prev_asm = -1;
        for (uint64_t i = start; i < stop; ++i) {
            int asm_i = static_cast<int>(kmers_assembly_idx[i]);
            if (asm_i != prev_asm) {
                prev_asm = asm_i;
                if (remaining[asm_i]) {
                    remaining[asm_i] = 0;
                }
            }
        }
    }
}

static bool better_state(const SearchState& a, const SearchState& b, int n_initial_pos) {
    if (a.risk != b.risk) return a.risk < b.risk;
    if (a.n_remaining_neg != b.n_remaining_neg) return a.n_remaining_neg < b.n_remaining_neg;
    int a_removed_pos = n_initial_pos - a.n_remaining_pos;
    int b_removed_pos = n_initial_pos - b.n_remaining_pos;
    if (a_removed_pos != b_removed_pos) return a_removed_pos < b_removed_pos;
    if (a.cumulative_utility != b.cumulative_utility) return a.cumulative_utility > b.cumulative_utility;
    return false;
}

FitResult fit_impl(
    const uint64_t* nodes_start,
    const uint64_t* nodes_stop,
    size_t n_nodes,
    const uint16_t* kmers_assembly_idx,
    const uint8_t* is_target,
    size_t n_assemblies,
    int max_rules,
    double p,
    bool disjunction,
    int beam_width,
    int branch_width
) {
    std::vector<uint8_t> y(n_assemblies);
    int n_initial_pos = 0;
    if (disjunction) {
        for (size_t i = 0; i < n_assemblies; ++i) {
            y[i] = static_cast<uint8_t>(1 - is_target[i]);
            n_initial_pos += y[i];
        }
    } else {
        for (size_t i = 0; i < n_assemblies; ++i) {
            y[i] = static_cast<uint8_t>(is_target[i]);
            n_initial_pos += y[i];
        }
    }
    int n_initial_neg = static_cast<int>(n_assemblies) - n_initial_pos;

    SearchState initial_state;
    initial_state.remaining = std::vector<uint8_t>(n_assemblies, 1);
    initial_state.n_remaining_pos = n_initial_pos;
    initial_state.n_remaining_neg = n_initial_neg;
    initial_state.risk = static_cast<double>(n_initial_neg);
    initial_state.cumulative_utility = 0.0;

    std::vector<SearchState> beam;
    beam.push_back(initial_state);
    SearchState best_state_seen = initial_state;

    std::vector<int> seen_stamp(n_assemblies, 0);
    int stamp = 1;

    for (int depth = 0; depth < max_rules; ++depth) {
        std::vector<SearchState> children;
        for (const auto& state : beam) {
            if (state.n_remaining_neg == 0) {
                continue;
            }

            std::vector<RuleCandidate> top_rules = find_top_rules(
                nodes_start, nodes_stop, n_nodes,
                kmers_assembly_idx, y.data(), state.remaining.data(),
                state.n_remaining_pos, state.n_remaining_neg, p, branch_width
            );

            for (const auto& cand : top_rules) {
                SearchState child = state;
                child.rule_nodes.push_back(cand.node_idx);
                child.rule_polarities.push_back(cand.polarity);

                apply_rule(
                    cand.node_idx, cand.polarity,
                    nodes_start, nodes_stop,
                    kmers_assembly_idx,
                    child.remaining.data(),
                    seen_stamp.data(), stamp,
                    static_cast<int>(n_assemblies)
                );
                ++stamp;

                child.n_remaining_pos = state.n_remaining_pos - cand.n_removed_pos;
                child.n_remaining_neg = state.n_remaining_neg - cand.n_removed_neg;

                int removed_positive_total = n_initial_pos - child.n_remaining_pos;
                child.risk = p * static_cast<double>(removed_positive_total) + static_cast<double>(child.n_remaining_neg);
                child.cumulative_utility = state.cumulative_utility + cand.utility;

                if (better_state(child, best_state_seen, n_initial_pos)) {
                    best_state_seen = child;
                }
                children.push_back(std::move(child));
            }
        }

        if (children.empty()) {
            break;
        }

        std::stable_sort(children.begin(), children.end(),
            [n_initial_pos](const SearchState& a, const SearchState& b) {
                return better_state(a, b, n_initial_pos);
            }
        );
        if (static_cast<int>(children.size()) > beam_width) {
            children.resize(beam_width);
        }
        beam = std::move(children);
    }

    std::vector<int64_t> rule_nodes = std::move(best_state_seen.rule_nodes);
    std::vector<uint8_t> rule_polarities = std::move(best_state_seen.rule_polarities);
    std::vector<uint8_t> remaining = std::move(best_state_seen.remaining);

    if (disjunction) {
        for (auto& pol : rule_polarities) {
            pol = 1 - pol;
        }
        for (auto& r : remaining) {
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
