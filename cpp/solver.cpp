#include <algorithm>
#include <limits>
#include <queue>

#include "solver.hpp"

static const uint8_t PRESENCE = 0;
static const uint8_t ABSENCE = 1;
static const int CANDIDATE_POOL_MULTIPLIER = 20;
static const double CANDIDATE_MMR_LAMBDA = 0.30;
static const double STATE_MMR_LAMBDA = 0.15;
static const int STATE_ELITE_FRACTION_DENOMINATOR = 4;

struct Rule {
    int node_idx;
    uint8_t polarity;
    int n_removed_pos;
    int n_removed_neg;
    double utility;
    int net_gain;
    int scan_order;
    std::vector<uint8_t> removed_mask;
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

static bool is_better_rule(const Rule& a, const Rule& b) {
    if (a.utility != b.utility) return a.utility > b.utility;
    if (a.net_gain != b.net_gain) return a.net_gain > b.net_gain;
    return a.scan_order < b.scan_order;
}

std::vector<Rule> find_top_rules(
    const uint64_t* nodes_start,
    const uint64_t* nodes_stop,
    size_t n_nodes,
    const uint16_t* kmers_assembly_idx,
    const uint8_t* y,
    const uint8_t* remaining,
    size_t n_assemblies,
    int n_remaining_pos,
    int n_remaining_neg,
    double p,
    int branch_width
) {
    int candidate_pool_size = CANDIDATE_POOL_MULTIPLIER * branch_width;
    if (candidate_pool_size < 1) candidate_pool_size = 1;

    // Stage 1: keep only the top candidate_pool_size candidates while scanning all rules
    struct WorseFirst {
        bool operator()(const Rule& a, const Rule& b) const {
            return is_better_rule(a, b);
        }
    };

    std::priority_queue<Rule, std::vector<Rule>, WorseFirst> heap;

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

        Rule presence;
        presence.node_idx = static_cast<int>(node_idx);
        presence.polarity = PRESENCE;
        presence.n_removed_pos = n_remaining_pos - n_present_pos;
        presence.n_removed_neg = n_remaining_neg - n_present_neg;
        presence.utility = -p * static_cast<double>(presence.n_removed_pos) + static_cast<double>(presence.n_removed_neg);
        presence.net_gain = -presence.n_removed_pos + presence.n_removed_neg;
        presence.scan_order = 2 * static_cast<int>(node_idx);

        if (static_cast<int>(heap.size()) < candidate_pool_size) {
            heap.push(presence);
        } else if (is_better_rule(presence, heap.top())) {
            heap.pop();
            heap.push(presence);
        }

        Rule absence;
        absence.node_idx = static_cast<int>(node_idx);
        absence.polarity = ABSENCE;
        absence.n_removed_pos = n_present_pos;
        absence.n_removed_neg = n_present_neg;
        absence.utility = -p * static_cast<double>(absence.n_removed_pos) + static_cast<double>(absence.n_removed_neg);
        absence.net_gain = -absence.n_removed_pos + absence.n_removed_neg;
        absence.scan_order = 2 * static_cast<int>(node_idx) + 1;

        if (static_cast<int>(heap.size()) < candidate_pool_size) {
            heap.push(absence);
        } else if (is_better_rule(absence, heap.top())) {
            heap.pop();
            heap.push(absence);
        }
    }

    std::vector<Rule> out;
    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::stable_sort(out.begin(), out.end(), is_better_rule);

    // Stage 2a: compute removed masks for retained pool using apply_rule semantics.
    for (auto& rule : out) {
        rule.removed_mask.assign(remaining, remaining + n_assemblies);
        uint64_t start = nodes_start[rule.node_idx];
        uint64_t stop = nodes_stop[rule.node_idx];
        if (rule.polarity == PRESENCE) {
            std::vector<uint8_t> seen(rule.removed_mask.size(), 0);
            int prev_asm = -1;
            for (uint64_t i = start; i < stop; ++i) {
                int asm_i = static_cast<int>(kmers_assembly_idx[i]);
                if (asm_i != prev_asm) {
                    prev_asm = asm_i;
                    if (rule.removed_mask[asm_i]) {
                        seen[asm_i] = 1;
                    }
                }
            }
            for (size_t asm_i = 0; asm_i < rule.removed_mask.size(); ++asm_i) {
                if (rule.removed_mask[asm_i] && seen[asm_i]) {
                    rule.removed_mask[asm_i] = 0;
                }
            }
        } else {
            int prev_asm = -1;
            for (uint64_t i = start; i < stop; ++i) {
                int asm_i = static_cast<int>(kmers_assembly_idx[i]);
                if (asm_i != prev_asm) {
                    prev_asm = asm_i;
                    if (rule.removed_mask[asm_i]) {
                        rule.removed_mask[asm_i] = 0;
                    }
                }
            }
        }
    }

    // Stage 2b: MMR reranking from retained pool.
    double min_utility = std::numeric_limits<double>::infinity();
    double max_utility = -std::numeric_limits<double>::infinity();
    for (const auto& rule : out) {
        min_utility = std::min(min_utility, rule.utility);
        max_utility = std::max(max_utility, rule.utility);
    }
    auto normalized_utility = [min_utility, max_utility](double utility) {
        if (max_utility == min_utility) return 1.0;
        return (utility - min_utility) / (max_utility - min_utility);
    };
    auto jaccard_removed = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        int inter = 0;
        int uni = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            uint8_t ai = a[i];
            uint8_t bi = b[i];
            inter += static_cast<int>(ai && bi);
            uni += static_cast<int>(ai || bi);
        }
        if (uni == 0) return 0.0;
        return static_cast<double>(inter) / static_cast<double>(uni);
    };

    std::vector<Rule> selected;
    selected.reserve(static_cast<size_t>(std::min(branch_width, static_cast<int>(out.size()))));
    std::vector<uint8_t> used(out.size(), 0);
    if (!out.empty() && branch_width > 0) {
        selected.push_back(out[0]);
        used[0] = 1;
    }
    while (static_cast<int>(selected.size()) < branch_width && static_cast<int>(selected.size()) < static_cast<int>(out.size())) {
        int best_idx = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < out.size(); ++i) {
            if (used[i]) continue;
            double max_jaccard = 0.0;
            for (const auto& picked : selected) {
                max_jaccard = std::max(max_jaccard, jaccard_removed(out[i].removed_mask, picked.removed_mask));
            }
            double score = normalized_utility(out[i].utility) - CANDIDATE_MMR_LAMBDA * max_jaccard;
            if (best_idx == -1 || score > best_score ||
                (score == best_score && is_better_rule(out[i], out[best_idx]))) {
                best_idx = static_cast<int>(i);
                best_score = score;
            }
        }
        used[best_idx] = 1;
        selected.push_back(out[best_idx]);
    }
    return selected;
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
        // Mark assemblies seen in this node slice
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

    // Expand beam level by level up to max_rules
    for (int depth = 0; depth < max_rules; ++depth) {
        std::vector<SearchState> children;
        for (const auto& state : beam) {
            if (state.n_remaining_neg == 0) {
                // Terminal state for this transformed objective; skip expansion
                continue;
            }

            std::vector<Rule> top_rules = find_top_rules(
                nodes_start, nodes_stop, n_nodes,
                kmers_assembly_idx, y.data(), state.remaining.data(), n_assemblies,
                state.n_remaining_pos, state.n_remaining_neg, p, branch_width
            );

            for (const auto& cand : top_rules) {
                // Create one child per retained candidate rule
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

                // Track the best risk state seen at any depth
                if (better_state(child, best_state_seen, n_initial_pos)) {
                    best_state_seen = child;
                }
                children.push_back(std::move(child));
            }
        }

        if (children.empty()) {
            // All beam states were terminal (or produced no expansions)
            break;
        }

        // Keep elite, then fill remainder by MMR over remaining mask redundancy.
        std::stable_sort(children.begin(), children.end(),
            [n_initial_pos](const SearchState& a, const SearchState& b) {
                return better_state(a, b, n_initial_pos);
            }
        );
        if (static_cast<int>(children.size()) > beam_width) {
            int elite_width = std::max(1, beam_width / STATE_ELITE_FRACTION_DENOMINATOR);
            elite_width = std::min(elite_width, beam_width);
            std::vector<SearchState> next_beam;
            next_beam.reserve(beam_width);
            for (int i = 0; i < elite_width && i < static_cast<int>(children.size()); ++i) {
                next_beam.push_back(children[i]);
            }

            double min_risk = std::numeric_limits<double>::infinity();
            double max_risk = -std::numeric_limits<double>::infinity();
            for (const auto& c : children) {
                min_risk = std::min(min_risk, c.risk);
                max_risk = std::max(max_risk, c.risk);
            }
            auto normalized_risk_relevance = [min_risk, max_risk](double risk) {
                if (max_risk == min_risk) return 1.0;
                return (max_risk - risk) / (max_risk - min_risk);
            };
            auto jaccard_remaining = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
                int inter = 0;
                int uni = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    uint8_t ai = a[i];
                    uint8_t bi = b[i];
                    inter += static_cast<int>(ai && bi);
                    uni += static_cast<int>(ai || bi);
                }
                if (uni == 0) return 0.0;
                return static_cast<double>(inter) / static_cast<double>(uni);
            };

            std::vector<uint8_t> used(children.size(), 0);
            for (int i = 0; i < elite_width && i < static_cast<int>(children.size()); ++i) {
                used[i] = 1;
            }
            while (static_cast<int>(next_beam.size()) < beam_width) {
                int best_idx = -1;
                double best_score = -std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < children.size(); ++i) {
                    if (used[i]) continue;
                    double max_jaccard = 0.0;
                    for (const auto& picked : next_beam) {
                        max_jaccard = std::max(max_jaccard, jaccard_remaining(children[i].remaining, picked.remaining));
                    }
                    double score = normalized_risk_relevance(children[i].risk) - STATE_MMR_LAMBDA * max_jaccard;
                    if (best_idx == -1 || score > best_score ||
                        (score == best_score && better_state(children[i], children[best_idx], n_initial_pos))) {
                        best_idx = static_cast<int>(i);
                        best_score = score;
                    }
                }
                if (best_idx == -1) break;
                used[best_idx] = 1;
                next_beam.push_back(children[best_idx]);
            }
            children = std::move(next_beam);
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
