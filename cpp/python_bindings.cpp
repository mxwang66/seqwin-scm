#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "scm_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Core SCM implementation (C++ via pybind11)";

    m.def("_fit_impl",
        [](py::array_t<uint64_t> node_start,
           py::array_t<uint64_t> node_stop,
           py::array_t<uint16_t> kmer_assembly_idx,
           py::array_t<bool> is_target,
           int max_rules, double p, bool disjunction
        ) {
            // Extract raw pointers and sizes
            auto ns_buf = node_start.request();
            auto no_buf = node_stop.request();
            auto km_buf = kmer_assembly_idx.request();
            auto it_buf = is_target.request();
            const uint64_t* ns_ptr = static_cast<uint64_t*>(ns_buf.ptr);
            const uint64_t* no_ptr = static_cast<uint64_t*>(no_buf.ptr);
            const uint16_t* km_ptr = static_cast<uint16_t*>(km_buf.ptr);
            const bool* it_ptr = static_cast<bool*>(it_buf.ptr);
            size_t n_nodes = ns_buf.shape[0];
            size_t n_assemblies = it_buf.shape[0];

            // Release the Python GIL while running the CPU-bound C++ core.
            FitResult res;
            {
                py::gil_scoped_release release;
                res = fit_impl(
                    ns_ptr, no_ptr, n_nodes,
                    km_ptr,
                    it_ptr, n_assemblies,
                    max_rules, p, disjunction
                );
            }

            // Convert results to NumPy arrays
            py::array_t<long long> py_nodes(res.nodes.size());
            long long* py_nodes_ptr = static_cast<long long*>(py_nodes.request().ptr);
            for (size_t i = 0; i < res.nodes.size(); ++i) {
                py_nodes_ptr[i] = res.nodes[i];
            }
            py::array_t<bool> py_pol(res.polarities.size());
            bool* py_pol_ptr = static_cast<bool*>(py_pol.request().ptr);
            for (size_t i = 0; i < res.polarities.size(); ++i) {
                py_pol_ptr[i] = (res.polarities[i] != 0);
            }
            py::array_t<bool> py_pred(res.pred.size());
            bool* py_pred_ptr = static_cast<bool*>(py_pred.request().ptr);
            for (size_t i = 0; i < res.pred.size(); ++i) {
                py_pred_ptr[i] = (res.pred[i] != 0);
            }

            return py::make_tuple(res.disjunction, py_nodes, py_pol, py_pred);
        },
        "Fit the SCM and return (disjunction, nodes, polarities, pred)."
    );
}
