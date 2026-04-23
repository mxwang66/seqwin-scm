#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Core SCM implementation (C++ via pybind11)";

    m.def("_fit_native",
        [](py::array_t<uint64_t> nodes_start,
           py::array_t<uint64_t> nodes_stop,
           py::array_t<uint16_t> kmers_assembly_idx,
           py::array_t<uint8_t> is_target,
           int max_rules, double p, bool disjunction
        ) {
            // Extract raw pointers and sizes
            auto ns_buf = nodes_start.request();
            auto no_buf = nodes_stop.request();
            auto km_buf = kmers_assembly_idx.request();
            auto it_buf = is_target.request();
            const uint64_t* ns_ptr = static_cast<uint64_t*>(ns_buf.ptr);
            const uint64_t* no_ptr = static_cast<uint64_t*>(no_buf.ptr);
            const uint16_t* km_ptr = static_cast<uint16_t*>(km_buf.ptr);
            const uint8_t* it_ptr = static_cast<uint8_t*>(it_buf.ptr);
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

            auto res_owner = std::make_shared<FitResult>(std::move(res));
            auto capsule = py::capsule(
                new std::shared_ptr<FitResult>(res_owner),
                [](void* p) { delete static_cast<std::shared_ptr<FitResult>*>(p); }
            );

            auto py_nodes = py::array_t<int64_t>(
                {static_cast<py::ssize_t>(res_owner->nodes.size())},
                {static_cast<py::ssize_t>(sizeof(int64_t))},
                res_owner->nodes.data(),
                capsule
            );
            auto py_pol = py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(res_owner->polarities.size())},
                {static_cast<py::ssize_t>(sizeof(uint8_t))},
                res_owner->polarities.data(),
                capsule
            );
            auto py_pred = py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(res_owner->pred.size())},
                {static_cast<py::ssize_t>(sizeof(uint8_t))},
                res_owner->pred.data(),
                capsule
            );

            return py::make_tuple(res_owner->disjunction, py_nodes, py_pol, py_pred);
        },
        "Fit the SCM and return (disjunction, nodes, polarities, pred)."
    );
}
