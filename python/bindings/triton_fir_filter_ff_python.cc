/*
 * Copyright 2022 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(triton_fir_filter_ff.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(164ce1116e2c77387f76afc73d00eedc)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <torchdsp/triton_fir_filter_ff.h>
// pydoc.h is automatically generated in the build directory
#include <triton_fir_filter_ff_pydoc.h>

void bind_triton_fir_filter_ff(py::module& m) {

    using triton_fir_filter_ff = gr::torchdsp::triton_fir_filter_ff;


    py::class_<triton_fir_filter_ff, gr::sync_decimator, gr::block, gr::basic_block, std::shared_ptr<triton_fir_filter_ff>>(
        m, "triton_fir_filter_ff", D(triton_fir_filter_ff))

        .def(
            py::init(&triton_fir_filter_ff::make),
            py::arg("model_name"),
            py::arg("max_batch_size"),
            py::arg("triton_url"),
            py::arg("tap_size"),
            D(triton_fir_filter_ff, make))


        ;
}
