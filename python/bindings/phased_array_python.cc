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
/* BINDTOOL_HEADER_FILE(phased_array.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(eea19005591bf0332dd9511b5a9908ea)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <torchdsp/phased_array.h>
// pydoc.h is automatically generated in the build directory
#include <phased_array_pydoc.h>

void bind_phased_array(py::module& m) {

    using phased_array = ::gr::torchdsp::phased_array;


    py::class_<
        phased_array,
        gr::sync_block,
        gr::block,
        gr::basic_block,
        std::shared_ptr<phased_array>>(m, "phased_array", D(phased_array))

        .def(
            py::init(&phased_array::make),
            py::arg("num_elements"),
            py::arg("sep_lambda"),
            py::arg("angle_degrees"),
            py::arg("scale_db"),
            D(phased_array, make))


        ;
}
