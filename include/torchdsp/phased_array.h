/* -*- c++ -*- */
/*
 * Copyright 2022 Peraton Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TORCHDSP_PHASED_ARRAY_H
#define INCLUDED_TORCHDSP_PHASED_ARRAY_H

#include <gnuradio/sync_block.h>
#include <torchdsp/api.h>

namespace gr {
namespace torchdsp {

/*!
 * \brief <+description of block+>
 * \ingroup torchdsp
 *
 */
class TORCHDSP_API phased_array : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<phased_array> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of torchdsp::phased_array.
     *
     * To avoid accidental use of raw pointers, torchdsp::phased_array's
     * constructor is in a private implementation
     * class. torchdsp::phased_array::make is the public interface for
     * creating new instances.
     */
    static sptr make(int num_elements, float sep_lambda, float angle_degrees, float scale_db);
};

} // namespace torchdsp
} // namespace gr

#endif /* INCLUDED_TORCHDSP_PHASED_ARRAY_H */
