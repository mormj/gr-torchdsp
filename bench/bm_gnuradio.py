#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.0.0-git

from gnuradio import blocks
import torchdsp
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
import time

class benchmark_copy(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Benchmark Copy", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args.samples
        veclen = args.veclen
        self.actual_samples = actual_samples = int(nsamples /  veclen)
        num_blocks = args.nblocks

        ##################################################
        # Blocks
        ##################################################
        op_blocks = []
        for i in range(num_blocks):
            op_blocks.append(
                blocks.add_ff()
            )

        src = blocks.null_source(
            gr.sizeof_float*veclen)
        snk = blocks.null_sink(
            gr.sizeof_float*veclen)
        hd = blocks.head(
            gr.sizeof_float*veclen, actual_samples)

        ##################################################
        # Connections
        ##################################################
        self.connect(hd, snk)
        for idx, blk in enumerate(op_blocks):
            if (idx == 0):
                self.connect(src, (blk, 0))
            else:
                self.connect(op_blocks[idx-1], (blk, 0))

            ns = blocks.null_source(gr.sizeof_float*veclen)
            self.connect(ns, (blk, 1))


        self.connect((op_blocks[num_blocks-1], 0), hd)


def main(top_block_cls=benchmark_copy, options=None):

    parser = ArgumentParser(description='Run a flowgraph iterating over parameters for benchmarking')
    parser.add_argument('--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--samples', type=int, default=1e8)
    parser.add_argument('--operation', type=str, default='add')
    parser.add_argument('--nblocks', type=int, default=1)
    parser.add_argument('--veclen', type=int, default=1)

    args = parser.parse_args()
    print(args)

    if args.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    tb = top_block_cls(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("starting ...")
    startt = time.time()
    tb.start()

    tb.wait()
    endt = time.time()

    print(f'[PROFILE_TIME]{endt-startt}[PROFILE_TIME]')

if __name__ == '__main__':
    main()