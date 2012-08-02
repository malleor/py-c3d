#!/usr/bin/python

import sys
import numpy
import logging

from lmj import c3d

if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format=('%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)

    with open(sys.argv[1], 'rb') as h:
        r = c3d.Reader(h)
        logging.info('%d points in this file',
                     sum(p.size for p, a in r.read_frames()))

    with open('write_test.c3d', 'wb') as h:
        frame = (numpy.array([[1, 2, 3, 4]] * 50, 'd'), [])
        w = c3d.Writer(h)
        w.write_like_phasespace([frame] * 50, 50)
        logging.info('Written to file \'%s\'.', h.name)
