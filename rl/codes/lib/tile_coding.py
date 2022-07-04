#######################################################################
# Copyright (C)                                                       #
# Rich Sutton                                                         #
# 2022 umiushio (umiushio@163.com)                                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Tile Coding Library
# adapted from Tile Coding Software version 3.0beta by Rich Sutton
# based on a program created by Steph Schaeffer and others

# This is an implementation of grid-style tile codings, based originally on
# the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.

from math import floor
import numpy as np


# compute the power exponent based on base that greater or equal to the given number
# @low_bound: the infimum
# @base: base, default is 2
def min_power_exponent(low_bound, base=2):
    i = 1
    while i < low_bound:
        i *= base
    return i


# compute the recommended max size of tile code with k-dimension
# @dimension: the dimension k
# @ints: other factors
def compute_max_size(dimension, ints=None):
    width = min_power_exponent(4 * dimension)
    max_size = np.power(width, dimension + 1)
    if ints is not None:
        for i in ints:
            max_size *= min_power_exponent(i)
    return max_size


class IndexHashTable:
    def __init__(self, max_size):
        self.size = max_size
        self.dictionary = {}
        self.overflow_cnt = 0

    def __str__(self):
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overflow count:" + str(self.overflow_cnt) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        else:
            count = self.count()
            size = self.size
            if count >= size:
                if self.overflow_cnt == 0:
                    print('Index Hash Table is full, starting to allow collisions')
                self.overflow_cnt += 1
                return hash(obj) % size
            else:
                d[obj] = count
                return count


# return the index of coordinates in the iht or the integer size or nil
# @coordinates: the coordinates need to be indexed by hash
# @iht_or_size: iht or integer size
def hash_coordinates(coordinates, iht_or_size, read_only=False):
    if isinstance(iht_or_size, IndexHashTable):
        return iht_or_size.get_index(tuple(coordinates), read_only)
    elif isinstance(iht_or_size, int):
        return hash(tuple(coordinates)) % iht_or_size
    else:
        return coordinates


# maps floating and integer variables to a list of tiles
# @iht_size: either an index hash table of a given size (created by (make-iht size)),
#   an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
#   coordinates are to be returned without being converted to indices).
# @num_of_tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
#   also be greater than or equal to four times the number of floats.
# @floats: these variables will be gridded at unit intervals, so generalization
#   will be by approximately 1 in each direction, and any scaling will have
#   to be done externally before calling tiles.
# @ints: optional factors for tiles
def tiles(iht_or_size, num_of_tilings, floats, ints=[], read_only=False):
    q_floats = [floor(f * num_of_tilings) for f in floats]
    tiles = []
    for tiling in range(num_of_tilings):
        offset = tiling
        coordinates = [tiling]
        for q in q_floats:
            coordinates.append((q + offset) // num_of_tilings)
            offset += 2 * tiling
        coordinates.extend(ints)
        tiles.append(hash_coordinates(coordinates, iht_or_size, read_only))
    return tiles

