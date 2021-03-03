import argparse
from os.path import isfile
from crysremesh.triangular_grid import Grid
from crysremesh.io import read_tecplot, write_tecplot
from crysremesh.algorithms import *
from time import time


def choose_method(name):
    if name not in methods.keys():
        raise ValueError('Wrong parameter ')
    return methods[name]


def check_argument(name):
    if not isfile(name):
        print('File {} does not exist'.format(name))
        exit(1)
    else:
        if not name[-4:] == '.dat':
            print('File {} should be .dat file'.format(name))
            exit(1)


parser = argparse.ArgumentParser()
parser.add_argument('file', help='triangular mesh mesh .dat file')
parser.add_argument("-v", "--verbosity", action="count",
                    help="increase output verbosity", default=0)
args = parser.parse_args()

filename = args.file
check_argument(filename)

start = time()
grid = Grid()
read_tecplot(grid, filename)

if args.verbosity > 0:
    print('Mesh read')

outputfilename = filename[:-4] + "_final.dat"
write_tecplot(grid, outputfilename)

if args.verbosity > 0:
    print('Result grid was written into{}'.format(outputfilename))
    print('Total time:', time() - start)
