import argparse
import os
from crysremesh.triangular_grid import Grid
from crysremesh.io import read_tecplot, write_tecplot
from crysremesh.algorithms import *
from time import time


def choose_method(name):
    if name not in methods.keys():
        raise ValueError('Wrong parameter ')
    return methods[name]


def check_argument(name):
    if not os.path.isfile(name):
        print('File {} does not exist'.format(name))
        exit(1)
    else:
        if not name[-4:] == '.dat':
            print('File {} should be .dat file'.format(name))
            exit(1)


outdir = '.'

parser = argparse.ArgumentParser()
parser.add_argument('workdir', help='all grids in this dir will be smoothed')
parser.add_argument('-o', '--outdir', help='all result grids will be stored here')
parser.add_argument("-v", "--verbosity", action="count",
                    help="increase output verbosity", default=0)
args = parser.parse_args()

workdir = args.workdir

if args.outdir:
    outdir = args.outdir

meshes = [path for path in os.listdir(workdir)]

start = time()

for mesh in meshes:
    grid = Grid()
    read_tecplot(grid, mesh)
    if args.verbosity > 0:
        print('Mesh read')
    filename = mesh[mesh.rfind('/'):]
    outputfilename = outdir + '/' + mesh.replace('_r_', '_')
    write_tecplot(grid, outputfilename)

    if args.verbosity > 0:
        print('Result grid was written into{}'.format(outputfilename))

if args.verbosity > 0:
    print('Total time:', time() - start)
