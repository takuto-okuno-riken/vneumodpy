import argparse


class ParseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = []

    def initialize(self):
        alist = lambda x: list(map(float, x.split(',')))
        self.parser.add_argument('in_files', metavar='filename', nargs='*', help='filename of subject permutation (1 x length)')
        self.parser.add_argument('--cx', type=str, nargs=1, default='', help='set cells of subject time-series (<filename>.mat)')
        self.parser.add_argument('--model', type=str, nargs=1, default='', help='set (VAR) group surrogate model (<filename>_gsm_var.mat)')
        self.parser.add_argument('--atlas', type=str, nargs=1, default='', help='set cube atlas nifti file (<filename>.nii.gz)')
        self.parser.add_argument('--targatl', type=str, nargs=1, default='', help='set modulation target atlas nifti file (<filename>.nii.gz)')
        self.parser.add_argument('--roi', type=str, nargs=1, default='', help='set modulation target ROI <num> or <range text>')
        self.parser.add_argument('--out', type=int, default=1, help='set output perm & surrogate files number <num> (default:1)')
        self.parser.add_argument('--outfrom', type=int, default=1, help='set surrogate output from <num> (default:1)')
        self.parser.add_argument('--surrnum', type=int, default=40, help='output surrogate sessions per one file <num> (default:40)')
        self.parser.add_argument('--srframes', type=int, default=160, help='output surrogate frames <num> (default:160)')
        self.parser.add_argument('--vnparam', type=alist, default=[28,22,0.15], help='set virtual neuromodulation params <num,num,num> (default:28,22,0.15)')
        self.parser.add_argument('--tr', type=float, default=1.0, help='set TR (second) of fMRI time-series <num> (default:1)')
        self.parser.add_argument('--hrfparam', type=alist, default=[16,8], help='set HRF (for convolution) params <num,num> (default:16,8)')
        self.parser.add_argument('--glm', action='store_true', help='output GLM result nifti file')
        self.parser.add_argument('--outpath', nargs=1, default='results', help='output files path (default:"results")')
        self.parser.add_argument('--nocache', action='store_true', help='do not output surrogate file')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

