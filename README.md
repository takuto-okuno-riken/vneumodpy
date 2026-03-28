[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

# Virtual Neuromodulation Toolbox for Python
The Virtual Neuromodulation Toolbox for Python (MATLAB version is [here](https://github.com/takuto-okuno-riken/vneumod)).


## Introduction
To realize the digital brain, various approaches have been taken at multiple levels.
We proposed [Group Surrogate Data Generating Model (GSDGM)](https://github.com/takuto-okuno-riken/gsdgmpy), which does not incorporate structural connectivity and focuses on reproducing resting-state functional MRI dynamics (BOLD signal). 
This model can learn multivariate time-series data (BOLD signal) of a group and generate a centroidal and representative multivariate time-series of the group.
The group surrogate model preserves the statistics of multivariate time-series well, so this allows for the generation of whole-brain dynamics with extremely high accuracy. 

Whole-brain data-driven model (group surrogate model) assumes no specific brain tissue structure, one voxel of 4 × 4 × 4 mm gray matter, and 25445 voxels based on the Allen human brain atlas. 
All voxels are fully connected, and vector auto-regression (VAR) surrogate is used to learn group data and generate group surrogate data.

<div align="center">
<img src="data/img/fig1.jpg" width="70%">
</div>
<br>

Then, we extended output of VAR surrogate for virtual neuromodulation.
<div align="center">
<img src="data/img/fig2.jpg" width="30%">
</div>


where x<sub>i</sub>(t) is output of voxel i, y<sub>i</sub>(t) is VAR surrogate output of voxel i, z<sub>i</sub>(t),u<sub>i</sub>(t) are modulation terms of voxel i, and c<sub>i</sub>,σ<sub>X</sub>∈R. 
σ<sub>X</sub> is calculated as the standard deviation from the entire voxel time-series.
z<sub>i</sub>(t) is constructed by convolution of the canonical Hemodynamic Response Function (HRF) and the Box-car task design.
u<sub>i</sub>(t) performs direct adjustment to the output of VAR surrogate.
In other words, if there is a virtual neuromodulation stimulus, it adjusts to prioritize neuromodulation stimulus, such as u<sub>i</sub>(t)=1-0.5∙z<sub>i</sub>(t).
Using the above-mentioned virtual neuromodulation, BOLD signal addition, i.e., DBS treatment, can be virtually performed for specific voxels. 


<b>Command line tools</b>

| name | description |
|:---|:---|
| gsdgm | Generate a whole-brain data-driven model based on the group surrogate model (VAR surrogate).|
| vneumod | Generate virtual neuromodulation time-series surrogate data based on the group surrogate model.|
| mtess | Calculate and plot MTESS for a group of multivariate time-series data. |


## Requirements: software
* Python 3.11.14
* matplotlib 3.10.8
* scikit_learn 1.8.0
* h5py 3.15.1
* pandas 2.3.3
* statsmodels 0.14.6


## Installation
1. Download this [Toolbox](https://github.com/takuto-okuno-riken/vneumodpy/archive/refs/heads/main.zip) zip files.
2. Extract zip file under your working directory <work_path>/vneumodpy-main.
3. This is not required, but we recommend using the conda virtual environment.
~~~
(base) work_path>cd vneumodpy-main
(base) vneumodpy-main>conda create -n vneumod python=3.11.14
...
(base) vneumodpy-main>conda activate vneumod
(vneumod) vneumodpy-main>
~~~
4. Install several packages.
~~~
(vneumod) vneumodpy-main>pip install -r requirements.txt
...
~~~
5. Run the following demos.


## Command Line Tools Demos
<b>Demo 1</b><br>




## Command Line Tools Reference
<b>gsdgm command</b><br>
~~~
(vneumod) vneumodpy-main>python gsdgm.py -h
usage: gsdgm.py [-h] [--var] [--lag LAG] [--noise NOISE] [--outpath OUTPATH] [--transopt TRANSOPT]
                [--format FORMAT] [--surrnum SURRNUM] [--siglen SIGLEN] [--range RANGE] [--cache]
                [--njobs NJOBS] [--showinsig] [--showinras] [--showsig] [--showras]
                filename [filename ...]

positional arguments:
  filename             filename of node status time-series (node x frames)

options:
  -h, --help           show this help message and exit
  --var                output Vector Auto-Regression (VAR) group surrogate model
                       (<filename>_gsm_var.mat)
  --lag LAG            time lag <num> for VAR (default:1)
  --noise NOISE        noise type for VAR surrogate model (default:"gaussian" or "residuals")
  --outpath OUTPATH    output files path (default:"results")
  --transopt TRANSOPT  signal transform option (for type 1:centroid value)
  --format FORMAT      save file format <type> 0:csv, 1:mat(each), 2:mat(all) (default:2)
  --surrnum SURRNUM    output surrogate sample number <num> (default:0)
  --siglen SIGLEN      output time-series length <num> (default:same as input time-series)
  --range RANGE        output surrogate value range (default:"auto", sigma:<num>, full:<num>,
                       <min>:<max> or "none")
  --cache              save cache file at model calculation
  --njobs NJOBS        number of jobs (multiprocessing) for model calculation (default:-1)
  --showinsig          show input time-series data of <filename>.csv
  --showinras          show raster plot of input time-series data of <filename>.csv
  --showsig            show output surrogate time-series data
  --showras            show raster plot of output surrogate time-series data
~~~
The input .mat file should include input cell data described as follows. The node count must be the same within the group, whereas the time-series length does not have to be the same.
| name | cell | description |
|:---|:---|:---|
|CX |{&lt;nodes&gt; x &lt;length&gt;} x &lt;cell number&gt; |group of multivariate time-series|
|names |{'data name string'} x &lt;cell number&gt; |names of each time-series data|

The output (group surrogate model) .mat file includes the following struct data:

| name | type | description |
|:---|:---|:---|
|net | struct |struct of group surrogate model|
|gRange | struct |struct of group range information|
|name | string |name of group surrogate model|

The output (group surrogate data) .mat file includes the following cell data:

| name | cell | description |
|:---|:---|:---|
|CX |{&lt;nodes&gt; x &lt;length&gt;} x &lt;cell number&gt; |group of multivariate time-series|
|names |{'data name string'} x &lt;cell number&gt; |names of each time-series data|


##
<b>vneumod command</b><br>
~~~
(vneumod) vneumodpy-main>python vneumod.py -h
usage: vneumod.py [-h] [--cx CX] [--pymodel PYMODEL] [--model MODEL] [--atlas ATLAS]
                  [--targatl TARGATL] [--roi ROI] [--out OUT] [--outfrom OUTFROM]
                  [--surrnum SURRNUM] [--srframes SRFRAMES] [--vnparam VNPARAM] [--tr TR]
                  [--hrfparam HRFPARAM] [--glm] [--njobs NJOBS] [--outpath OUTPATH] [--nocache]
                  [filename ...]

positional arguments:
  filename             filename of subject permutation (1 x length)

options:
  -h, --help           show this help message and exit
  --cx CX              set cells of subject time-series (<filename>.mat)
  --pymodel PYMODEL    set (VAR) group surrogate model <path> by vneumodpy
  --model MODEL        set (VAR) group surrogate model (<filename>_gsm_var.mat)
  --atlas ATLAS        set cube atlas nifti file (<filename>.nii.gz)
  --targatl TARGATL    set modulation target atlas nifti file (<filename>.nii.gz)
  --roi ROI            set modulation target ROI <num> or <range text>
  --out OUT            set output trials (perm & surrogate files) number <num> (default:1)
  --outfrom OUTFROM    set surrogate output from <num> (default:1)
  --surrnum SURRNUM    output surrogate sessions per one file <num> (default:40)
  --srframes SRFRAMES  output surrogate frames <num> (default:160)
  --vnparam VNPARAM    set virtual neuromodulation params <num,num,num> (default:28,22,0.15)
  --tr TR              set TR (second) of fMRI time-series <num> (default:1)
  --hrfparam HRFPARAM  set HRF (for convolution) params <num,num> (default:16,8)
  --glm                output GLM result nifti file
  --njobs NJOBS        number of jobs (multiprocessing) for glm calculation (default:8)
  --outpath OUTPATH    output files path (default:"results")
  --nocache            do not output surrogate file
~~~
The input .mat files are optional. It should include subject permutation data for surrogate data generation:
| name | matrix | description |
|:---|:---|:---|
|perm |&lt;1&gt; x &lt;length&gt; | time-series permutation order|

The output will be T-value 3D matrix nifti file (GLM result) aligned with cube atlas nifti file.

##
<b>mtess command</b><br>
~~~
(vneumod) vneumodpy-main>python mtess.py -h
usage: mtess.py [-h] [--range RANGE] [--aclag ACLAG] [--paclag PACLAG] [--cclag CCLAG]
                [--pcclag PCCLAG] [--outpath OUTPATH] [--format FORMAT] [--transform TRANSFORM]
                [--transopt TRANSOPT] [--showinsig] [--showinras] [--showmat] [--showsig]
                [--showprop] [--shownode] [--showdend SHOWDEND] [--cache] [--cachepath CACHEPATH]
                filename [filename ...]

positional arguments:
  filename              filename of node status time-series (node x frames)

options:
  -h, --help            show this help message and exit
  --range RANGE         input group value range (default:"auto", sigma:<num>, full:<num> or
                        <min>:<max>)
  --aclag ACLAG         time lag <num> for Auto Correlation (default:5)
  --paclag PACLAG       time lag <num> for Partial Auto Correlation (default:13)
  --cclag CCLAG         time lag <num> for Cross Correlation (default:2)
  --pcclag PCCLAG       time lag <num> for Partial Cross Correlation (default:4)
  --outpath OUTPATH     output files path (default:"results")
  --format FORMAT       save file format <type> 0:csv, 1:mat (default:1)
  --transform TRANSFORM
                        input signal transform 0:raw, 1:sigmoid (default:0)
  --transopt TRANSOPT   signal transform option (for type 1:centroid value)
  --showinsig           show input time-series data of <filename>.csv
  --showinras           show raster plot of input time-series data of <filename>.csv
  --showmat             show result MTESS matrix
  --showsig             show 1 vs. others node signals
  --showprop            show result polar chart of 1 vs. others MTESS statistical properties
  --shownode            show result line plot of 1 vs. others node MTESS
  --showdend SHOWDEND   show dendrogram of <algo> hierarchical clustering based on MTESS matrix.
  --cache               use cache file for MTESS calculation
  --cachepath CACHEPATH
                        cache files <path> (default:"results/cache")
~~~
The input .mat file should include input cell data. The node count must be the same within the group, whereas time-series length does not have to be the same.
| name | cell | description |
|:---|:---|:---|
|CX |{&lt;nodes&gt; x &lt;length&gt;} x &lt;cell number&gt; |group of multivariate time-series|
|names |{'data name string'} x &lt;cell number&gt; |names of each time-series data|

The output .mat file includes the following matrix data:

| name | matrix | description |
|:---|:---|:---|
|MTS |&lt;cell number&gt; x &lt;cell number&gt; | MTESS matrix (2D)|
|MTSp |&lt;cell number&gt; x &lt;cell number&gt; x 8| MTESS statistical property matrix (3D)|
|nMTS |&lt;cell number&gt; x &lt;cell number&gt; x &lt;nodes&gt;| Node MTESS matrix (3D)|
|nMTSp |&lt;cell number&gt; x &lt;cell number&gt; x &lt;nodes&gt; x 8| Node MTESS statistical property matrix (4D)|

Similarities are generated for the following 8 statistical properties: mean, standard deviation, DFT amplitude, correlation, partial correlation, cross-correlation and partial cross-correlation.


## Citing Virtual Neuromodulation Toolbox
If you find Virtual Neuromodulation Toolbox useful in your research, please cite it as follows: 
