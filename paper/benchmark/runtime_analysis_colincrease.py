from mbpls.mbpls import MBPLS
import time
import rpy2

# define Parameters
rand_seed = 25

# initialize
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd

np.seterr(all='raise')


def mbpls_ade4(data, num_comp=20):
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    import time

    x1, x2, y = data

    pandas2ri.activate()  # to activate easy conversion from r to pandas dataframes

    # to generate a dataframe in R global environment
    robjects.globalenv['x1'] = pandas2ri.py2ri_pandasdataframe(x1)
    robjects.globalenv['x2'] = pandas2ri.py2ri_pandasdataframe(x2)
    robjects.globalenv['ref'] = pandas2ri.py2ri_pandasdataframe(y)

    start_time = time.time()
    # how to execute R code using the global environment variables defined above
    robjects.r(
        '''
        library(ade4)
        library(adegraphics)

        dudiY.act <- dudi.pca(ref, center = TRUE, scale = TRUE, scannf =
                                  FALSE,nf=1)
        ktabX.act <- ktab.list.df(list(mainPerformance=x1,mainInput=x2))
        resmbpls.act <- mbpls(dudiY.act, ktabX.act, scale = TRUE,
                                option = "none", scannf = FALSE, nf=''' + str(num_comp) + ''')

    bip <- resmbpls.act$bip
    ''')

    return start_time

###################################################################
# General settings
###################################################################

# How many runs?
reruns = 3

number_components = 20

feature_size_y = 10

standardize = True

# Should all parameters be calculated
calc_all = False

# Verbosity
# 0-show nothing
# 1-show datasets
# 2-show methods
# 3-show run information
# 4-show time of each run
verbosity_level = 4 #higher verbosity levels includer lower ones


# Calculation Measurements
include_timing = True

# Plot timing graphs in the end
plot_results = True

# Datasets
x_sizes = np.arange(1000, 1001, 1000)
y_sizes = np.arange(100, 20001, 100)


methods = [
    ('Ade4', 'Ade4'),
    ('Unipals', 'UNIPALS'),
    ('Kernel', 'KERNEL'),
    ('Nipals', 'NIPALS'),
    ('Simpls', 'SIMPLS')
]


###################################################################
# General settings - END
###################################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Dictionary to save timing
if include_timing:
    timing_results = np.zeros(shape=(len(methods), len(y_sizes), len(x_sizes), reruns))
    indicator_matrix = np.zeros(shape=(len(y_sizes), len(x_sizes)))

initiate_files = True
if initiate_files:
    pickle.dump(indicator_matrix, open('indicator_matrix_colincrease.pkl', 'wb'))
    pickle.dump(timing_results, open('timing_results_colincrease.pkl', 'wb'))

# Derive shapes from datasets
datasets_shape = []

finished = False

# load the indicator matrix that shows what has been done already
# Lock the access first
while not finished:
    while True:
        print('Loading indicator matrix data')
        indicator_matrix = pickle.load(open('indicator_matrix_colincrease.pkl', 'rb'))
        try:
            index = np.argwhere(indicator_matrix == 0)
            index = index[0]
            feature_index = index[0]
            sample_index = index[1]
            indicator_matrix[feature_index, sample_index] = 2
            pickle.dump(indicator_matrix, open('indicator_matrix_colincrease.pkl', 'wb'))
        except (ValueError, IndexError):
            print("Couldn't find indexes that are not finished")
            finished = True
            break
        print('Saved indicator matrix data')
        break

    sample_size = x_sizes[sample_index]
    feature_size = y_sizes[feature_index]

    datasets_shape.append(
        'X-shape: ' + str(sample_size) + 'x' + str(feature_size) + '\n Y-shape: ' + str(
            sample_size) + 'x' + str(feature_size_y))

    print('Loading data')
    timing_results_tmp = pickle.load(open('timing_results_colincrease.pkl', 'rb'))

    for run in range(reruns):
        if verbosity_level>2:
            print(bcolors.OKGREEN, "Starting run {:d} of {:d}".format(run+1, reruns), bcolors.ENDC)

        if verbosity_level > 0:
            print(bcolors.OKGREEN, "Creating dataset size {:d}x{:d}"
                  .format(sample_size, feature_size), bcolors.ENDC)

        dataset_x_1 = np.random.randint(1, 100, size=(sample_size, feature_size//2))
        dataset_x_2 = np.random.randint(1, 100, size=(sample_size, feature_size//2))
        dataset_y = np.random.randint(1, 100, size=(sample_size, feature_size_y))

        if verbosity_level > 0:
            print(bcolors.OKGREEN, "Starting calculations for dataset size {:d}x{:d}"
                  .format(sample_size, feature_size), bcolors.ENDC)

        for j, (methodname, method) in enumerate(methods):
            if methodname == 'Ade4':
                class_call = mbpls_ade4
            else:
                class_call = MBPLS(n_components=number_components, method=method, standardize=standardize,
                                   calc_all=calc_all)
            # It has to be checked if the methods still run in reasonable time
            if timing_results_tmp.mean(axis=3)[j].max() > 630:
                if include_timing:
                    timing_results[j, feature_index, sample_index, run] = 9999
                if verbosity_level > 1:
                    print(bcolors.OKGREEN, "Skipping calculating for {:s}-method, because it gets too slow now"
                          .format(methodname), bcolors.ENDC)
            else:
                if verbosity_level > 1:
                    print(bcolors.OKGREEN, "Calculating {:s}-method".format(methodname), bcolors.ENDC)
                if include_timing:
                    timing_results[j, feature_index, sample_index, run] = time.time()
                # The ade4-package might run into convergence problems, which result in numerical problems that cause
                # the package to crash. When this happens the code is restarted with a new random matrix.
                if method == 'Ade4':
                    count = 0
                    while True:
                        try:
                            results = class_call([pd.DataFrame(dataset_x_1), pd.DataFrame(dataset_x_2),
                                                  pd.DataFrame(dataset_y)],number_components)
                            if include_timing:
                                timing_results[j, feature_index, sample_index, run] = \
                                                                time.time() - results
                            if verbosity_level == 4:
                                print("Task finished in {:.2f} seconds".format(timing_results[j, feature_index,
                                                                                              sample_index, run]))
                            break
                        except:
                            count += 1
                            if count > 100:
                                print("Aborting. The method Ade4 has failed more than 100 times on the same task.")
                                timing_results[j, feature_index, sample_index, run] = 9999
                                break
                            else:
                                print("Tried. Trying again.")
                                dataset_x_1 = np.random.randint(1, 100, size=(sample_size, feature_size // 2))
                                dataset_x_2 = np.random.randint(1, 100, size=(sample_size, feature_size // 2))
                                dataset_y = np.random.randint(1, 100, size=(sample_size, feature_size_y))
                else:
                    results = class_call.fit([dataset_x_1, dataset_x_2], dataset_y)
                    if include_timing:
                        timing_results[j, feature_index, sample_index, run] = \
                                                        time.time() - timing_results[j, feature_index, sample_index,
                                                                                     run]

                    if verbosity_level == 4:
                        print("Task finished in {:.2f} seconds".format(timing_results[j, feature_index,
                                                                                      sample_index, run]))

        if verbosity_level>2:
            print(bcolors.OKGREEN, "Finished run {:d} of {:d}".format(run+1, reruns), bcolors.ENDC)


    print('Loading data')
    indicator_matrix_shared = pickle.load(open('indicator_matrix_colincrease.pkl', 'rb'))
    timing_results_shared = pickle.load(open('timing_results_colincrease.pkl', 'rb'))

    indicator_matrix_shared[feature_index, sample_index] = 1
    for i in range(len(methods)):
        timing_results_shared[i, feature_index, sample_index, :] = \
                                                        timing_results[i, feature_index, sample_index, :]


    print('Saving data')
    pickle.dump(indicator_matrix_shared, open('indicator_matrix_colincrease.pkl', 'wb'))
    pickle.dump(timing_results_shared, open('timing_results_colincrease.pkl', 'wb'))

# Generate the plot of the runtimes (Not shown in the paper)
if plot_results:
    import matplotlib
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    import matplotlib.pyplot as plt
    timing_mean = timing_results.min(axis=3)

    # Only include those results with runtimes lower than 1000 seconds in average
    Ade4 = timing_mean[0, timing_mean[0, :, 0] < 1000, 0]
    UNIPALS = timing_mean[1, timing_mean[1, :, 0] < 1000, 0]
    KERNEL = timing_mean[2, timing_mean[2, :, 0] < 1000, 0]
    NIPALS = timing_mean[3, timing_mean[3, :, 0] < 1000, 0]
    SIMPLS = timing_mean[4, timing_mean[4, :, 0] < 1000, 0]

    plt.figure(figsize=(6, 4))
    plt.plot(Ade4, 'b--', label=r'Ade4 ($P>N$)')
    plt.plot(UNIPALS, 'c', label='UNIPALS with fixed samples')
    plt.plot(KERNEL, 'g', label='KERNEL with fixed samples')
    plt.plot(NIPALS, 'r', label='NIPALS with fixed samples')
    plt.plot(SIMPLS, 'm', label='SIMPLS with fixed samples\n(Non-Multiblock')
    plt.legend(fontsize=13, loc=(0.1, 0.45))
    plt.xlabel(r'Number of samples $N$ in $\boldsymbol{X}$', fontsize=14)
    plt.ylabel('Time in seconds', fontsize=14)
    plt.ylim((0, 15))
    plt.xlim(0, 199)
    xticks = np.arange(-1, 200, 50)
    xticks[0] = -1
    xlabels = xticks*100 + 100
    xlabels[0] = 100
    plt.xticks(xticks, xlabels, fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.show()
