from mbpls.mbpls import MBPLS
from sklearn import metrics
import scipy

# If running on server
import matplotlib
matplotlib.use('QT5Agg')
import math
import time


from mbpls.data.get_data import data_path
# def loaddata():
#     from scipy.io import loadmat
#     data = loadmat(os.path.join(data_path(), 'MBdata.mat'))
#     y = data['Y']
#     x1 = data['X'][:, :50]
#     x2 = data['X'][:, 50:]
#     return x1, x2, y

# define Parameters
rand_seed = 25
num_samples = 400
num_vars_x1 = 200
num_vars_x2 = 200
noise = 5      # add noise between 0..10

# initialize
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
sb.set_style(style="whitegrid")
plt.close('all')

print('start loading')
# Generate Loadings
np.random.seed(rand_seed)
loading1 = np.expand_dims(np.random.randint(0, 10, num_vars_x1), 1)
loading2 = np.expand_dims(np.sin(np.linspace(0, 5, num_vars_x2)), 1)

print('start orthogonalizing')
# Generate orthogonal Scores
from scipy.stats import ortho_group
# y = ortho_group.rvs(num_samples, random_state=rand_seed)[:, :2]
y = np.random.rand(num_samples, 2)

print('start generating data')
# Generate data from scores and loadings
x1 = np.dot(y[:, 0:1], loading1.T)
x2 = np.dot(y[:, 1:2], loading2.T)

# Add noise to x1 and x2 (orthogonality of Latent Variable structure will be destroyed)
x1 = np.random.normal(x1, 0.05*noise)
x2 = np.random.normal(x2, 0.05*noise)


np.seterr(all='raise')

###################################################################
# General settings
###################################################################

# How many runs?
reruns = 1

number_components = 20

standardize = True

# Should all parameters be calculated
calc_all = True

#0-show nothing 1-show datasets 2-show methods 3-show run information
verbosity_level = 3 #higher verbosity levels includer lower ones
plot_graph = True
savefigs = True

# Metrics
include_metrics = False
plot_metrics = False
# Metrics dependent on Y will only be calculated when Y is available for that network
if include_metrics:
    metric_list = [metrics.adjusted_rand_score]
    metricname_list = ['Adjusted_RAND-Index']
    metric_Y_dependency = [True]

# Calculation Measurements
include_timing = True
plot_timing = True
# Create a combined line graph showing all chosen methods over all datasets (please import datasets in growing order)
plot_timing_graph = True

#x1, x2, y = loaddata()

# Datasets is a list of tuples
datasets = [
    #([x1, x2], y[:, 0:1]),
    #([x1, x2], y[:, 0:3]),
    #([x1 for i in range(5000)], y[:, 0:3]),
    #([np.repeat(x1, 200000, axis=0)], np.repeat(y[:, 0:3], 200000, axis=0)),
    #([np.repeat(np.repeat(x1, 20, axis=0), 40000, axis=1)], np.repeat(y[:, 0:3], 20, axis=0)),
    #([np.repeat(np.repeat(x1, 200, axis=0), 40, axis=1)], np.repeat(np.repeat(y[:, 0:3], 200, axis=0), 10, axis=1)),
    #([np.repeat(np.repeat(x1, 20, axis=0), 2, axis=1), np.repeat(np.repeat(x1, 20, axis=0), 2, axis=1)], np.repeat(y[:, 0:3], 20, axis=0)),
    #([np.repeat(np.repeat(x1, 10, axis=0), 2, axis=1)], np.repeat(y[:, 0:3], 10, axis=0)),
    #([np.repeat(np.repeat(x1, 20, axis=0), 4, axis=1)], np.repeat(y[:, 0:3], 20, axis=0)),
    #([np.repeat(np.repeat(x1, 40, axis=0), 8, axis=1)], np.repeat(y[:, 0:3], 40, axis=0)),
    #([np.repeat(np.repeat(x1, 80, axis=0), 16, axis=1)], np.repeat(y[:, 0:3], 80, axis=0)),
    #([np.repeat(np.repeat(x1, 160, axis=0), 32, axis=1)], np.repeat(y[:, 0:3], 160, axis=0)),
    #([np.repeat(np.repeat(x1, 320, axis=0), 64, axis=1)], np.repeat(y[:, 0:3], 320, axis=0)),
    ([x1, x2], y),
    ([x1, x2], y),
]

methods = [
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
    timing_results = np.zeros(shape=(reruns, len(datasets), len(methods)))


# Create array to save metric resuls
if include_metrics:
    metric_results = np.zeros(shape=(reruns, len(metric_list), len(methods), len(datasets)))

# Derive shapes from datasets
datasets_shape = []
for dataset in datasets:
    datasets_shape.append('X-shape: ' + str(np.hstack(dataset[0]).shape) + '\n Y-shape: ' + str(dataset[1].shape))

for k, dataset in enumerate(datasets):
    if verbosity_level>0:
        print(bcolors.OKGREEN, "Starting calculations on dataset #{:d}".format(k), bcolors.ENDC)
    for j, (methodname, method) in enumerate(methods):
        class_call = MBPLS(n_components=number_components, method=method, standardize=standardize, calc_all=calc_all)
        if verbosity_level>1:
            print(bcolors.OKGREEN, "Calculating {:s}-method".format(methodname), bcolors.ENDC)
        for run in range(reruns):
            if verbosity_level>2:
                print(bcolors.OKGREEN, "Starting run {:d} of {:d}".format(run+1, reruns), bcolors.ENDC)

            if include_timing:
                timing_results[run, k, j] = time.time()
            results = class_call.fit(dataset[0], dataset[1])

            if include_timing:
                timing_results[run, k, j] = time.time() - timing_results[run, k, j]

            if verbosity_level>2:
                print(bcolors.OKGREEN, "Finished run {:d} of {:d}".format(run+1, reruns), bcolors.ENDC)

            # ## COMPUTE METRICS
            if include_metrics:
                for l, metric in enumerate(metric_list):
                    if metric_Y_dependency[l] and dataset[1] is None:
                        metric_results[run, l, j, k] = None
                    else:
                        if metric_Y_dependency[l]:
                            metric_results[run, l, j, k] = metric(dataset[1], results.predict(dataset[0]))
                        else:
                            metric_results[run, l, j, k] = metric(results.predict(dataset[0]))

if plot_metrics:
    if reruns == 1:
        for run in range(reruns):
            for l, metric in enumerate(metric_list):
                fig = plt.figure()
                fig.subplots_adjust(left=0.2, top=0.8, wspace=1)
                ax = plt.subplot2grid((10, 10), (1, 1 + max([math.ceil(len(method[0]) / 10) for method in methods])), colspan=8,
                                      rowspan=8)
                ax.table(cellText=np.round(metric_results[run, l], 3),
                         rowLabels=[method[0] for method in methods],
                         colLabels=tuple(datasets_shape),
                         loc='center', clip_on=True,
                         cellLoc='center')
                ax.axis('off')
                ax2 = plt.subplot2grid((10, 10), (9, 2), colspan=8, rowspan=1)
                ax2.text(0, 0, 'Average results over\nnan indicates that the ground truth was not available')
                ax2.axis('off')
                plt.tight_layout()
                fig.set_size_inches(w=6, h=5)
                plt.suptitle("\nResults for {:s} of run {:d}".format(metricname_list[l], run + 1))
                if savefigs:
                    plt.savefig("./img/{:s}_run_{:d}".format(metricname_list[l], run + 1))

    # Plot total statistics about the results
    else:
        clustering_results_total = np.moveaxis(metric_results, 0, 3)
        clustering_results_stats = np.empty(shape=(clustering_results_total.shape[:-1] + (3,)))
        for k in range(clustering_results_total.shape[0]):
            results = []
            # l for cut and m for dataset
            for l in range(clustering_results_total.shape[1]):
                results.append([])
                for m in range(clustering_results_total.shape[2]):
                    try:
                        mean, _, _ = scipy.stats.bayes_mvs(clustering_results_total[k, l, m, :], alpha=0.95)
                    except:
                        clustering_results_total[k, l, m, 0] = clustering_results_total[k, l, m, 0] + 0.000001
                        mean, _, _ = scipy.stats.bayes_mvs(clustering_results_total[k, l, m, :], alpha=0.95)
                    results[-1].extend(["{:.3f}\n({:.3f}, {:.3f})".format(mean[0], mean[1][0], mean[1][1])])
                    clustering_results_stats[k, l, m, 0] = mean[0]  # mean
                    clustering_results_stats[k, l, m, 1] = mean[1][0]  # mean lower confidence interval
                    clustering_results_stats[k, l, m, 2] = mean[1][1]  # mean upper confidence interval

            fig = plt.figure()
            fig.subplots_adjust(left=0.2, top=0.8, wspace=1)
            ax = plt.subplot2grid((10, 10), (1, 1 + max([math.ceil(len(method[0]) / 10) for method in methods])), colspan=8,
                                  rowspan=8)
            tab = ax.table(cellText=results,
                           rowLabels=[method[0] for method in methods],
                           colLabels=tuple(datasets_shape),
                           loc='center', clip_on=True,
                           cellLoc='center')
            tab.scale(1, 2)
            ax.axis('off')
            ax2 = plt.subplot2grid((10, 10), (9, 2), colspan=8, rowspan=1)
            ax2.text(0, 0,
                     '( . , . ) values indicate 95% confidence inverval\nnan indicates that the ground truth was not available')
            ax2.axis('off')
            plt.tight_layout()
            fig.set_size_inches(w=6, h=5)
            plt.suptitle("\nAverage results for {:s} for {:d} runs".format(metricname_list[k], reruns))
            if savefigs:
                plt.savefig("./img/{:s}_average_{:d}runs_stats".format(metricname_list[k], reruns))

if plot_timing:
    for k in range(len(datasets)):
        plt.figure(figsize=(6, 8))
        if reruns == 1:
            plt.bar([method[0] for method in methods], list(timing_results[0, k, :]))
        else:
            plt.boxplot(timing_results[:, k, :], labels=[method[0] for method in methods])
        plt.ylabel('seconds')
        plt.xlabel('Method')
        plt.xticks(rotation=80)
        plt.ylim(bottom=0)
        plt.title("\nTiming results on dataset #{:d} for {:d} repeated runs\n"
                     "{:s}\n"
                     "Number of latent variables: {:d}\n"
                  "All parameters calculated: {}".format(k + 1, reruns, datasets_shape[k], number_components, calc_all))
        plt.tight_layout()

        plt.show()
        if savefigs:
            plt.savefig("./Timing_Dataset{:d}_reruns-{:d}".format(k+1, reruns))

if plot_timing_graph:
    # timing_results shape is [runs, datasets, methods]
    timing_results_total = np.moveaxis(timing_results, [0, 1, 2], [2, 1, 0])
    # timing_results shape is now [methods, datasets, runs]
    timing_results_mean = timing_results_total.mean(axis=2)
    plt.figure(figsize=(11, 8))
    for k in range(timing_results_total.shape[0]):
        plt.plot(np.arange(len(timing_results_mean[k])), timing_results_mean[k], '-o', label=methods[k][0])
    plt.ylabel('seconds')
    plt.xlabel('Dataset size')
    plt.xticks(np.arange(len(datasets_shape)), datasets_shape)
    plt.xticks(rotation=80)
    plt.legend()
    plt.ylim(ymin=0)
    plt.title("\nTiming results based on the mean of {:d} runs\n"
                 "Number of latent variables: {:d}".format(reruns, number_components))
    #plt.tight_layout(rect=[0, 0.03, 1, 0.5])
    plt.tight_layout()

    plt.show()
    if savefigs:
        plt.savefig("./Timing_Dataset_plot".format(k+1, reruns))



# old table style
"""# timing results axes are [runs, dataset, method]
timing_results_total = np.moveaxis(timing_results, 0, 2)
# Now it is [dataset, method, runs]
timing_results_stats = np.empty(shape=(timing_results_total.shape[:-1] + (3,)))
# k for each dataset
for k in range(timing_results_total.shape[0]):
    results = []
    # l for each method
    for l in range(timing_results_total.shape[1]):
        results.append([])
        try:
            mean, _, _ = scipy.stats.bayes_mvs(timing_results_total[k, l, :], alpha=0.95)
        except:
            timing_results_total[k, l, 0] = timing_results_total[k, l, 0] + 0.000001
            mean, _, _ = scipy.stats.bayes_mvs(timing_results_total[k, l, :], alpha=0.95)
        results[-1].extend(["{:.3f}\n({:.3f}, {:.3f})".format(mean[0], mean[1][0], mean[1][1])])
        timing_results_stats[k, l, 0] = mean[0]  # mean
        timing_results_stats[k, l, 1] = mean[1][0]  # mean lower confidence interval
        timing_results_stats[k, l, 2] = mean[1][1]  # mean upper confidence interval

    fig = plt.figure()
    fig.subplots_adjust(left=0.2, top=0.8, wspace=1)
    ax = plt.subplot2grid((10, 10), (1, 1 + max([math.ceil(len(method[0]) / 10) for method in methods])),
                          colspan=8,
                          rowspan=8)
    tab = ax.table(cellText=results,
                   rowLabels=[method[0] for method in methods],
                   colLabels=tuple([datasets_shape[k]]),
                   loc='center', clip_on=True,
                   cellLoc='center')
    tab.scale(1, 2)
    ax.axis('off')
    ax2 = plt.subplot2grid((10, 10), (9, 2), colspan=8, rowspan=1)
    ax2.text(0, 0,
             '( . , . ) values indicate 95% confidence inverval\nnan indicates that the ground truth was not available')
    ax2.axis('off')
    plt.tight_layout()
    fig.set_size_inches(w=6, h=5)
    plt.suptitle("\nTiming results on dataset #{:d} for {:d} repeated runs".format(k+1, reruns))
    plt.show()
    if savefigs:
        plt.savefig("./Dataset-{:d}_average_{:d}runs_stats".format(k+1, reruns))"""