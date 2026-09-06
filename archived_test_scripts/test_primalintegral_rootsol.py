import pathlib
import numpy as np
from utilities import instancetypes
import gzip
import pickle
import matplotlib.pyplot as plt

mode = 'improve-supportbinvars'
incumbent_mode = 'rootsol'
instancetype = 'setcovering-row5000col2000den0.01'
time_limit = 60

directory = './result/generated_instances/'+ instancetype +'/'+ mode +'/'

# load the opt solution data

# directory_opt =  './result/generated_instances/'+ instancetype + '/' + 'optimal-solution/'
# filename = f'{directory_opt}optimal-obj-time-' + incumbent_mode + '.pkl'
# with gzip.open(filename, 'rb') as f:
#     data_opt = pickle.load(f)
# objs_opt, times_opt = data_opt # objs_opt contains optimal obj over all the instances

# load the local branching test data
directory_lb_test = directory + 'lb-from-' + incumbent_mode + '-60s/'
primal_int_baselines = []
primal_int_preds = []
for i in range(10):
    instance_name = instancetype + '-' + str(i+100) # instance 100-199
    filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
    print(filename)
    sample_files = [str(path) for path in pathlib.Path(directory_lb_test).glob(filename)]
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    objs, times, objs_pred, times_pred = data # objs contains objs of a single instance of a lb test
    obj_opt = np.minimum(objs.min(), objs_pred.min())

    # compute primal gap for baseline localbranching run
    gamma_baseline = np.zeros(len(objs))
    for j in range(len(objs)):
        if objs[j] == 0 and obj_opt == 0:
            gamma_baseline[j]= 0
        elif objs[j] * obj_opt <0:
            gamma_baseline[j] = 1
        else:
            gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))
    # compute primal interal
    primal_int_baseline = 0
    for j in range(len(objs) - 1):
        primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
    primal_int_baselines.append(primal_int_baseline)

    # compute primal gap for k_init predicted localbranching run
    gamma_pred = np.zeros(len(objs_pred))
    for j in range(len(objs_pred)):
        if objs_pred[j] == 0 and obj_opt == 0:
            gamma_pred[j] = 0
        elif objs_pred[j] * obj_opt < 0:
            gamma_pred[j] = 1
        else:
            gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]), np.abs(obj_opt))
    # compute primal interal
    primal_int_pred = 0
    for j in range(len(objs_pred) - 1):
        primal_int_pred += gamma_pred[j] * (times_pred[j + 1] - times_pred[j])

    primal_int_preds.append(primal_int_pred)

    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6.4))
    fig.suptitle("Test Result: comparison of objective")
    fig.subplots_adjust(top=0.5)
    ax.set_title(instance_name, loc='right')
    ax.plot(times, objs, label='lb baseline')
    ax.plot(times_pred, objs_pred, label='lb with k predicted')
    ax.set_xlabel('time /s')
    ax.set_ylabel("objective")
    ax.legend()
    plt.show()

    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6.4))
    fig.suptitle("Test Result: comparison of primal gap")
    fig.subplots_adjust(top=0.5)
    ax.set_title(instance_name, loc='right')
    ax.plot(times, gamma_baseline, label='lb baseline')
    ax.plot(times_pred, gamma_pred, label='lb with k predicted')
    ax.set_xlabel('time /s')
    ax.set_ylabel("objective")
    ax.legend()
    plt.show()

primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
primal_int_preds = np.array(primal_int_preds).reshape(-1)

# avarage primal integral over test dataset
primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
primal_int_pred_ave = primal_int_preds.sum()/ len(primal_int_preds)
print('First Solution Primal integral:')
print('original lb: ', primal_int_base_ave)
print('k_init pred lb: ', primal_int_pred_ave)






