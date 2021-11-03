from pyscipopt import Model
#
#
# model = Model("MILP")
#
# x = model.addVar("x")
# y = model.addVar("y", vtype="INTEGER")
# model.setObjective(x + y)
# cons = model.addCons(2*x - y*y >= 0, )
# model.optimize()
# lp_sol = model.createLPSol()
# sol = model.getBestSol()
# print("Model MILP is solved ")
# print("x: {}".format(sol[x]))
# print("y: {}".format(sol[y]))
import matplotlib.pyplot as plt
import numpy

from geco.mips.loading.miplib import Loader

# miplib2017_binary39 = ['10teams',
#                      'a1c1s1',
#                      'aflow30a',
#                      'aflow40b',
#                      'air04',
#                      'air05',
#                      'cap6000',
#                      'dano3mip',
#                      'danoint',
#                      'ds',
#                      'fast0507',
#                      'fiber',
#                      'glass4',
#                      'harp2',
#                      'liu',
#                      'misc07',
#                      'mkc',
#                      'mod011',
#                      'momentum1',
#                      'net12',
#                      'nsrand-ipx',
#                      'nw04',
#                      'opt1217',
#                      'p2756',
#                      'protfold',
#                      'qiu',
#                      'rd-rplusc-21',
#                      'seymour',
#                      'sp97ar',
#                      'swath',
#                      't1717',
#                      'tr12-30',
#                      'van'
#                      ]

miplib2017_binary39 = ['10teams',
                     'a1c1s1',
                     'aflow30a',
                     'aflow40b',
                     'air04',
                     'air05',
                     'cap6000',
                     'dano3mip',
                     'danoint',
                     'ds',
                     'fast0507',
                     'fiber',
                     'glass4',
                     'harp2',
                     'liu',
                     'misc07',
                     'mkc',
                     'mod011',
                     'momentum1',
                     'net12',
                     'nsrand-ipx',
                     'nw04',
                     'opt1217',
                     'p2756',
                     'protfold',
                     'qiu',
                     'rd-rplusc-21',
                     'seymour',
                     'sp97ar',
                     'swath',
                     't1717',
                     'tr12-30',
                     'van'
                     ]

# numpy.savez('./result/miplib2017/miplib2017_binary39', miplib2017_binary39 = miplib2017_binary39)
# data = numpy.load('./result/miplib2017/miplib2017_binary39.npz')
# miplib2017_binary = data['miplib2017_binary39']
#
# for i in range(len(miplib2017_binary)):
#     instance = Loader().load_instance(miplib2017_binary[i]+'.mps.gz')
#     MIP_model = instance
#     print(MIP_model.getProbName())

# print repair violation
# print(len(miplib2017_binary39))
data = numpy.load('./result/miplib2017/miplib2017_binary39.npz')
miplib2017_binary39 = data['miplib2017_binary39']

modes = ['repair-slackvars', 'repair-supportbinvars', 'repair-binvars', 'improve-supportbinvars', 'improve-binvars']
plottitles = ["LB to repair (over slack variables)", "LB to repair (over support of binary vars)", "LB to repair (over binary variables)", "LB to improve (over support of binary vars)", "LB to improve (over binary variables)"]
lb_neigh_basis = ['violations','supportofbins','binvars', 'supportofbins','binvars']

print(len(miplib2017_binary39))

for i in range(len(miplib2017_binary39)): #
    instance_name = miplib2017_binary39[i]
    # if instance_name == 'mkc':

    # for each instance: plot the result of each mode
    for j in range(4,5): #len(modes)
        mode = modes[j]
        directory = './result/miplib2017/' + mode + '/'
        data = numpy.load(directory + instance_name + '.npz')
        neigh_sizes = data['neigh_sizes']
        t = data['t']
        objs = data['objs']

        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
        fig.suptitle(plottitles[j])
        fig.subplots_adjust(top=0.5)
        ax[0].plot(neigh_sizes, objs)
        ax[0].set_title(instance_name, loc='right')
        ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{' + lb_neigh_basis[j] + '}$)')
        ax[0].set_ylabel('Objective')
        ax[1].plot(neigh_sizes, t)
        # ax[1].set_ylim([0, 31])
        ax[1].set_ylabel("Solving time")
        ax[0].grid()
        ax[1].grid()
        plt.show()


# modes = ['repair-nviolations','repair-nbinvars','improve']
# mode = modes[0]
#
# if mode == 'repair-nviolations':
#     directory = './result/miplib2017/repair/timesofviolations/'
# elif mode == 'repair-nbinvars':
#     directory = './result/miplib2017/repair_asym/timesofbinvars/'
# elif mode == 'improve':
#     directory = './result/miplib2017/improve/'
#
#
#
# for i in range(len(miplib2017_binary39)):
#
#     instance_name = miplib2017_binary39[i]
#     data = numpy.load(directory+instance_name+'.npz')
#     neigh_sizes = data['neigh_sizes']
#     t =data['t']
#     objs = data['objs']
#
#     if mode == 'repair-nviolations':
#         neigh_sizes = numpy.log10(neigh_sizes)
#         plt.clf()
#         fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
#
#         fig.suptitle("LB to repair")
#         fig.subplots_adjust(top=0.5)
#         ax[0].plot(neigh_sizes, objs)
#         ax[0].set_title(instance_name, loc='right')
#         ax[0].set_xlabel(r'$log(\alpha)$   '+'(Neighborhood size: '+ r'$K = \alpha \times N_{violations}$)')
#         ax[0].set_ylabel(r'$N_{violations}$')
#         ax[1].plot(neigh_sizes, t)
#         # ax[1].set_ylim([0, 31])
#         ax[1].set_ylabel("Solving time")
#         plt.show()
#
#         instance_name = miplib2017_binary39[i]
#         data = numpy.load('./result/miplib2017/repair_asym/timesofviolations/' + instance_name + '.npz')
#         neigh_sizes = data['neigh_sizes']
#         t = data['t']
#         objs = data['objs']
#
#         neigh_sizes = numpy.log10(neigh_sizes)
#         plt.clf()
#         fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
#
#         fig.suptitle("LB to repair (asymmetric)")
#         fig.subplots_adjust(top=0.5)
#         ax[0].plot(neigh_sizes, objs)
#         ax[0].set_title(instance_name, loc='right')
#         ax[0].set_xlabel(r'$log(\alpha)$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{violations}$)')
#         ax[0].set_ylabel(r'$N_{violations}$')
#         ax[1].plot(neigh_sizes, t)
#         # ax[1].set_ylim([0, 31])
#         ax[1].set_ylabel("Solving time")
#         plt.show()
#
#     elif mode == 'repair-nbinvars':
#         plt.clf()
#         fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
#
#         fig.suptitle("LB to repair")
#         fig.subplots_adjust(top=0.5)
#         ax[0].plot(neigh_sizes, objs)
#         ax[0].set_title(instance_name, loc='right')
#         ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
#         ax[0].set_ylabel(r'$N_{violations}$')
#         ax[1].plot(neigh_sizes, t)
#         # ax[1].set_ylim([0, 31])
#         ax[1].set_ylabel("Solving time")
#         plt.show()
#     elif mode == 'improve':
#         plt.clf()
#         fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
#         fig.suptitle("LB to improve")
#         fig.subplots_adjust(top=0.5)
#         ax[0].plot(neigh_sizes, objs)
#         ax[0].set_title(instance_name, loc='right')
#         ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
#         ax[0].set_ylabel("Objective")
#         ax[1].plot(neigh_sizes, t)
#         # ax[1].set_ylim([0,31])
#         ax[1].set_ylabel("Solving time")
#         plt.show()
#
#
#
#
