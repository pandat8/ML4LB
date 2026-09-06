import matplotlib.pyplot as plt
import numpy

instancetypes = ['setcovering', 'capacitedfacility', 'independentset', 'combinatorialauction']
modes = ['repair-slackvars', 'repair-supportbinvars', 'repair-binvars', 'improve-supportbinvars', 'improve-binvars']

instancetype = instancetypes[2]
mode = modes[4]
directory = './result/generated_instances/'+ instancetype +'/'+mode+'/'

# modes = ['repair-nviolations','repair-nbinvars','improve']
# mode = modes[2]
#
# if mode == 'repair-nviolations':
#     directory = './result/generated_instances/setcovering_asym/repair/timesofviolations/'
# elif mode == 'repair-nbinvars':
#     directory = './result/generated_instances/setcovering_asym/repair/timesofbinvars/'
# elif mode == 'improve':
#     directory = './result/generated_instances/setcovering_asym/improve/'


if mode == 'repair-nviolations':
    for i in range(100):
        if not i == 38:
            instance_name = instancetype + '-' + str(i)
            data = numpy.load(directory + instance_name + '.npz')
            neigh_sizes = data['neigh_sizes']
            t = data['t']
            objs = data['objs']

            # objs = objs / objs[0]
            print(objs[0])
            neigh_sizes = numpy.log10(neigh_sizes)
            if i==0:
                objs_all = objs
                t_all = t
            else:
                objs_all += objs
                t_all += t
    t_ave= t_all/99
    objs_ave = objs_all/99

    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))

    fig.suptitle("LB to repair")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$log(\alpha)$   '+'(Neighborhood size: '+ r'$K = \alpha \times N_{violations}$)')
    ax[0].set_ylabel(r'$N_{violations}$')
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0, 31])
    ax[1].set_ylabel("Solving time")
    plt.show()

elif mode == 'repair-nbinvars':
    for i in range(100):
        if not i == 38:
            instance_name =instancetype + '-' + str(i)
            data = numpy.load(directory + instance_name + '.npz')
            neigh_sizes = data['neigh_sizes']
            t = data['t']
            objs = data['objs']
            print(objs[0])
            # objs = objs / objs[0]

            if i == 0:
                objs_all = objs
                t_all = t
            else:
                objs_all += objs
                t_all += t
    t_ave = t_all / 99
    objs_ave = objs_all / 99
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))

    fig.suptitle("LB to repair")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
    ax[0].set_ylabel(r'$N_{violations}$')
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0, 31])
    ax[1].set_ylabel("Solving time")
    plt.show()

elif mode == 'improve':
    for i in range(100):
        # if not i == 38:
        instance_name = instancetype + '-' +  str(i)
        data = numpy.load(directory + instance_name + '.npz')
        neigh_sizes = data['neigh_sizes']
        t = data['t']
        objs = data['objs']
        # objs = objs / objs[0]

        if i == 0:
            objs_all = objs
            t_all = t
        else:
            objs_all += objs
            t_all += t
    t_ave = t_all / 100
    objs_ave = objs_all / 100
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
    fig.suptitle("LB to improve")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
    ax[0].set_ylabel("Objective")
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0,31])
    ax[1].set_ylabel("Solving time")
    plt.show()

elif mode == 'repair-slackvars':
    for i in range(100):
        if not i == 38:
            instance_name = instancetype + '-' +  str(i)
            data = numpy.load(directory + instance_name + '.npz')
            neigh_sizes = data['neigh_sizes']
            t = data['t']
            objs = data['objs']
            # objs = objs / objs[0]
            print(objs[0])

            if i == 0:
                objs_all = objs
                t_all = t
            else:
                objs_all += objs
                t_all += t
    t_ave = t_all / 99
    objs_ave = objs_all / 99
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))

    fig.suptitle("LB to repair (over slack variables)")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{violations}$)')
    ax[0].set_ylabel(r'$N_{violations}$')
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0, 31])
    ax[1].set_ylabel("Solving time")
    plt.show()

elif mode == 'repair-supportbinvars':
    for i in range(100):
        if not i == 38:
            instance_name = instancetype + '-' +  str(i)
            data = numpy.load(directory + instance_name + '.npz')
            neigh_sizes = data['neigh_sizes']
            t = data['t']
            objs = data['objs']
            # objs = objs / objs[0]
            print(objs[0])

            if i == 0:
                objs_all = objs
                t_all = t
            else:
                objs_all += objs
                t_all += t
    t_ave = t_all / 99
    objs_ave = objs_all / 99
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))

    fig.suptitle("LB to repair (over support of binary vars)")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{supportofbins}$)')
    ax[0].set_ylabel(r'$N_{violations}$')
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0, 31])
    ax[1].set_ylabel("Solving time")
    plt.show()

elif mode == 'repair-binvars':
    for i in range(100):
        if not i == 38:
            instance_name = instancetype + '-' +  str(i)
            data = numpy.load(directory + instance_name + '.npz')
            neigh_sizes = data['neigh_sizes']
            t = data['t']
            objs = data['objs']
            objs = objs / objs[0]

            if i == 0:
                objs_all = objs
                t_all = t
            else:
                objs_all += objs
                t_all += t
    t_ave = t_all / 99
    objs_ave = objs_all / 99
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))

    fig.suptitle("LB to repair (over binary variables)")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
    ax[0].set_ylabel(r'$N_{violations}$')
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0, 31])
    ax[1].set_ylabel("Solving time")
    plt.show()

elif mode =='improve-supportbinvars':
    for i in range(0, 100):
        # if not i == 38:
        instance_name = instancetype + '-' + str(i)
        print(instance_name)
        data = numpy.load(directory + instance_name + '.npz')
        neigh_sizes = data['neigh_sizes']
        t = data['t']
        objs = data['objs']
        # objs = objs / objs[0]

        if i == 0:
            objs_all = objs
            t_all = t
        else:
            objs_all += objs
            t_all += t
    t_ave = t_all / 100
    objs_ave = objs_all / 100
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
    fig.suptitle("LB to improve (over support of binary vars)")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{supportofbins}$)')
    ax[0].set_ylabel("Objective")
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0,31])
    ax[1].set_ylabel("Solving time")
    plt.show()

# elif mode =='improve-binvars':
#     for i in range(60, 70):
#         # if not i == 38:
#         instance_name = instancetype + '-' +  str(i)
#         print(instance_name)
#         data = numpy.load(directory + instance_name + '.npz')
#         neigh_sizes = data['neigh_sizes']
#         t = data['t']
#         objs = data['objs']
#         # objs = objs / objs[0]
#
#
#         plt.clf()
#         fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
#         fig.suptitle("LB to improve (over all bins)")
#         fig.subplots_adjust(top=0.5)
#         ax[0].plot(neigh_sizes, objs)
#         ax[0].set_title(instance_name, loc='right')
#         ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
#         ax[0].set_ylabel("Objective")
#         ax[1].plot(neigh_sizes, t)
#         # ax[1].set_ylim([0,31])
#         ax[1].set_ylabel("Solving time")
#         plt.show()

elif mode =='improve-binvars':
    for i in range(0, 100):
        # if not i == 38:
        instance_name = instancetype + '-' +  str(i)
        print(instance_name)
        data = numpy.load(directory + instance_name + '.npz')
        neigh_sizes = data['neigh_sizes']
        t = data['t']
        objs = data['objs']
        t = t/30
        objs = (objs - numpy.min(objs))
        objs = objs / numpy.max(objs)

        if i == 0:
            objs_all = objs
            t_all = t
        else:
            objs_all += objs
            t_all += t
    t_ave = t_all / 100
    objs_ave = objs_all / 100
    alpha = 1/3
    perf = alpha * t_ave + (1-alpha) * objs_ave

    print(neigh_sizes[numpy.where(perf == perf.min())])

    plt.clf()
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
    fig.suptitle("LB to improve (over all bins)")
    fig.subplots_adjust(top=0.5)
    ax[0].plot(neigh_sizes, objs_ave)
    ax[0].set_title(instance_name, loc='right')
    ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
    ax[0].set_ylabel("Objective")
    ax[1].plot(neigh_sizes, t_ave)
    # ax[1].set_ylim([0,31])
    ax[1].set_ylabel("Solving time")
    ax[2].plot(neigh_sizes, perf)
    ax[2].set_ylabel("Performance score")
    plt.show()
