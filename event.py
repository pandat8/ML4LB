from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE

class PrimalBoundChangeEventHandler(Eventhdlr):

    def __init__(self):
        super().__init__()
        self.primal_bounds = []
        self.primal_times = []

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        # update the integral
        self.primal_bounds.append(self.model.getPrimalbound())
        self.primal_times.append(self.model.getSolvingTime())


class StopWhenFirstLPSolvedEventHandler(Eventhdlr):
    def __init__(self):
        super().__init__()
        self.lp_status = 0
        self.n_lps = 0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.FIRSTLPSOLVED, self)

    def eventexec(self, event):
        status = self.model.getStatus()
        self.lp_status = self.model.getLPSolstat()
        firstlp_time_2 = self.model.getFirstLpTime()
        stage = self.model.getStage()
        n_sols = self.model.getNSols()
        time = self.model.getSolvingTime()
        self.n_lps = self.model.getNLPs()

        print('after re-solving LP:')
        print('* solving time: ', time)
        print('*  first LP time: ', firstlp_time_2)
        print("* Model status: %s" % status)
        print("* Solve stage: %s" % stage)
        print("* LP status: %s" % self.lp_status)
        print('* number of LP sol : ', self.n_lps)
        print('* number of sol : ', n_sols)

        if (self.lp_status == 1 ) and self.n_lps == 1:
            print('Event: Optimal LP is found after the first LP solved! LP status =' + str(self.lp_status) + '.')
            # self.sol_lp = self.model.createLPSol()  # get current LP solution
        else:
            print('Error: no optimal LP is found after the first LP solved! LP status =' + str(self.lp_status) + '.')

        self.model.interruptSolve()

