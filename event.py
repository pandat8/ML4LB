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
