import ecole
import pyscipopt


class SimpleConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):

    def reset_dynamics(self, model):
        # Share memory with Ecole model
        pyscipopt_model = model.as_pyscipopt()

        pyscipopt_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF) # disable heuristics
        pyscipopt_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF) # disable cuts

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().models.py(model)

class DynamicsEnablecuts(ecole.dynamics.ConfiguringDynamics):

    def reset_dynamics(self, model):
        # Share memory with Ecole model
        pyscipopt_model = model.as_pyscipopt()

        pyscipopt_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF) # disable heuristics

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().reset_dynamics(model)

class DynamicsEnableHeuristics(ecole.dynamics.ConfiguringDynamics):

    def reset_dynamics(self, model):
        # Share memory with Ecole model
        pyscipopt_model = model.as_pyscipopt()

        pyscipopt_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF) # disable cuts

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().reset_dynamics(model)

class SimpleConfiguring(ecole.environment.Environment):
    """
    only disable heuristics and pre-solving
    """
    __Dynamics__ = SimpleConfiguringDynamics

class SimpleConfiguringEnablecuts(ecole.environment.Environment):
    """
    disable heuristics, cuts and presolving
    """
    __Dynamics__ = DynamicsEnablecuts

class SimpleConfiguringEnableheuristics(ecole.environment.Environment):
    """
    disable heuristics, cuts and presolving
    """
    __Dynamics__ = DynamicsEnableHeuristics
