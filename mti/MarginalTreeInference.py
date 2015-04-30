from pgmpy.models import JunctionTree
from pgmpy.models import BayesianModel
from pgmpy.factors.Factor import factor_product
from pgmpy.inference import Inference
from pgmpy.inference import EliminationOrdering
from mti.MarginalTree import MarginalTree


class MarginalTreeInference(Inference):

    def __init__(self, model):
        super().__init__(model)

        if not isinstance(model, JunctionTree):
            self.junction_tree = model.to_junction_tree(False)
        else:
            self.model = model
            self.junction_tree = model
        self.clique_beliefs = {}
        self.sepset_beliefs = {}
        self.root = ()
        self.evidence = {}
        self.observed = []
        # All marginal trees are saved
        self.marginal_trees = []

    def _build(self, variables, evidence=None, elimination_order=None):
        if not isinstance(variables, list):
            variables = [variables]
        # Removing barren and independent variables generate sub-models
        # (a modified version of the model).
        # Then, a copy is used to do not disturb the original model.
        model_copy = self.model.copy()
        factors_copy = self.factors.copy()

        # Load all factors used in this session of Variable Elimination
        working_factors = {node: {factor for factor in factors_copy[node]}
                           for node in factors_copy}

        # Dealing with evidence. Reducing factors over it before VE is run.
        if evidence:
            for evidence_var in evidence:
                for factor in working_factors[evidence_var]:
                    factor_reduced = factor.reduce(
                        '{evidence_var}_{state}'
                        .format(evidence_var=evidence_var,
                                state=evidence[evidence_var]),
                        inplace=False)
                    for var in factor_reduced.scope():
                        working_factors[var].remove(factor)
                        working_factors[var].add(factor_reduced)
                del working_factors[evidence_var]

        if not elimination_order:
            # If is BayesianModel, find a good elimination ordering
            # using Weighted-Min-Fill heuristic.
            if isinstance(model_copy, BayesianModel):
                elim_ord = EliminationOrdering(model_copy)
                elimination_order = elim_ord.find_elimination_ordering(
                    list(set(model_copy.nodes()) -
                         set(variables) -
                         set(evidence.keys()
                             if evidence else [])),
                    elim_ord.weighted_min_fill)
            else:
                elimination_order = list(set(self.variables) -
                                         set(variables) -
                                         set(evidence.keys()
                                             if evidence else []))

        elif any(var in elimination_order for var in
                 set(variables).union(
                     set(evidence.keys() if evidence else []))):
            raise ValueError("Elimination order contains variables"
                             " which are in variables or evidence args")

        # Perform elimination ordering while constructing new Marginal Tree
        marginal_tree = MarginalTree()
        eliminated_variables = set()
        messages = []
        # Variables to keep the last "phi" message and last created "node"
        phi = None
        node = None
        for var in elimination_order:
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [factor for factor in working_factors[var]
                       if not set(factor.variables).intersection(
                eliminated_variables)]
            phi = factor_product(*factors)
            phi = phi.marginalize(var, inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add(phi)
            eliminated_variables.add(var)
            # Save new message
            messages.append(phi)
            # Build a Marginal Tree node
            node = set()
            for f in factors:
                node = node.union(f.scope())
            node = tuple(node)
            marginal_tree.add_node(node)
            marginal_tree.add_factors_to_node(factors, node)
            # Connect nodes
            messages_intersection = set(factors).intersection(set(messages))
            messages_used = []
            for m in list(messages_intersection):
                for edge in marginal_tree.separators.copy():
                    if m in marginal_tree.separators[edge]:
                        marginal_tree.add_edge(edge[0], node)
                        new_edge = (edge[0], node)
                        marginal_tree.add_messages_to_separator(m, new_edge)
                        del marginal_tree.separators[edge]
                        messages_used.append(m)
            # If message wasn't used to create the new message,
            # point it to the "empty node".
            if phi not in messages_used:
                marginal_tree.add_messages_to_separator(phi, (node,))
        query_node = tuple(phi.variables)
        marginal_tree.add_node(query_node)
        remaining_factors = []
        for var in working_factors:
            remaining_factors.extend([factor for factor in working_factors[var]
                                      if (not set(
                                          factor.variables
                                          ).intersection(
                                          eliminated_variables)
                                          ) and var in factor.left_hand_side])
        # Adding the query node and the last message to it.
        marginal_tree.add_edge(node, query_node)
        marginal_tree.separators[(node, query_node)] = [phi]
        marginal_tree.add_factors_to_node(remaining_factors, query_node)
        # Define the root node as the query node.
        marginal_tree.root = query_node
        return marginal_tree

    def propagate(self, query, evidence=None, elimination_order=None):
        new_mt = self._build(query, evidence)
        self.marginal_trees.append(new_mt)

    def query(self, query, evidence=None, elimination_order=None):
        variables = query
        if evidence:
            variables.extend(list(evidence))
