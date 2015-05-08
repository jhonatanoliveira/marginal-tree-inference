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
            self.junction_tree = model
        self.model = model
        self.clique_beliefs = {}
        self.sepset_beliefs = {}
        self.root = ()
        self.observed = []
        # All marginal trees are saved
        self.marginal_trees = []

    def find_reusable(self, query, evidence=None):
        if not isinstance(query, list):
            query = [query]
        reusable = []
        for saved_mt in self.marginal_trees:
            if set(saved_mt.evidence).issubset(set(evidence)) or set(
                    saved_mt.evidence) == set(evidence):
                if all(saved_mt.evidence[k] == evidence[k]
                        for k in saved_mt.evidence):
                    reusable.append(saved_mt)
        return reusable

    def propagate(self, query, evidence=None, elimination_order=None):
        new_mt = self._build(query, evidence)
        self.marginal_trees.append(new_mt)

    def query(self, query, evidence=None, elimination_order=None):
        if not isinstance(query, list):
            query = [query]
        # See if it is possible to reuse saved MTs
        marginal_tree = None
        reuse_mts = self.find_reusable(query, evidence)
        if len(reuse_mts) > 0:
            # TODO: choose MT by the size of it
            # Choose one MT and rebuild it for the new query and evidence
            marginal_tree = reuse_mts[0]
            ### DEBUG
            # print(">>> Nodes before evidence")
            # print(marginal_tree.nodes())
            ### --- DEBUG
            # Reduce new evidences
            new_evidence = {k: evidence[k]
                            for k in evidence
                            if k not in marginal_tree.evidence}
            marginal_tree.set_evidence(new_evidence)
            ### DEBUg
            # print(">>> Nodes after evidence")
            # print(marginal_tree.nodes())
            ### --- DEBUG
            # Rebuild MT to answer new query
            marginal_tree = marginal_tree.rebuild(query + list(evidence))
            # Perform one way propagation
            self.partial_one_way_propagation(marginal_tree)
        else:
            marginal_tree = self._build(query, evidence, elimination_order)
            self.marginal_trees.append(marginal_tree)
        # Save the new MT
        self.marginal_trees.append(marginal_tree)
        ### DEBUG
        # marginal_tree.draw()
        ### --- DEBUG
        # Answer the query
        node = marginal_tree.root
        # Define the variables to marginalize
        marginalize = set(node) - set(query)
        # Collect factors for the node
        neighbors = marginal_tree.neighbors(node)
        factors = []
        # Collect incoming messages
        for neighbor in neighbors:
            # separator_neighbor = frozenset(node).intersection(
            #                       frozenset(neighbor))
            if (neighbor, node) in marginal_tree.separators:
                factors.extend(marginal_tree.separators[(neighbor, node)])
        ### DEBUG
        # print(">>>>>--------------")
        # print(">>> Root: %s" % marginal_tree.root.__str__())
        # print("incoming...")
        # for f in factors:
        #     print(f)
        ### --- DEBUG
        # Collect assigned Factors
        factors.extend(marginal_tree.factor_node_assignment[node])
        ### DEBUG
        # print("assigned...")
        # for f in marginal_tree.factor_node_assignment[node]:
        #     print(f)
        ### --- DEBUG
        # Sum out variables from factors
        result = Inference.sum_out(marginalize, factors)
        ### DEBUG
        # print(">>> After SUM OUT")
        # for f in result:
        #     print(f)
        ### --- DEBUG
        # Multiply all remaining CPDs
        result = factor_product(*result)
        ### DEBUG
        # print(">>> After PRODUCT")
        # print(result)
        ### --- DEBUG
        # Normalize
        result.normalize()
        ### DEBUG
        # print(">>> After Normalize")
        # print(result)
        # marginal_tree.draw()
        ### --- DEBUG
        return result

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

        # Reduce working factors
        if evidence:
            for evidence_var in evidence:
                for factor in working_factors[evidence_var].copy():
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
            ### DEBUG
            # print(">>> *** Eliminating %s ***" % var)
            ### --- DEBUG
            factors = [factor for factor in working_factors[var]
                       if not set(factor.variables).intersection(
                eliminated_variables)]
            ### DEBUG
            # print(">>> Factors involved")
            # for f in factors:
            #     print(f)
            ### --- DEBUG
            phi = factor_product(*factors)
            ### DEBUG
            # print(">>> Product")
            # print(phi)
            ### --- DEBUG
            phi = phi.marginalize(var, inplace=False)
            ### DEBUG
            # print(">>> Marginalize")
            # print(phi)
            ### --- DEBUG
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
            messages_intersection = set(factors).intersection(set(messages))
            marginal_tree.add_factors_to_node(
                list(set(factors) - messages_intersection), node)
            # Connect nodes, if past messages are used
            messages_used = []
            for m in list(messages_intersection):
                for separator in marginal_tree.separators.copy():
                    if m in marginal_tree.separators[separator]:
                        marginal_tree.add_edge(separator[0], node)
                        new_separator = (separator[0], node)
                        marginal_tree.add_messages_to_separator(
                            m, new_separator)
                        del marginal_tree.separators[separator]
                        messages_used.append(m)
            ### DEBUG
            # print(">>> Messages used")
            # print(marginal_tree.separators[separator])
            ### --- DEBUG
            # If message wasn't used to create the new message,
            # point it to the "empty node".
            if phi not in messages_used:
                marginal_tree.add_messages_to_separator(phi, (node,))
        ### DEBUG
        # print("===> Remaining Factors")
        # for var in working_factors:
        #     print("===> var %s" % var)
        #     for f in working_factors[var]:
        #         print(f)
        ### --- DEBUG
        # Create the query node (where the query is answered)
        query_node = tuple(phi.variables)
        marginal_tree.add_node(query_node)

        ### DEBUG
        # print(">>> All Remaining Factors")
        # for var in working_factors:
        #     print(">>> All Remaining for var %s" % var)
        #     for f in working_factors[var]:
        #         if not set(f.variables).intersection(
        #                eliminated_variables):
        #             if f in messages:
        #                 print("---> A message.")
        #             else:
        #                 print("---> An original factor.")
        #             print(f)
        ### --- DEBUG

        # Add remaining original factors to the query node
        remaining_assignment_factors = []
        remaining_message_factors = []
        for var in working_factors:
            for factor in working_factors[var]:
                if not set(factor.variables).intersection(
                           eliminated_variables):
                    if factor in messages:
                        remaining_message_factors.append(factor)
                    else:
                        remaining_assignment_factors.append(factor)
        # ### DEBUG
        # print("===> Collected ASSIGMENT remaining Factors")
        # for f in remaining_assignment_factors:
        #     print(f)
        # print("===> Collected MESSAGES remaining Factors")
        # for f in remaining_message_factors:
        #     print(f)
        ### --- DEBUG
        # Redirect remaining message factors to query node
        for message in remaining_message_factors:
            for separator in marginal_tree.separators.copy():
                if message in marginal_tree.separators[separator]:
                    marginal_tree.add_edge(separator[0], query_node)
                    new_separator = (separator[0], query_node)
                    marginal_tree.add_messages_to_separator(
                        message, new_separator)
                    del marginal_tree.separators[separator]
        # TO REMOVE
        # marginal_tree.add_edge(node, query_node)
        # marginal_tree.add_messages_to_separator(
        #     phi, (node, query_node))
        # Add remaining original factors to query node
        marginal_tree.add_factors_to_node(
            remaining_assignment_factors, query_node)
        # Define the root node as the query node.
        marginal_tree.root = query_node
        # Update evidence variables of the Marginal tree
        for k in evidence:
            marginal_tree.evidence[k] = evidence[k]
        marginal_tree.observed.extend(list(evidence.keys()))
        return marginal_tree

    def partial_one_way_propagation(self, marginal_tree):
        propagation = marginal_tree.propagation_to_node(marginal_tree.root)
        # Check if the message was already created in each separator
        for separator in propagation:
            separators_copy = marginal_tree.separators.copy()
            if (separator not in separators_copy):
                self._absorption(marginal_tree, separator[1], separator[0])
        return marginal_tree

    def _find_relevant_potentials(self, factors, separator, marginal_tree):

        def _add_dconnected_factors(factors, R_s):
            for factor in factors:
                reachable = []
                for var in separator:
                    reachable.extend(self.model.active_trail_nodes(
                        var, list(marginal_tree.observed)))
                if len(set(factor.scope()).intersection(
                   set(reachable))) != 0:
                    R_s.append(factor)

        def _remove_unity_factors(separator, R_s):

            def _var_appears_only_once(var, factors):
                appears = False
                for f in factors:
                    if var in f.scope():
                        if appears:
                            return False
                        appears = True
                return True

            # Recursively remove barren tables
            while True:
                set_before_changes = R_s.copy()
                for factor in R_s:
                    for var in factor.left_hand_side:
                        if _var_appears_only_once(var, R_s
                                                  ) and (var not in separator):
                            R_s.remove(factor)
                            break
                        else:
                            break
                if len(set_before_changes) == len(R_s):
                    break

        if isinstance(separator, str):
            separator = [separator]
        R_s = []
        # Find relevant potentials by keeping the oned
        _add_dconnected_factors(factors, R_s)
        _remove_unity_factors(separator, R_s)
        return R_s

    def _absorption(self, marginal_tree, c_i, c_j):
        """
        Send a message from c_j to c_i during propagation.
        """
        # Union of all factors in c_j and its separators' factors
        neighbors = marginal_tree.neighbors(c_j)
        if c_i in neighbors:
            neighbors.remove(c_i)
        R_s = []
        for neighbor in neighbors:
            if (neighbor, c_j) in marginal_tree.separators:
                R_s.extend(marginal_tree.separators[(neighbor, c_j)])
        R_s.extend(marginal_tree.factor_node_assignment[c_j])
        separator = frozenset(c_i).intersection(frozenset(c_j))
        # Variables to marginalize from R_s
        marginalize = list(set(c_j) - set(separator))
        R_s = self._find_relevant_potentials(R_s, separator, marginal_tree)
        R_s = Inference.sum_out(marginalize, R_s)
        # Associate the messages with the separator of c_i and c_j,
        # in the right direction (from c_j to c_i)
        marginal_tree.separators[(c_j, c_i)] = R_s
