from pgmpy.models import JunctionTree
import networkx as nx


class MarginalTree(JunctionTree):

    def __init__(self):
        super().__init__()
        self.separators = {}
        self.root = None

    def copy(self):
        mt = MarginalTree()
        mt.add_nodes_from(self.nodes())
        mt.add_edges_from(self.edges())
        mt.factors = self.factors.copy()
        mt.factor_node_assignment = self.factor_node_assignment.copy()
        mt.separators = self.separators.copy()
        return mt

    def add_factors_to_node(self, factors, node):
        if not isinstance(factors, list):
            factors = [factors]
        for factor in factors:
            self.factors.append(factor)
            if node in self.factor_node_assignment:
                self.factor_node_assignment[node].append(factor)
            else:
                self.factor_node_assignment[node] = [factor]

    def add_messages_to_separator(self, messages, separator):
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            if separator in self.separators:
                self.separators[separator].append(message)
            else:
                self.separators[separator] = [message]

    def draw(self, node_size=5000):
        # Check library
        import matplotlib.pyplot as plt
        pos = None
        try:
            pos = nx.graphviz_layout(self)
        except:
            pos = nx.circular_layout(self)
        # Draw nodes
        nx.draw_networkx_nodes(G=self, pos=pos, node_size=node_size,
                               node_color="white")
        nodes_labels = {}
        for n in self.nodes():
            nodes_labels[n] = ", ".join(n)
        # Draw labels for nodes
        nx.draw_networkx_labels(self, pos=pos,
                                labels=nodes_labels, font_weight='bold',
                                label_pos=0.3, font_size=20)
        # Draw edges
        nx.draw_networkx_edges(G=self, pos=pos, width=2.0)

        # Draw labels for edges
        edge_labels = {}
        for e in self.separators:
            if len(e) > 1:
                factors = []
                for f in self.separators[e]:
                    factors.append(_get_factor_structure(f))
                edge_labels[e] = ", ".join(factors)
        nx.draw_networkx_edge_labels(G=self, pos=pos,
                                     edge_labels=edge_labels,
                                     label_pos=0.5,
                                     font_size=15)
        # Put text with CPTs under nodes
        for n in self.nodes():
            all_factors = []
            for f in self.factor_node_assignment[n]:
                all_factors.append(_get_factor_structure(f))
            plt.text(pos[n][0]-0.07, pos[n][1]+0.17, ", ".join(all_factors))
        # Shows the figure
        plt.axis('off')
        plt.show()

    def selective_reduction(self, marked_variables):
        if not isinstance(marked_variables, list):
            marked_variables = [marked_variables]
        nodes = self.nodes()
        original_nodes = self.nodes()
        old_nodes = self.nodes()
        while True:
            # Delete variables not in *marked_variables* that only
            # appears in only one node.
            checked_vars = []
            appears_more = []
            for node in nodes:
                for var in node:
                    if var not in marked_variables:
                        if var not in checked_vars:
                            checked_vars.append(var)
                        else:
                            appears_more.append(var)
            appears_once = list(set(checked_vars)-set(appears_more))
            # Delete var from node
            for var in appears_once:
                node = list(filter(lambda x: var in x, nodes))
                node = node[0]
                # update the node
                mod_node = list(node)
                mod_node.remove(var)
                new_node = tuple(mod_node)
                nodes[nodes.index(node)] = new_node
            # Remove nodes that are subsets of other ones.
            for idx, n1 in enumerate(nodes.copy()):
                for n2 in nodes.copy():
                    if n1 != n2:
                        if set(n1).issubset(set(n2)):
                            nodes[idx] = ()
            # Check if continue the loop
            if set(nodes) == set(old_nodes):
                break
            else:
                old_nodes = nodes.copy()
        # Return the original node for the node outputted
        # from SRA
        idx_original_node = next(i
                                 for i, j in enumerate(nodes)
                                 if len(set(j)) != 0)
        # TODO: return more than one node
        return original_nodes[idx_original_node]

    def rebuild(self, variables):
        new_mt = self.copy()
        node = new_mt.selective_reduction(variables)
        # TODO: make the lump of returned nodes
        new_mt.root = node
        return new_mt


def _get_factor_structure(factor):
    structure = "p("
    structure += ", ".join(factor.left_hand_side)
    structure += "|"
    structure += ",".join(factor.right_hand_side)
    structure += ")"
    return structure
