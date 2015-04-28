from pgmpy.models import JunctionTree


class MarginalTree(JunctionTree):

    def __init__(self):
        super().__init__()
        self.separators = {}
