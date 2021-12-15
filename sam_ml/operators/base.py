from sam_ml.node import Node


### Operators ###
class Operator(Node):
    """An operator node in the computational graph.
    Args:
        name: defaults to "operator name/"+count
    """

    def __init__(self, name='Operator'):
        _g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name

    def __repr__(self):
        return f"Operator: name:{self.name}"