import numpy as np


### This won't do anything other than allow us to check
### if in object is a Graph node or not
class Node:
    def __init__(self):
        pass

    ### Placeholders ###


class Placeholder(Node):
    """An placeholder node in the computational graph. This holds
    a node, and awaits further input at computation time.
    Args: 
        name: defaults to "Plc/"+count
        dtype: the type that the node holds, float, int, etc.
    """
    count = 0

    def __init__(self, name, dtype=float):
        _g.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count += 1

    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"


### Constants ###
class Constant(Node):
    """A constant node in the computational graph.
    Args: 
        name: defaults to "const/"+count
        value: a property protected value that prevents user 
               from reassigning value
    """
    count = 0

    def __init__(self, value, name=None):
        _g.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1

    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constant")
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count += 1

    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"


### Variables ###
class Variable(Node):
    """An variable node in the computational graph. Variables are
    automatically tracked during graph computation.
    Args: 
        name: defaults to "var/"+count
        value: a mutable value
    """
    count = 0

    def __init__(self, value, name=None):
        _g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"


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


############################################
##### Fundamental Operator Definitions #####
############################################
class add(Operator):
    count = 0
    """Binary addition operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'add/{add.count}' if name is None else name
        add.count += 1

    def forward(self, a, b):
        return a + b

    def backward(self, a, b, dout):
        return dout, dout


class multiply(Operator):
    count = 0
    """Binary multiplication operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'mul/{multiply.count}' if name is None else name
        multiply.count += 1

    def forward(self, a, b):
        return a * b

    def backward(self, a, b, dout):
        return dout * b, dout * a


class divide(Operator):
    count = 0
    """Binary division operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'div/{divide.count}' if name is None else name
        divide.count += 1

    def forward(self, a, b):
        return a / b

    def backward(self, a, b, dout):
        return dout / b, dout * a / np.power(b, 2)


class power(Operator):
    count = 0
    """Binary exponentiation operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'pow/{power.count}' if name is None else name
        power.count += 1

    def forward(self, a, b):
        return np.power(a, b)

    def backward(self, a, b, dout):
        return dout * b * np.power(a, (b - 1)), dout * np.log(a) * np.power(a, b)


class matmul(Operator):
    count = 0
    """Binary multiplication operation."""

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f'matmul/{matmul.count}' if name is None else name
        matmul.count += 1

    def forward(self, a, b):
        return a @ b

    def backward(self, a, b, dout):
        return dout @ b.T, a.T @ dout


############################################
#####       Operator overloading       #####
############################################
def node_wrapper(func, self, other):
    """ Check to make sure that the two things we're comparing are
    actually graph nodes. Also, if we use a constant, automatically
    make a Constant node for it"""
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError("Incompatible types.")


Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__div__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1))
Node.__pow__ = lambda self, other: node_wrapper(power, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)


class Graph():
    """ Computational graph class.
    Initilizes a global variable _g that describes the graph.
    Each graph consists of a set of
        1. operators
        2. variables
        3. constants
        4. placeholders
    """

    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        global _g
        _g = self

    def reset_counts(self, root):
        if hasattr(root, 'count'):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        try:
            del _g
        except:
            pass
        self.reset_counts(Node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()



def topological_sort(head_node, graph):
    """Performs topological sort of all nodes prior to and
    including the head_node.
    Args:
        graph: the computational graph. This is the global value by default
        head_node: last node in the forward pass. The "result" of the graph.
    Returns:
        a sorted array of graph nodes.
    """
    vis = set()
    ordering = []

    def _dfs(node):
        if node not in vis:  # check if the current node has been visited (traversal)
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
                ordering.append(node)

    if head_node is None:
        for node in graph.operators:
            _dfs(node)
    else:
        _dfs(head_node)

    return ordering


def iterative_search(equation_head):
    # Iteratively search over the computation graph via Depth-First Search
    visited = set()
    stack = list()
    fully_scraped = False
    current_position = equation_head
    while fully_scraped is not True:
        if isinstance(current_position, Operator):
            # We need to see if the current operator has any operator children that have not been traversed
            has_child_ops = any(isinstance(child, Operator) for child in current_position.inputs)
            if has_child_ops and current_position in visited:
                stack.append(sub_node)
                current_position = list(visited)[list(visited).index(current_position)-1]
            elif has_child_ops:
                for sub_node_idx, sub_node in enumerate(current_position.inputs):
                    if sub_node not in visited:
                        if isinstance(sub_node, Operator):
                            visited.add(current_position)
                            current_position = sub_node
                            # current_position.inputs.pop(sub_node_idx)
                            break
            else:
                if current_position in stack:
                    fully_scraped = True
                else:
                    stack.append(current_position)
                    visited.add(current_position)
                    current_position = list(visited)[list(visited).index(current_position)-1]
    return stack





##############################
#####    Forward pass    #####
##############################
def forward_pass(order, feed_dict={}):
    """ Performs the forward pass, returning the output of the graph.
    Args:
        order: a topologically sorted array of nodes
        feed_dict: a dictionary values for placeholders.
    Returns:
        1. the final result of the forward pass.
        2. directly edits the graph to fill in its current values.
    """
    for node in order:

        if isinstance(node, Placeholder):
            node.value = feed_dict[node.name]

        elif isinstance(node, Operator):
            node.value = node.forward(*[prev_node.value for prev_node in node.inputs])

    return order[-1].value


##############################
#####    Backward pass   #####
##############################
def backward_pass(order):
    """ Perform the backward pass to retrieve gradients.
    Args:
        order: a topologically sorted array of graph nodes.
               by default, this assigns the graident of the final node to 1
    Returns:
        gradients of nodes as listed in same order as input argument
    """
    vis = set()
    order[-1].gradient = 1
    for node in reversed(order):
        inputs = node.inputs
        grads = node.backward(*[x.value for x in inputs], dout=node.gradient)
        for inp, grad in zip(inputs, grads):
            if inp not in vis:
                inp.gradient = grad
            else:
                inp.gradient += grad
            vis.add(inp)
    return [node.gradient for node in order]


val1, val2, val3 = 0.8, 0.4, 1.3

with Graph() as g:
  x = Variable(val1, name='x')
  y = Variable(val2, name='y')
  c = Variable(val3, name='c')
  z = (x*y+c)*c
  rec_order = topological_sort(z, _g)
  it_order = iterative_search(z)
  res = forward_pass(order)
  grads = backward_pass(order)
  print("Node ordering:")
  for node in order:
      print(node)
  print('-'*10)
  print(f"Forward pass expected: {(val1*val2+val3)*val3}")
  print(f"Forward pass computed: {res}")
