import math
from typing import List


class ScalarNodeCache:
    value = None
    local_partial_derivatives_wrt_children = None
    global_derivative_wrt_self = None


class ScalarNode:

    def __init__(self, children: List['ScalarNode']) -> None:
        # We need to "wire-up" all the children nodes. Each of them must know its parent
        # (which is this node), otherwise they would not be part of the computational
        # graph and backpropagation on them would not work.
        self._parents = []
        self._children = children
        for child in self._children:
            child._parents.append(self)

        # Empty the cache
        self._cache = ScalarNodeCache()

    def value(self) -> float:
        raise NotImplementedError()

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        raise NotImplementedError()

    def find_self_position_in_parents_children(self, parent: 'ScalarNode') -> int:
        for i, child in enumerate(parent._children):
            if self == child:
                return i

        raise Exception("Self found not in parent's children")

    def global_derivative_wrt_self(self) -> float:
        # Look up in the cache first
        if self._cache.global_derivative_wrt_self is not None:
            return self._cache.global_derivative_wrt_self

        if len(self._parents) == 0:
            # no parent, this must be the output node, and d out/d out = 1.0
            return 1.0
        else:
            result = 0.0
            # Generalized chain rule: For each parent, get its "global" derivative and multiply by its partial
            # derivative with respect to this node; sum the products up
            for p in self._parents:
                index_in_parents_children = self.find_self_position_in_parents_children(p)
                parent_to_self_derivative = p.local_partial_derivatives_wrt_children()[index_in_parents_children]
                parent_global_derivative = p.global_derivative_wrt_self()

                result += parent_to_self_derivative * parent_global_derivative

            # Save to the cache
            self._cache.global_derivative_wrt_self = result

            return result

    def reset_cache(self) -> None:
        self._cache = ScalarNodeCache()

    def update_parameters_by_gradient_step(self, learning_rate: float) -> None:
        # Update if trainable parameter; this is little hacky, we should implement this part
        # simply as a method in ParameterNode, but it's here for simplicity to keep the update
        # at one place
        if isinstance(self, ParameterNode):
            # --- EX3_TASK_5 ---
            gradient_of_node = self.global_derivative_wrt_self()
            # print("Grad of par:  {:10.4f}".format(gradient_of_node))

            # Perform step
            new_parameter_value = self.value() - learning_rate * gradient_of_node
            self.set_value(new_parameter_value)
            # --- EX3_TASK_5 ---

        # And call it recursively on all children
        for child in self._children:
            child.update_parameters_by_gradient_step(learning_rate)

    def clean_cache_recursively(self) -> None:
        # We need to do a certain operation with every node
        # --- EX3_TASK_5 ---
        self._cache = ScalarNodeCache()
        # --- EX3_TASK_5 ---

        # And call it recursively on all children
        for child in self._children:
            child.clean_cache_recursively()

    def __repr__(self):
        return '{0}_({1})'.format(self.__class__.__name__, hex(id(self)))


class ConstantNode(ScalarNode):

    def __init__(self, value: float) -> None:
        super().__init__([])
        self._value = value

    def value(self) -> float:
        return self._value


class SumNode(ScalarNode):

    def value(self) -> float:
        if self._cache.value is not None:
            return self._cache.value

        result = 0.0
        # Sum all arguments' values
        for child in self._children:
            result += child.value()

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        # Partial derivative wrt. each argument is 1.0, for example
        # y = w_1 + w_2 + w_3
        # dy/dw_1 = 1
        # dy/dw_2 = 1
        # dy/dw_3 = 1

        return [1.0] * len(self._children)


class ProductNode(ScalarNode):

    def value(self) -> float:
        if self._cache.value is not None:
            return self._cache.value

        result = 1.0
        # Multiply all arguments values
        for child in self._children:
            result *= child.value()

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        # Partial derivative wrt. each argument is a product of all other arguments, for example
        # y = w_1 * w_2 * w_3
        # dy/dw_1 = w_2 * w_3
        # dy/dw_2 = w_1 * w_3
        # dy/dw_3 = w_1 * w_2

        if self._cache.local_partial_derivatives_wrt_children is not None:
            return self._cache.local_partial_derivatives_wrt_children

        # zero-filled result
        result = [0.0] * len(self._children)

        # For each i-th argument, compute the product of all other arguments
        for i in range(len(self._children)):
            ith_result = 1.0
            for j in range(len(self._children)):
                if i != j:  # Skip the i-th argument in the product computation
                    j_value = self._children[j].value()
                    ith_result *= j_value
            result[i] = ith_result

        # Save to the cache
        self._cache.local_partial_derivatives_wrt_children = result

        return result


class ParameterNode(ConstantNode):

    def set_value(self, value: float) -> None:
        # --- EX3_TASK_0 ---
        # print("Param update: {:10.4f} -> {:10.4f}".format(self._value, value))
        self._value = value
        # --- EX3_TASK_0 ---


class LinearNode(ScalarNode):

    def __init__(self, arguments: List[ScalarNode], weights: List[ParameterNode], bias: ParameterNode) -> None:
        # This is an important but arbitrary design choice!
        # We pack arguments, weights, and bias in a single list of children
        super().__init__(arguments + weights + [bias])
        self._arguments = arguments
        self._weights = weights
        self._bias = bias

        # We must have the same number of weights as arguments
        assert len(weights) == len(arguments)

    def value(self) -> float:
        if self._cache.value is not None:
            return self._cache.value

        result = 0.0

        # --- EX3_TASK_1 ---
        for i, arg in enumerate(self._arguments):
            x_i = arg.value()
            w_i = self._weights[i].value()
            result += w_i * x_i

        # Add the bias
        result += self._bias.value()
        # --- EX3_TASK_1 ---

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        if self._cache.local_partial_derivatives_wrt_children is not None:
            return self._cache.local_partial_derivatives_wrt_children

        result = None
        # --- EX3_TASK_2 ---
        result = [a.value() for a in self._weights] + [w.value() for w in self._arguments] + [1]
        # --- EX3_TASK_2 ---

        # Save to the cache
        self._cache.local_partial_derivatives_wrt_children = result

        return result


class SigmoidNode(ScalarNode):

    def __init__(self, argument: ScalarNode) -> None:
        # Single-item list of children
        super().__init__([argument])

    def value(self) -> float:
        if self._cache.value is not None:
            return self._cache.value

        result = 0.0
        # --- EX3_TASK_3 ---
        result = 1 / (1 + math.exp(- self._children[0].value()))
        # --- EX3_TASK_3 ---

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        if self._cache.local_partial_derivatives_wrt_children is not None:
            return self._cache.local_partial_derivatives_wrt_children

        result = []
        # --- EX3_TASK_3 ---
        result = [self.value() * (1 - self.value())]
        # --- EX3_TASK_3 ---

        # Save to the cache
        self._cache.local_partial_derivatives_wrt_children = result

        return result


class CrossEntropyLoss(ScalarNode):

    def __init__(self, y_hat: ScalarNode, gold_label: ConstantNode) -> None:
        # Single-item list of children
        super().__init__([y_hat, gold_label])

        self._gold_label = gold_label

    def value(self) -> float:
        if self._cache.value is not None:
            return self._cache.value

        result = 0.0
        # --- EX3_TASK_4 ---
        y_val = self._gold_label.value()
        y_hat_val = self._children[0].value()

        # Corner case
        if y_val == y_hat_val:  # We discussed this in the exercise class and in reality this should never happen
            result = 0
        else:
            result = - y_val * math.log2(y_hat_val) - (1 - y_val) * math.log2(1 - y_hat_val)
        # --- EX3_TASK_4 ---

        # Save to the cache
        self._cache.value = result

        return result

    def local_partial_derivatives_wrt_children(self) -> List[float]:
        if self._cache.local_partial_derivatives_wrt_children is not None:
            return self._cache.local_partial_derivatives_wrt_children

        result = []
        # --- EX3_TASK_4 ---
        y_val = self._gold_label.value()
        y_hat_val = self._children[0].value()

        result = [- (y_val - y_hat_val) / (y_hat_val * (1 - y_hat_val))]
        # --- EX3_TASK_4 ---

        # Save to the cache
        self._cache.local_partial_derivatives_wrt_children = result

        return result


class InputNode(ConstantNode):

    def __init__(self):
        # Zero valued by default
        super().__init__(0)

    def set_value(self, value: float) -> None:
        self._value = value
