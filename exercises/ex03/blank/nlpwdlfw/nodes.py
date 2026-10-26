from typing import List


class ScalarNodeCache:
    value = None
    local_partial_derivatives_wrt_arguments = None
    global_derivative_wrt_self = None


class ScalarNode:

    def __init__(self, arguments: List['ScalarNode']) -> None:
        self._parents = []
        self._arguments = arguments
        for arg in self._arguments:
            arg._parents.append(self)
        # --- TASK_6 ---
        self._cache = None
        # --- TASK_6 ---

    def value(self) -> float:
        raise NotImplementedError()

    def local_partial_derivatives_wrt_arguments(self) -> List[float]:
        raise NotImplementedError()

    def find_self_position_in_parents_arguments(self, parent: 'ScalarNode') -> int:
        for i, arg in enumerate(parent._arguments):
            if self == arg:
                return i
        raise Exception("Self found not in parent's arguments")

    def global_derivative_wrt_self(self) -> float:
        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        if len(self._parents) == 0:
            # no parent, this must be the output node, and d out/d out = 1.0
            return 1.0
        else:
            result = 0.0
            # --- TASK_5 ---
            pass
            # --- TASK_5 ---

            # --- TASK_6 ---
            pass
            # --- TASK_6 ---

            return result


class ConstantNode(ScalarNode):

    def __init__(self, value: float) -> None:
        super().__init__([])
        self._value = value

    def value(self) -> float:
        return self._value


class SumNode(ScalarNode):

    def value(self) -> float:
        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        result = 0.0
        # sum all arguments values

        # --- TASK_2 ---
        pass
        # --- TASK_2 ---

        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        return result

    def local_partial_derivatives_wrt_arguments(self) -> List[float]:
        # Partial derivative wrt. each argument is 1.0, for example
        # y = w_1 + w_2 + w_3
        # dy/dw_1 = 1
        # dy/dw_2 = 1
        # dy/dw_3 = 1

        # --- TASK_2 ---
        pass
        # --- TASK_2 ---


class ProductNode(ScalarNode):

    def value(self) -> float:
        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        result = 1.0
        # multiply all arguments values

        # --- TASK_3 ---
        pass
        # --- TASK_3 ---

        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        return result

    def local_partial_derivatives_wrt_arguments(self) -> List[float]:
        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        # Partial derivative wrt. each argument is a product of all other arguments, for example
        # y = w_1 * w_2 * w_3
        # dy/dw_1 = w_2 * w_3
        # dy/dw_2 = w_1 * w_3
        # dy/dw_3 = w_1 * w_2

        # zero-filled result
        result = [0.0] * len(self._arguments)

        # --- TASK_3 ---
        pass
        # --- TASK_3 ---

        # --- TASK_6 ---
        pass
        # --- TASK_6 ---

        return result

