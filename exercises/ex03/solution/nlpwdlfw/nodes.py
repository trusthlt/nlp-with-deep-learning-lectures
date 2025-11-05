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
        self._cache = ScalarNodeCache()
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
        if self._cache.global_derivative_wrt_self is not None:
            return self._cache.global_derivative_wrt_self
        # --- TASK_6 ---

        if len(self._parents) == 0:
            # no parent, this must be the output node, and d out/d out = 1.0
            return 1.0
        else:
            result = 0.0
            # --- TASK_5 ---
            # multiply and add (generalized chain rule)
            for p in self._parents:
                index_in_parents_arguments = self.find_self_position_in_parents_arguments(p)
                parent_to_self_derivative = p.local_partial_derivatives_wrt_arguments()[index_in_parents_arguments]
                parent_global_derivative = p.global_derivative_wrt_self()

                result += parent_to_self_derivative * parent_global_derivative
            # --- TASK_5 ---

            # --- TASK_6 ---
            self._cache.global_derivative_wrt_self = result
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
        if self._cache.value is not None:
            return self._cache.value
        # --- TASK_6 ---

        result = 0.0
        # sum all arguments values

        # --- TASK_2 ---
        for arg in self._arguments:
            result += arg.value()
        # --- TASK_2 ---

        # --- TASK_6 ---
        self._cache.value = result
        # --- TASK_6 ---

        return result

    def local_partial_derivatives_wrt_arguments(self) -> List[float]:
        # Partial derivative wrt. each argument is 1.0, for example
        # y = w_1 + w_2 + w_3
        # dy/dw_1 = 1
        # dy/dw_2 = 1
        # dy/dw_3 = 1

        # --- TASK_2 ---
        return [1.0] * len(self._arguments)
        # --- TASK_2 ---


class ProductNode(ScalarNode):

    def value(self) -> float:
        # --- TASK_6 ---
        if self._cache.value is not None:
            return self._cache.value
        # --- TASK_6 ---

        result = 1.0
        # multiply all arguments values

        # --- TASK_3 ---
        for arg in self._arguments:
            result *= arg.value()
        # --- TASK_3 ---

        # --- TASK_6 ---
        self._cache.value = result
        # --- TASK_6 ---

        return result

    def local_partial_derivatives_wrt_arguments(self) -> List[float]:
        # --- TASK_6 ---
        if self._cache.local_partial_derivatives_wrt_arguments is not None:
            return self._cache.local_partial_derivatives_wrt_arguments
        # --- TASK_6 ---

        # Partial derivative wrt. each argument is a product of all other arguments, for example
        # y = w_1 * w_2 * w_3
        # dy/dw_1 = w_2 * w_3
        # dy/dw_2 = w_1 * w_3
        # dy/dw_3 = w_1 * w_2

        # zero-filled result
        result = [0.0] * len(self._arguments)

        # --- TASK_3 ---
        for i in range(len(self._arguments)):
            ith_result = 1.0
            for j in range(len(self._arguments)):
                if i != j:
                    j_value = self._arguments[j].value()
                    ith_result *= j_value
            result[i] = ith_result
        # --- TASK_3 ---

        # --- TASK_6 ---
        self._cache.local_partial_derivatives_wrt_arguments = result
        # --- TASK_6 ---

        return result

