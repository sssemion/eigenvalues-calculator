import pytest
import numpy as np

from eigenvalues_calculator import power_method, inverse_power_method
from eigenvalues_calculator.main import EigenvaluesCalculatorException


class TestPowerMethod:

    @pytest.mark.parametrize(
        'matrix',
        (
            [[1, 2, 3], [2, 3, 1], [3, 1, 2]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 5], [7, 9]],
            [[1.23, 2.556, 3.345], [123.45321, 323.532, 112.34566], [0.00013, 1.2123, 0.66]],
            [[1000, 22222, 556743], [212312, 343453, 112312], [345323, 542351, 9487212]],
        ),
    )
    def test_general(self, matrix: list[list[int | float]]):
        eigenvalue = power_method(matrix)
        assert eigenvalue == pytest.approx(max(np.linalg.eig(np.array(matrix)).eigenvalues, key=abs))

    @pytest.mark.parametrize(
        ('matrix', 'expected_msg'),
        (
            ([], 'Пустая матрица'),
            ([[], []], 'Пустая матрица'),
        ),
    )
    def test_error(self, matrix: list[list[int | float]], expected_msg: str):
        with pytest.raises(EigenvaluesCalculatorException, match=expected_msg):
            _ = power_method(matrix)


class TestInversePowerMethod:

    @pytest.mark.parametrize(
        'matrix',
        (
            [[0, 5], [7, 9]],
            [[1000, 22222, 556743], [212312, 343453, 112312], [345323, 542351, 9487212]],
        ),
    )
    def test_general(self, matrix: list[list[int | float]]):
        eigenvalue = inverse_power_method(matrix)
        assert eigenvalue == pytest.approx(min(np.linalg.eig(np.array(matrix)).eigenvalues, key=abs))

    @pytest.mark.parametrize(
        ('matrix', 'expected_msg'),
        (
            ([], 'Пустая матрица'),
            ([[], []], 'Пустая матрица'),
        ),
    )
    def test_error(self, matrix: list[list[int | float]], expected_msg: str):
        with pytest.raises(EigenvaluesCalculatorException, match=expected_msg):
            _ = inverse_power_method(matrix)
