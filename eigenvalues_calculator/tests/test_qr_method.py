import pytest
import numpy as np

from eigenvalues_calculator.main import EigenvaluesCalculatorException, qr_method


class TestQRMethod:

    @pytest.mark.parametrize(
        'matrix',
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 5], [7, 9]],
            [[1000, 22222, 556743], [212312, 343453, 112312], [345323, 542351, 9487212]],
        ),
    )
    def test_general(self, matrix: list[list[int | float]]):
        reference_result = np.linalg.eig(np.array(matrix))
        eigenvalues, eigenvectors = qr_method(matrix)
        for actual, expected in zip(
            sorted(eigenvalues),
            sorted(reference_result.eigenvalues)
        ):
            assert actual == pytest.approx(expected)


    @pytest.mark.parametrize(
        ('matrix', 'expected_msg'),
        (
            ([], 'Пустая матрица'),
            ([[], []], 'Пустая матрица'),
        ),
    )
    def test_error(self, matrix: list[list[int | float]], expected_msg: str):
        with pytest.raises(EigenvaluesCalculatorException, match=expected_msg):
            _ = qr_method(matrix)
