import numpy as np


class EigenvaluesCalculatorException(Exception):
    pass


class EmptyMatrixException(Exception):
    message = 'Пустая матрица'


def normalize(x: np.ndarray) -> tuple[int | float, np.ndarray]:
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def _validate_square_matrix(matrix: list[list[int | float]]) -> int:
    if not (n := len(matrix)) or not (m := len(matrix[0])):
        raise EigenvaluesCalculatorException('Пустая матрица')
    if n != m:
        raise EigenvaluesCalculatorException('Прямоугольная матрица не поддерживается')
    return n


def power_method(matrix: list[list[int | float]], num_iterations: int = 1000) -> int | float:
    """
    Степенной метод. Возвращает доминирующее собственное значение матрицы.
    Доминирующим собственным значением является значение с наибольшим абсолютным значением. Этот метод не возвращает
    соответствующий собственный вектор.
    """
    n = _validate_square_matrix(matrix)
    if num_iterations <= 0:
        raise EigenvaluesCalculatorException('num_iterations <= 0')
    matrix = np.array(matrix)

    # создаем "случайный" вектор
    x = np.array([1] * n)
    for _ in range(num_iterations):
        # умножаем матрицу на вектор
        x = np.dot(matrix, x)
        # нормализуем вектор
        lambda_1, x = normalize(x)

    return lambda_1


def inverse_power_method(matrix: list[list[int | float]], num_iterations: int = 1000) -> int | float:
    n = _validate_square_matrix(matrix)
    if num_iterations <= 0:
        raise EigenvaluesCalculatorException('num_iterations <= 0')
    matrix = np.linalg.inv(np.array(matrix))

    # создаем "случайный" вектор
    x = x = np.ones(matrix.shape[0])

    for _ in range(num_iterations):
        x_new = matrix @ x
        lambda_new = np.dot(x, x_new) / np.dot(x, x)
        x = x_new / np.linalg.norm(x_new)

        lambda_1 = lambda_new

    return 1 / lambda_1
