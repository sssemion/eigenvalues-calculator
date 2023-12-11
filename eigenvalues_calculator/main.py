import numpy as np


class EigenvaluesCalculatorException(Exception):
    pass


class EmptyMatrixException(Exception):
    message = 'Пустая матрица'


def normalize(x: np.ndarray) -> tuple[int | float, np.ndarray]:
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def power_method(matrix: list[list[int | float]], num_iterations: int = 1000) -> int | float:
    """
    Степенной метод. Возвращает доминирующее собственное значение матрицы.
    Доминирующим собственным значением является значение с наибольшим абсолютным значением. Этот метод не возвращает
    соответствующий собственный вектор.
    """
    if not (n := len(matrix)) or not (m := len(matrix[0])):
        raise EigenvaluesCalculatorException('Пустая матрица')
    if num_iterations <= 0:
        raise EigenvaluesCalculatorException('num_iterations <= 0')
    if n != m:
        raise EigenvaluesCalculatorException('Прямоугольная матрица не поддерживается')
    matrix = np.array(matrix)

    # создаем "случайный" вектор
    x = np.array([1] * n)
    for _ in range(num_iterations):
        # умножаем матрицу на вектор
        x = np.dot(matrix, x)
        # нормализуем вектор
        lambda_1, x = normalize(x)

    return lambda_1
