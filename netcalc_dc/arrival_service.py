"""
Модуль с реализацией базовых объектов сетевого исчисления:
кривых поступления, сервис-кривых и вспомогательных операций.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class TimeStoppingError(ValueError):
    """Выбрасывается, если система time-stopping недопустима."""


@dataclass(frozen=True)
class ArrivalCurve:
    """
    Линейная кривая поступления вида alpha(t) = b + r * t.

    Параметр r задаёт устоявшуюся скорость поступления, а b ограничивает всплеск.
    """

    rate: float
    burst: float

    def value(self, t: float) -> float:
        """Возвращает значение кривой в момент времени t."""
        if t < 0:
            raise ValueError("t must be non-negative")
        return self.burst + self.rate * t

    def delay_upper_bound(self, service: "ServiceCurve") -> float:
        """
        Вычисляет верхнюю границу задержки для пары (alpha, beta).

        Для строгой сервис-кривой beta_{R,T}(t) = R * max(0, t - T) верхняя граница
        равна T + b / (R - r) при условии r < R. Если условие нарушено,
        задержка считается бесконечной.
        """
        if self.rate >= service.rate:
            return float("inf")
        return service.latency + self.burst / (service.rate - self.rate)


@dataclass(frozen=True)
class ServiceCurve:
    """
    Линейная сервис-кривая вида beta(t) = R * max(0, t - T).

    Здесь R --- гарантированная скорость, T --- латентность.
    """

    rate: float
    latency: float

    def value(self, t: float) -> float:
        """Возвращает значение сервис-кривой в момент t."""
        if t < 0:
            raise ValueError("t must be non-negative")
        return max(0.0, self.rate * (t - self.latency))

    def convolve(self, other: "ServiceCurve") -> "ServiceCurve":
        """
        Мин-плюс конволюция двух линейных сервис-кривых.

        Для линейных элементов результатом является снова линейная сервиса-кривая
        с суммарной латентностью и минимальной скоростью.
        """
        if self.rate <= 0 or other.rate <= 0:
            raise ValueError("Service rate must be positive")
        new_rate = min(self.rate, other.rate)
        new_latency = self.latency + other.latency
        return ServiceCurve(rate=new_rate, latency=new_latency)

    @staticmethod
    def convolve_many(curves: Iterable["ServiceCurve"]) -> "ServiceCurve":
        """Конволюция последовательности сервис-кривых."""
        iterator = iter(curves)
        try:
            result = next(iterator)
        except StopIteration:
            raise ValueError("At least one service curve is required") from None
        for curve in iterator:
            result = result.convolve(curve)
        return result


def spectral_radius(matrix: np.ndarray) -> float:
    """
    Возвращает спектральный радиус матрицы.

    Используется при проверке условия rho(A) < 1 для метода остановки времени.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return float(max(abs(l) for l in eigenvalues))


def solve_time_stopping(
    influence: np.ndarray,
    base_delays: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Решает систему неравенств x <= A x + a, возникающую в методе остановки времени.

    Возвращает вектор оценок задержек x = (I - A)^{-1} a. Если спектральный
    радиус матрицы A не меньше 1, выбрасывается исключение TimeStoppingError.
    """
    if influence.shape[0] != influence.shape[1]:
        raise ValueError("Influence matrix must be square")
    if base_delays.shape[0] != influence.shape[0]:
        raise ValueError("Incompatible shapes for influence matrix and base delays")
    rho = spectral_radius(influence)
    if rho >= 1.0 - epsilon:
        raise TimeStoppingError(
            f"Time-stopping system infeasible: spectral radius {rho:.6f} >= 1"
        )
    identity = np.eye(influence.shape[0])
    solution = np.linalg.solve(identity - influence, base_delays)
    return solution


def aggregate_service_curve(
    edges: Sequence[ServiceCurve],
) -> ServiceCurve:
    """
    Формирует эквивалентную сервис-кривую маршрута как конволюцию всех звеньев.
    """
    return ServiceCurve.convolve_many(edges)


def aggregate_arrival_curve(flows: Iterable[ArrivalCurve]) -> ArrivalCurve:
    """
    Агрегирует несколько потоков с одинаковой скоростью сервисов.

    Получается суммарная кривая поступления с суммарным всплеском и скоростью.
    """
    total_rate = 0.0
    total_burst = 0.0
    for flow in flows:
        total_rate += flow.rate
        total_burst += flow.burst
    return ArrivalCurve(rate=total_rate, burst=total_burst)
