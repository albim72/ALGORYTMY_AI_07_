"""
Prosty przykład algorytmu Q-learning w środowisku GridWorld.

Agent startuje w lewym dolnym rogu, uczy się omijać przeszkody
i docierać do celu w prawym górnym rogu.

Wymagania:
    pip install numpy

Uruchomienie:
    python q_learning_gridworld.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


State = tuple[int, int]


@dataclass(frozen=True)
class GridWorld:
    """Dwuwymiarowe środowisko dla agenta Q-learning."""

    rows: int = 6
    cols: int = 6
    start: State = (5, 0)
    goal: State = (0, 5)
    obstacles: frozenset[State] = frozenset(
        {
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (3, 3),
            (4, 4),
        }
    )

    # 0 = góra, 1 = dół, 2 = lewo, 3 = prawo
    actions: Final[tuple[State, ...]] = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    def step(self, state: State, action: int) -> tuple[State, float, bool]:
        """
        Wykonaj akcję w środowisku.

        Zwraca:
            nowy_stan, nagroda, czy_epizod_zakończony
        """
        row_change, col_change = self.actions[action]
        candidate = (
            state[0] + row_change,
            state[1] + col_change,
        )

        # Wyjście poza planszę.
        if not self.is_inside(candidate):
            return state, -5.0, False

        # Próba wejścia na przeszkodę.
        if candidate in self.obstacles:
            return state, -10.0, False

        # Osiągnięcie celu.
        if candidate == self.goal:
            return candidate, 100.0, True

        # Zwykły krok ma mały koszt, aby agent szukał krótkiej drogi.
        return candidate, -1.0, False

    def is_inside(self, state: State) -> bool:
        """Sprawdź, czy stan znajduje się na planszy."""
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols


def choose_action(
    q_table: np.ndarray,
    state: State,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    Wybierz akcję metodą epsilon-greedy.

    Z prawdopodobieństwem epsilon agent eksploruje losową akcję.
    W przeciwnym przypadku wybiera akcję o najwyższej wartości Q.
    """
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[2]))

    row, col = state
    return int(np.argmax(q_table[row, col]))


def train_q_learning(
    environment: GridWorld,
    episodes: int = 6000,
    learning_rate: float = 0.15,
    discount_factor: float = 0.95,
    initial_epsilon: float = 1.0,
    minimum_epsilon: float = 0.02,
    epsilon_decay: float = 0.995,
    max_steps_per_episode: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """
    Wytrenuj agenta Q-learning.

    Równanie aktualizacji:

    Q(s,a) = Q(s,a) + alfa * [
        nagroda + gamma * max Q(s',a') - Q(s,a)
    ]
    """
    rng = np.random.default_rng(seed)

    q_table = np.zeros(
        (environment.rows, environment.cols, len(environment.actions)),
        dtype=float,
    )

    rewards_history: list[float] = []
    epsilon = initial_epsilon

    for _episode in range(episodes):
        state = environment.start
        total_reward = 0.0

        for _step in range(max_steps_per_episode):
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, done = environment.step(state, action)

            row, col = state
            next_row, next_col = next_state

            current_q = q_table[row, col, action]

            # Po osiągnięciu celu nie dodajemy wartości przyszłego stanu.
            best_future_q = 0.0 if done else np.max(
                q_table[next_row, next_col]
            )

            temporal_difference = (
                reward
                + discount_factor * best_future_q
                - current_q
            )

            q_table[row, col, action] += (
                learning_rate * temporal_difference
            )

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)
        epsilon = max(minimum_epsilon, epsilon * epsilon_decay)

    return q_table, rewards_history


def find_best_path(
    environment: GridWorld,
    q_table: np.ndarray,
    max_steps: int = 100,
) -> list[State]:
    """Odczytaj najlepszą ścieżkę wynikającą z wyuczonej tablicy Q."""
    state = environment.start
    path = [state]
    visited = {state}

    for _ in range(max_steps):
        row, col = state
        action = int(np.argmax(q_table[row, col]))

        next_state, _reward, done = environment.step(state, action)
        path.append(next_state)

        if done:
            return path

        # Zabezpieczenie przed utknięciem lub cyklem.
        if next_state == state or next_state in visited:
            raise RuntimeError(
                "Agent utknął. Zwiększ liczbę epizodów treningowych."
            )

        visited.add(next_state)
        state = next_state

    raise RuntimeError("Nie znaleziono celu w zadanym limicie kroków.")


def print_board(environment: GridWorld, path: list[State]) -> None:
    """Wyświetl planszę oraz najlepszą znalezioną ścieżkę."""
    path_set = set(path)

    print("\nLEGENDA: S=start, G=cel, #=przeszkoda, ·=ścieżka\n")

    for row in range(environment.rows):
        symbols: list[str] = []

        for col in range(environment.cols):
            state = (row, col)

            if state == environment.start:
                symbol = "S"
            elif state == environment.goal:
                symbol = "G"
            elif state in environment.obstacles:
                symbol = "#"
            elif state in path_set:
                symbol = "·"
            else:
                symbol = " "

            symbols.append(symbol)

        print(" | ".join(symbols))


def print_policy(environment: GridWorld, q_table: np.ndarray) -> None:
    """Wyświetl najlepszą akcję w każdym stanie."""
    arrows = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→",
    }

    print("\nWYUCZONA POLITYKA:\n")

    for row in range(environment.rows):
        symbols: list[str] = []

        for col in range(environment.cols):
            state = (row, col)

            if state == environment.start:
                symbol = "S"
            elif state == environment.goal:
                symbol = "G"
            elif state in environment.obstacles:
                symbol = "#"
            else:
                best_action = int(np.argmax(q_table[row, col]))
                symbol = arrows[best_action]

            symbols.append(symbol)

        print(" | ".join(symbols))


def main() -> None:
    """Uruchom trening i pokaż wynik."""
    environment = GridWorld()

    q_table, rewards = train_q_learning(
        environment=environment,
        episodes=6000,
        learning_rate=0.15,
        discount_factor=0.95,
        seed=42,
    )

    best_path = find_best_path(environment, q_table)

    print_policy(environment, q_table)
    print_board(environment, best_path)

    print("\nNajlepsza ścieżka:")
    print(" -> ".join(map(str, best_path)))

    print(f"\nLiczba ruchów: {len(best_path) - 1}")
    print(
        "Średnia nagroda z ostatnich 100 epizodów: "
        f"{np.mean(rewards[-100:]):.2f}"
    )

    print("\nWartości Q dla stanu początkowego:")
    print(
        "góra={:.2f}, dół={:.2f}, lewo={:.2f}, prawo={:.2f}".format(
            *q_table[environment.start[0], environment.start[1]]
        )
    )


if __name__ == "__main__":
    main()
