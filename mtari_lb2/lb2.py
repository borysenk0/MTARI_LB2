import numpy as np
import matplotlib.pyplot as plt
import time
import random


# --- 1. Визначення цільової функції (Варіант 2) ---

def rosenbrock_function(x):
    """Функція Розенброка (d=10) [cite: 12]"""
    d = len(x)
    sum_val = 0
    for i in range(d - 1):
        sum_val += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum_val


# --- 2. Клас Частинки ---

class Particle:
    """
    Клас для представлення однієї частинки в рої.
    Зберігає її поточний стан та найкращий особистий результат.
    """

    def __init__(self, n_dim, bounds_low, bounds_high):
        self.n_dim = n_dim
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high

        # Ініціалізація позиції випадковим чином в межах
        self.position = np.random.uniform(low=bounds_low, high=bounds_high, size=n_dim)
        # Ініціалізація швидкості нулями або малими випадковими значеннями
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=n_dim)

        # Найкраща особиста позиція (pbest)
        self.pbest_position = np.copy(self.position)
        self.pbest_value = float('inf')


# --- 3. Основний клас PSO ---

class PSO:
    """
    Реалізація алгоритму Particle Swarm Optimization.
    Підтримує 'gbest' та 'lbest' топології.
    """

    def __init__(self, func, n_dim, bounds, n_particles, n_iterations,
                 omega, c1, c2, topology='gbest', k_neighbors=4):

        # Параметри задачі
        self.func = func
        self.n_dim = n_dim
        self.bounds_low, self.bounds_high = bounds

        # Гіперпараметри PSO [cite: 4]
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.omega = omega  # Інерція
        self.c1 = c1  # Когнітивний коеф.
        self.c2 = c2  # Соціальний коеф.

        # Параметри для експерименту (Варіант 2)
        self.topology = topology
        self.k_neighbors = k_neighbors  # Кількість сусідів для lbest [cite: 14]

        # Ініціалізація рою
        self.swarm = [Particle(n_dim, self.bounds_low, self.bounds_high) for _ in range(n_particles)]

        # Глобально найкращі значення (gbest)
        self.gbest_position = np.zeros(n_dim)
        self.gbest_value = float('inf')

        # Історія збіжності для графіка
        self.convergence_history = []

    def _get_social_best(self, particle_index):
        """
        Ключова логіка для Варіанту 2: знаходження соціального лідера.
        [cite: 14]
        """
        if self.topology == 'gbest':
            # Для gbest лідер один для всіх - gbest_position
            return self.gbest_position

        elif self.topology == 'lbest':
            # Для lbest (кільце) - знаходимо найкращого серед k сусідів

            # Визначаємо індекси сусідів
            # Наприклад, для k=4 та індексу 10: [8, 9, 10, 11, 12] (включаючи себе)
            half_k = self.k_neighbors // 2
            indices = []
            for i in range(-half_k, half_k + 1):
                # Обробка "зациклення" індексів для топології "кільце"
                neighbor_idx = (particle_index + i) % self.n_particles
                indices.append(neighbor_idx)

            # Знаходимо найкращий pbest серед цієї групи
            lbest_value = float('inf')
            lbest_position = None

            for idx in set(indices):  # set() для уникнення дублікатів, якщо k > n_particles
                if self.swarm[idx].pbest_value < lbest_value:
                    lbest_value = self.swarm[idx].pbest_value
                    lbest_position = self.swarm[idx].pbest_position

            return lbest_position

        else:
            raise ValueError("Невідома топологія. Використовуйте 'gbest' або 'lbest'.")

    def solve(self):
        """Основний цикл оптимізації"""

        for iteration in range(self.n_iterations):
            # 1. Оцінка кожної частинки та оновлення pbest / gbest
            for i, particle in enumerate(self.swarm):

                current_value = self.func(particle.position)

                # Оновлення pbest
                if current_value < particle.pbest_value:
                    particle.pbest_value = current_value
                    particle.pbest_position = np.copy(particle.position)

                # Оновлення gbest
                if current_value < self.gbest_value:
                    self.gbest_value = current_value
                    self.gbest_position = np.copy(particle.position)

            # 2. Оновлення швидкостей та позицій частинок
            for i, particle in enumerate(self.swarm):
                # Отримання соціального лідера (gbest або lbest) [cite: 14]
                social_best_pos = self._get_social_best(i)

                r1 = np.random.random(self.n_dim)
                r2 = np.random.random(self.n_dim)

                # Формула оновлення швидкості [cite: 63, 68]
                cognitive_comp = self.c1 * r1 * (particle.pbest_position - particle.position)
                social_comp = self.c2 * r2 * (social_best_pos - particle.position)

                particle.velocity = (self.omega * particle.velocity) + cognitive_comp + social_comp

                # Формула оновлення позиції
                particle.position = particle.position + particle.velocity

                # Обробка меж: відсікання (найпростіший метод) [cite: 36]
                particle.position = np.clip(particle.position, self.bounds_low, self.bounds_high)

            # Збереження найкращого результату ітерації для графіка
            self.convergence_history.append(self.gbest_value)

            # (Опціонально) Вивід прогресу
            # if iteration % 100 == 0:
            #     print(f"Ітерація {iteration}: Best = {self.gbest_value:.4e}")

        # Повертаємо найкраще знайдене рішення та історію
        return self.gbest_position, self.gbest_value, self.convergence_history


# --- 4. Проведення експерименту (для оцінки "добре") ---

if __name__ == "__main__":

    # --- Налаштування експерименту ---

    # Параметри задачі 2
    DIMENSION = 10  # d=10 [cite: 12]
    BOUNDS = [-2.048, 2.048]  # xi∈[-2.048,2.048] [cite: 13]

    # Налаштування PSO (можете їх змінювати для дослідження)
    N_PARTICLES = 40
    N_ITERATIONS = 1000
    OMEGA = 0.7  # Інерція
    C1 = 1.5  # Когнітивний
    C2 = 1.5  # Соціальний

    N_RUNS = 10  # "запустіть щонайменше 10 незалежних прогонів"

    print("--- Розпочато експеримент (Варіант 2: gbest vs lbest) ---")
    print(f"Функція: Розенброк (d={DIMENSION})")
    print(f"Параметри PSO: Частинок={N_PARTICLES}, Ітерацій={N_ITERATIONS}, Runs={N_RUNS}")
    print(f"w={OMEGA}, c1={C1}, c2={C2}\n")

    # Списки для зберігання результатів 10 прогонів
    gbest_results = []
    gbest_histories = []
    lbest_results = []
    lbest_histories = []

    # --- Експеримент 1: 'gbest' ---
    print(f"Запуск 10 прогонів для 'gbest'...")
    start_time_gbest = time.time()

    for i in range(N_RUNS):
        # Встановлюємо різний seed для кожного прогону
        np.random.seed(i)
        random.seed(i)

        pso_gbest = PSO(rosenbrock_function, DIMENSION, BOUNDS,
                        n_particles=N_PARTICLES, n_iterations=N_ITERATIONS,
                        omega=OMEGA, c1=C1, c2=C2, topology='gbest')

        pos, val, hist = pso_gbest.solve()
        gbest_results.append(val)
        gbest_histories.append(hist)

    end_time_gbest = time.time()
    print(f"Завершено. Час: {end_time_gbest - start_time_gbest:.2f} сек.\n")

    # --- Експеримент 2: 'lbest' (кільце, 4 сусіди) ---
    print(f"Запуск 10 прогонів для 'lbest' (k=4)...")
    start_time_lbest = time.time()

    for i in range(N_RUNS):
        # Встановлюємо різний seed для кожного прогону
        np.random.seed(i)
        random.seed(i)

        pso_lbest = PSO(rosenbrock_function, DIMENSION, BOUNDS,
                        n_particles=N_PARTICLES, n_iterations=N_ITERATIONS,
                        omega=OMEGA, c1=C1, c2=C2,
                        topology='lbest', k_neighbors=4)  # [cite: 14]

        pos, val, hist = pso_lbest.solve()
        lbest_results.append(val)
        lbest_histories.append(hist)

        # (Опціонально) Зберегти найкращий вектор x для звіту [cite: 77]
        if val == min(lbest_results):
            best_solution_vector = pos

    end_time_lbest = time.time()
    print(f"Завершено. Час: {end_time_lbest - start_time_lbest:.2f} сек.\n")

    # --- 5. Аналіз результатів та Візуалізація ---

    print("--- Статистика по 10 прогонах (Якість та Стабільність)  ---")

    # Розрахунок статистики (best, mean, std)
    gbest_best = np.min(gbest_results)
    gbest_mean = np.mean(gbest_results)
    gbest_std = np.std(gbest_results)

    lbest_best = np.min(lbest_results)
    lbest_mean = np.mean(lbest_results)
    lbest_std = np.std(lbest_results)

    print(f"Топологія 'gbest':")
    print(f"  Найкращий (best): {gbest_best:.4e}")
    print(f"  Середній (mean):  {gbest_mean:.4e}")
    print(f"  Стаб. (std):    {gbest_std:.4e}\n")

    print(f"Топологія 'lbest' (k=4):")
    print(f"  Найкращий (best): {lbest_best:.4e}")
    print(f"  Середній (mean):  {lbest_mean:.4e}")
    print(f"  Стаб. (std):    {lbest_std:.4e}\n")

    print("--- Аналіз знайденого розв'язку (для кращого з алгоритмів) [cite: 77] ---")
    print(f"Очікуваний мінімум: f(x) = 0.0, при x_i = 1.0")
    print(f"Найкращий знайдений f(x): {lbest_best:.4e}")
    print(f"Найкращий знайдений вектор x (перші 5 компонент):")
    print(f"  {best_solution_vector[:5]}...")
    print(f"Відхилення від оптимуму (1.0): {np.mean(np.abs(best_solution_vector - 1.0)):.4e}")

    # --- Побудова графіка збіжності  ---

    # Усереднення історії збіжності по 10 прогонах
    gbest_avg_history = np.mean(np.array(gbest_histories), axis=0)
    lbest_avg_history = np.mean(np.array(lbest_histories), axis=0)

    plt.figure(figsize=(12, 7))
    # Використовуємо логарифмічну шкалу для осі Y, щоб краще бачити різницю
    plt.semilogy(gbest_avg_history, label="gbest (глобальна)", color='blue', linestyle='--')
    plt.semilogy(lbest_avg_history, label="lbest (кільце, k=4)", color='red')

    plt.title(f"Порівняння збіжності 'gbest' vs 'lbest' (Функція Розенброка, d={DIMENSION})")
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраще значення функції (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.ylim(bottom=1e-1)  # Встановіть нижню межу, якщо графік "прилипає" до нуля

    print("\nПоказ графіка збіжності...")
    plt.show()