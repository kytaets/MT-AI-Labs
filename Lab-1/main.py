import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import skfuzzy as fuzz

def main():
    x = np.linspace(-10, 10, 1000)
    views = []

    # 1. Трикутна та трапецієподібна
    tri = fuzz.trimf(x, [-6, -2, 2])
    trap = fuzz.trapmf(x, [-8, -4, -1, 3])
    views.append(("Трикутна і трапецієподібна МФ", x, [tri, trap], ["Трикутна МФ", "Трапецієподібна МФ"]))

    # 2. Гаусові
    g = fuzz.gaussmf(x, 0, 1.5)
    g2 = fuzz.gauss2mf(x, 1.0, 0.5, 2.0, 3)
    g3 = fuzz.gauss2mf(x, 2.0, 0.3, 3.0, 4)
    views.append(("Гаусові МФ", x, [g, g2, g3], ["Гаусова МФ", "Двостороння гаусова МФ #1", "Двостороння гаусова МФ #2"]))

    # 3. Узагальнена дзвоноподібна
    y1 = fuzz.gbellmf(x, 1.5, 2.0, 0.0)
    views.append(("Узагальнена дзвоноподібна МФ", x, [y1], ["Узагальнений дзвін"]))

    # 4. Сигмоїдальні
    s_right = fuzz.sigmf(x, 2, 1.5)
    s_two = fuzz.dsigmf(x, -2, 2, 2, 2)
    s_asym = fuzz.psigmf(x, 2, 2, 1, 1.5)
    views.append(("Сигмоїдальні МФ", x, [s_right, s_two, s_asym],
                  ["Відкрита справа", "Двостороння", "Асиметрична"]))

    # 5. Поліноміальні функції
    z = fuzz.zmf(x, -6, -2)
    s = fuzz.smf(x, 2, 6)
    pi = fuzz.pimf(x, -3, -1, 1, 3)
    views.append(("Z, S і Π функції", x, [z, s, pi], ["Z-функція", "S-функція", "Π-функція"]))

    # 6. Мінімаксна інтерпретація (min/max)
    A = fuzz.gaussmf(x, -1, 2.0)
    B = fuzz.gaussmf(x, 1, 2.0)

    and_ab = np.fmin(A, B)  # AND = min
    or_ab = np.fmax(A, B)  # OR  = max

    # Min
    views.append((
        "Мінімальна інтерпретація",
        x,
        [A, B, and_ab],
        ["A", "B", "A AND B (min)"]
    ))

    # Max
    views.append((
        "Максимальна інтерпретація",
        x,
        [A, B, or_ab],
        ["A", "B", "A OR B (max)"]
    ))

    # 7. Ймовірнісна інтерпретація
    A = 1 / (1 + np.exp(-(x + 1)))
    B = 1 / (1 + np.exp(-(x - 1)))

    # AND (кон’юнкція)
    prob_and_ab = A * B
    views.append((
        "Кон’юнкція AND",
        x,
        [A, B, prob_and_ab],
        ["A", "B", "A І B"]
    ))

    # OR (диз’юнкція)
    prob_or_ab = A + B - A * B
    views.append((
        "Диз’юнкція OR",
        x,
        [A, B, prob_or_ab],
        ["A", "B", "A АБО B"]
    ))

    # 8. Доповнення
    mu = 1 / (1 + np.exp(-(x - 3)))
    comp = 1 - mu
    views.append(("Доповнення нечіткої множини", x, [mu, comp], ["Функція", "Доповнення"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(bottom=0.16)

    state = {"i": 0}

    def draw_view(i):
        ax.clear()
        title, X, ys, labels = views[i]
        for y, lbl in zip(ys, labels):
            ax.plot(X, y, label=lbl)
        ax.legend()
        ax.set_title(f"{title} ({i+1}/{len(views)})")
        fig.canvas.draw_idle()

    def next_view(event=None):
        state["i"] = (state["i"] + 1) % len(views)
        draw_view(state["i"])

    def prev_view(event=None):
        state["i"] = (state["i"] - 1) % len(views)
        draw_view(state["i"])

    ax_prev = fig.add_axes([0.40, 0.04, 0.06, 0.05])
    ax_next = fig.add_axes([0.47, 0.04, 0.06, 0.05])
    btn_prev = Button(ax_prev, '◀')
    btn_next = Button(ax_next, '▶')
    btn_prev.on_clicked(lambda ev: prev_view())
    btn_next.on_clicked(lambda ev: next_view())

    def on_key(event):
        if event.key in ('right', '→'):
            next_view()
        elif event.key in ('left', '←'):
            prev_view()

    fig.canvas.mpl_connect('key_press_event', on_key)

    draw_view(0)
    plt.show()


if __name__ == "__main__":
    main()