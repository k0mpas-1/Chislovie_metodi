# Praktika_1.py
import time, math
import numpy as np

np.set_printoptions(suppress=True)
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def measure(fn, *args, repeat=3):
    best, result = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return result, best


def show(f_loop, f_np, *args, repeat=3, ok=None):
    r1, t1 = measure(f_loop, *args, repeat=repeat)
    r2, t2 = measure(f_np, *args, repeat=repeat)
    same = ok(r1, r2) if ok else np.allclose(r1, r2)
    print(f"  цикл {t1*1000:.2f} мс | NumPy {t2*1000:.2f} мс | ускорение {t1/t2:.1f}× | совпало: {same}")
    return r1, r2, t1, t2


# 1. Сумма массива
print("\nЗадача 1. Сумма массива")
def loop_sum(x):
    s = 0.0
    for v in x:
        s += v
    return s
for n in (10**5, 10**6, 10**7):
    print(f" n = {n:,}:", end="")
    show(loop_sum, np.sum, rng.normal(size=n), repeat=2)
print("  Вывод: разница >10× видна уже при n = 10^5 (и раньше). Цикл тратит время на каждый элемент")
print("  (интерпретатор), NumPy проходит по памяти одним циклом на C; на очень больших n упирается в память.")

# 2. Среднее и std
print("\nЗадача 2. Среднее и std")
def loop_ms(x):
    n, s, s2 = len(x), 0.0, 0.0
    for v in x:
        s += v
        s2 += v * v
    m = s / n
    return m, math.sqrt(max(s2 / n - m * m, 0.0))
r1, r2, *_ = show(loop_ms, lambda x: (x.mean(), x.std()), a)
print(f"  разница: {abs(r1[0]-r2[0]):.1e} (mean), {abs(r1[1]-r2[1]):.1e} (std) — меньше 1e-12")
sh = 1e9 + rng.normal(size=1_000_000)
print(f"  данные со смещением 1e9: цикл std = {loop_ms(sh)[1]:.1f}, NumPy std = {sh.std():.3f} (истина ≈ 1)")
print("  Вывод: формула «сумма квадратов − квадрат среднего» теряет точность. NumPy считает std в два")
print("  прохода (среднее, затем квадраты отклонений) и суммирует попарно — поэтому устойчивее.")

# 3. y = sin(x) + x^2
print("\nЗадача 3. y = sin(x) + x^2")
def loop_f(x):
    return [math.sin(v) + v * v for v in x]
show(loop_f, lambda x: np.sin(x) + x**2, a)
print(f"  память: массив {a.nbytes/1e6:.0f} МБ; список — миллион отдельных объектов (≥ 32 Б на элемент)")
print("  Вывод: NumPy короче (1 строка). В np.sin(a) + a**2 создаются 3 массива: sin(a), a**2 и сумма")
print("  (2 временных + результат).")

# 4. Clipping
print("\nЗадача 4. Clipping в [0, 1]")
def loop_clip(x):
    out = []
    for v in x:
        if v < 0:
            out.append(0.0)
        elif v > 1:
            out.append(1.0)
        else:
            out.append(v)
    return out
show(loop_clip, lambda x: np.clip(x, 0, 1), a)
print(f"  изменено элементов: {np.sum((a < 0) | (a > 1)):,}")
print("  Вывод: в цикле 1 for и 2 if, в NumPy — 0 циклов и 0 if; имя np.clip сразу говорит, что делается.")

# 5. Сумма по строкам
print("\nЗадача 5. Сумма по строкам (100000, 10)")
m = rng.normal(size=(100_000, 10))
def loop_rows(mat):
    res = [0.0] * mat.shape[0]
    for i in range(mat.shape[0]):
        s = 0.0
        for j in range(mat.shape[1]):
            s += mat[i, j]
        res[i] = s
    return res
show(loop_rows, lambda mat: mat.sum(axis=1), m)
print("  Вывод: m.sum(axis=1) читается как «сумма по строкам», цикл — как индексы и счётчики; проще поддерживать NumPy.")

# 6. Центрирование по столбцам
print("\nЗадача 6. Вычитание среднего по столбцам")
def loop_center(mat):
    n, k = mat.shape
    out = np.empty_like(mat)
    for j in range(k):
        mean = sum(mat[i, j] for i in range(n)) / n
        for i in range(n):
            out[i, j] = mat[i, j] - mean
    return out
r1, r2, *_ = show(loop_center, lambda mat: mat - mat.mean(axis=0), m)
print(f"  макс. |среднее столбцов| после: {np.abs(r1.mean(axis=0)).max():.1e} / {np.abs(r2.mean(axis=0)).max():.1e} (≈ 0)")
print("  Вывод: NumPy — одна строка (broadcasting), цикл — 8 строк с индексами [i, j].")

# 7. max и argmax
print("\nЗадача 7. max и argmax")
def loop_max(x):
    mv, mi = x[0], 0
    for i in range(1, len(x)):
        if x[i] > mv:
            mv, mi = x[i], i
    return mv, mi
show(loop_max, lambda x: (x.max(), x.argmax()), a, ok=lambda p, q: p == q)
b = a.copy(); b[0] = np.nan
print(f"  NaN первым: цикл → {loop_max(b)[0]}, a.max() → {b.max()}, nanmax → {np.nanmax(b):.4f}")
print("  Вывод: сравнения с NaN всегда False, поэтому цикл «ломается» (если NaN не первый — молча его")
print("  пропускает). nanmax/nanargmax явно игнорируют NaN — это защита от реальных данных с пропусками.")

# 8. Фильтрация
print("\nЗадача 8. Фильтрация x > 0")
r1, r2, *_ = show(lambda x: [v for v in x if v > 0], lambda x: x[x > 0], a)
print(f"  память: ndarray {r2.nbytes/1e6:.0f} МБ (8 Б/элемент), список ≈ {len(r1)*40/1e6:.0f} МБ (≈ 40 Б/элемент) — в ~5 раз больше")
print("  Вывод: результат цикла — список объектов Python, NumPy — компактный массив.")

# 9. In-place
print("\nЗадача 9. Отрицательные → модуль (in-place)")
def loop_abs(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = -x[i]
    return x
def np_abs(x):
    x[x < 0] *= -1
    return x
x1, x2 = a.copy(), a.copy()
_, t1 = measure(loop_abs, x1, repeat=1)
_, t2 = measure(np_abs, x2, repeat=1)
print(f"  цикл {t1*1000:.2f} мс | NumPy {t2*1000:.2f} мс | ускорение {t1/t2:.1f}× | верно: {np.array_equal(x1, np.abs(a)) and np.array_equal(x2, np.abs(a))}")
y = a.copy()
print(f"  in-place (тот же массив): {np_abs(y) is y}")
print("  Вывод: оба варианта меняют исходный массив. In-place важен для больших данных: копия 8 ГБ потребует ещё 8 ГБ.")

# 10. Комбинированная маска
print("\nЗадача 10. Маска -1 < x < 1")
show(lambda x: [v for v in x if -1 < v < 1], lambda x: x[(x > -1) & (x < 1)], a)
try:
    a > -1 & a < 1
except Exception as e:
    print(f"  без скобок: {type(e).__name__}")
print("  Вывод: & имеет приоритет выше, чем < и >, поэтому a > -1 & a < 1 = a > (-1 & a) < 1 — ошибка. Скобки обязательны.")

# 11. Матрица расстояний
print("\nЗадача 11. Матрица расстояний 1000×1000")
X, Y = rng.normal(size=(1000, 3)), rng.normal(size=(1000, 3))
def loop_dist(X, Y):
    D = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            D[i, j] = math.sqrt(sum((X[i, k] - Y[j, k]) ** 2 for k in range(3)))
    return D
def np_dist(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
def np_dist2(X, Y):
    d2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(d2, 0))
r1, t1 = measure(loop_dist, X, Y, repeat=1)
r2, t2 = measure(np_dist, X, Y)
r3, t3 = measure(np_dist2, X, Y)
print(f"  цикл {t1*1000:.0f} мс | broadcasting {t2*1000:.1f} мс ({t1/t2:.0f}×) | X @ Y.T {t3*1000:.1f} мс ({t1/t3:.0f}×)")
print(f"  совпало: {np.allclose(r1, r2)}, {np.allclose(r1, r3, atol=1e-6)}; временный массив (1000,1000,3) = {1000*1000*3*8/1e6:.0f} МБ")
print("  Вывод: broadcasting — самый короткий код, но тратит память (24 МБ; для 20 000 точек ~9.6 ГБ).")
print("  Через X @ Y.T (или scipy cdist) памяти меньше и быстрее. Компромисс: простота ↔ память.")
