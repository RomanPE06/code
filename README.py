import numpy as np

def input_matrix():
    """Функция для ввода матрицы (ручной или из файла)"""
    print("Выберите способ ввода данных:")
    print("1. Вручную")
    print("2. Из файла")
    choice = input("Введите 1 или 2: ")

    if choice == '1':
        n = int(input("Введите количество переменных: "))
        m = int(input("Введите количество уравнений: "))
        print(f"Введите матрицу коэффициентов {m}x{n} (по строкам):")
        A = []
        for _ in range(m):
            row = list(map(float, input().split()))
            if len(row) != n:
                raise ValueError("Неверное количество коэффициентов в строке")
            A.append(row)
        print("Введите свободные члены через пробел:")
        b = list(map(float, input().split()))
        if len(b) != m:
            raise ValueError("Неверное количество свободных членов")
        return np.array(A), np.array(b), n, m

    elif choice == '2':
        filename = input("Введите имя файла: ")
        with open(filename, 'r') as file:
            lines = file.readlines()
        n = int(lines[0])
        m = int(lines[1])
        A = []
        for line in lines[2:2+m]:
            row = list(map(float, line.strip().split()))
            if len(row) != n:
                raise ValueError("Неверное количество коэффициентов в строке матрицы")
            A.append(row)
        b = list(map(float, lines[2+m].strip().split()))
        if len(b) != m:
            raise ValueError("Неверное количество свободных членов")
        return np.array(A), np.array(b), n, m

    else:
        raise ValueError("Неверный выбор")

def gauss_elimination(A, b, n, m):
    """Метод Гаусса с определением всех типов решений"""
    augmented = np.hstack((A, b.reshape(-1, 1)))
    print("\nНачальная расширенная матрица:")
    print(augmented)

    # Прямой ход
    rank = 0
    for col in range(n):
        if rank >= m:
            break

        # Поиск ведущего элемента
        pivot_row = np.argmax(np.abs(augmented[rank:, col])) + rank
        if np.isclose(augmented[pivot_row, col], 0):
            continue  # Все элементы в столбце нулевые

        # Перестановка строк
        augmented[[rank, pivot_row]] = augmented[[pivot_row, rank]]
        print(f"\nШаг {rank+1}: Меняем строки {rank+1} и {pivot_row+1}")
        print(augmented)

        # Нормализация
        pivot = augmented[rank, col]
        augmented[rank] = augmented[rank] / pivot
        print(f"Нормализуем строку {rank+1} (делим на {pivot})")
        print(augmented)

        # Исключение
        for i in range(rank + 1, m):
            factor = augmented[i, col]
            augmented[i] -= factor * augmented[rank]
            print(f"Вычитаем из строки {i+1} строку {rank+1}, умноженную на {factor}")
            print(augmented)

        rank += 1

    # Проверка на несовместность
    for row in augmented[rank:]:
        if not np.allclose(row[:-1], 0) and not np.isclose(row[-1], 0):
            print("\nСистема несовместна (найдена строка [0 ... 0 | b ≠ 0])")
            return None, None, False

    # Обратный ход (только для базисных переменных)
    solutions = {}
    free_vars = set(range(n))
    lead_cols = []

    for i in range(min(rank, m)):
        for j in range(n):
            if not np.isclose(augmented[i, j], 0):
                lead_cols.append(j)
                free_vars.discard(j)
                break

    # Выражаем базисные переменные
    for i in reversed(range(len(lead_cols))):
        col = lead_cols[i]
        solutions[col] = augmented[i, -1]
        for j in range(col + 1, n):
            if j in solutions:
                solutions[col] -= augmented[i, j] * solutions[j]
            elif j in free_vars:
                solutions[col] -= augmented[i, j] * Symbol(f't_{j}')

    # Формируем результат
    result = []
    for j in range(n):
        if j in free_vars:
            result.append(f"t_{j} (свободная переменная)")
        else:
            result.append(str(simplify(solutions.get(j, 0))))

    has_infinite_solutions = len(free_vars) > 0
    return result, free_vars, has_infinite_solutions

def print_solutions(result, free_vars, has_infinite_solutions, n):
    """Вывод решений в удобном формате"""
    if has_infinite_solutions:
        print("\nСистема имеет бесконечно много решений:")
        print("Свободные переменные:", ", ".join([f"t_{j}" for j in free_vars]))
    else:
        print("\nСистема имеет единственное решение:")

    for i in range(n):
        print(f"x_{i+1} = {result[i]}")

def main():
    try:
        A, b, n, m = input_matrix()
        result, free_vars, has_infinite_solutions = gauss_elimination(A, b, n, m)
        
        if result is not None:
            print_solutions(result, free_vars, has_infinite_solutions, n)
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    from sympy import Symbol, simplify
    main()
