#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m, n, vec_size;

    // Считывание размеров матрицы и вектора (только на процессе 0)
    if (rank == 0) {
        cout << "Введите размер матрицы (m x n): ";
        cin >> m >> n;
        cout << "Введите размер вектора-столбца (должен быть равен n): ";
        cin >> vec_size;

        // Проверка корректности размеров
        if (n != vec_size) {
            cout << "Ошибка: размер вектора-столбца должен быть равен количеству столбцов матрицы!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (m % size != 0) {
            cout << "Ошибка: количество строк матрицы должно делиться на количество процессов!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Рассылка размеров матрицы и вектора-столбца от процесса с рангом 0 ко всем остальным процессам
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Создание матрицы и вектора-столбца на процессе 0
    vector<int> matrix(m * n);
    vector<int> vectorColumn(vec_size);

    // Генерация данных только на процессе 0
    if (rank == 0) {
        // Инициализация генератора случайных чисел
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, 9); // Равномерное распределение для случайных чисел от 0 до 9

        // Заполнение матрицы и вектора случайными числами от 0 до 9
        for (int i = 0; i < m * n; ++i) {
            matrix[i] = dis(gen);
        }
        for (int i = 0; i < vec_size; ++i) {
            vectorColumn[i] = dis(gen);
        }
    }

    // Выделение места для локальных частей матрицы и вектора на каждом процессе
    vector<int> localMatrix(m / size * n);
    vector<int> localVector(vec_size);

    // Разделение строк матрицы и частей вектора между процессами
    MPI_Scatter(matrix.data(), m / size * n, MPI_INT, localMatrix.data(), m / size * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vectorColumn.data(), vec_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Вычисление частичного результата на каждом процессе
    vector<int> partialResult(m / size, 0);
    for (int i = 0; i < m / size; ++i) {
        for (int j = 0; j < n; ++j) {
            partialResult[i] += localMatrix[i * n + j] * vectorColumn[j];
        }
    }

    // Сбор всех частичных результатов на процессе с рангом 0
    vector<int> finalResult(m, 0);
    MPI_Gather(partialResult.data(), m / size, MPI_INT, finalResult.data(), m / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Вывод результата на процессе с рангом 0
    if (rank == 0) {
        cout << "Результат умножения матрицы на вектор-столбец:" << endl;
        for (int i = 0; i < m; ++i) {
            cout << finalResult[i] << endl;
        }
    }

    MPI_Finalize();
    return 0;
}