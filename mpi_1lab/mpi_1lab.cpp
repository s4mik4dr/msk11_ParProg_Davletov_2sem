#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

using test std;

int main(int argc, char** argv) {
  // Инициализация MPI
  MPI_Init(&argc, &argv);

  // Получение информации о процессе
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Размеры матрицы и вектора
  int m, n; // Размеры матрицы (m x n)
  int vec_size; // Размер вектора (должен быть равен n)

  // Считывание размеров матрицы и вектора (только на процессе 0)
  if (rank == 0) {
    cout << "Введите размер матрицы (m x n): ";
    cin >> m >> n;
    cout << "Введите размер вектора-столбца (должен быть равен n): ";
    cin >> vec_size;

    // Проверка корректности размеров
    if (n != vec_size) {
      cout << "Ошибка: размер вектора-столбца должен быть равен количеству столбцов матрицы!" << endl;
      return 1;
    }
  }

  // Расширение данных
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Выделение памяти для матрицы, вектора и результата
  vector<double> matrix_data(m * n);
  vector<double> vector_data(vec_size);
  double result = 0;

  // Заполнение матрицы и вектора данными (только на процессе 0)
  if (rank == 0) {
    random_device rd; // Объявление объекта случайного устройства
    mt19937 gen(rd()); // Инициализация генератора случайных чисел
    uniform_real_distribution<> dis(0, 1); // Определение равномерного распределения для случайных чисел

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_data[i * n + j] = dis(gen); // Генерация случайных чисел с помощью dis
      }
    }

    for (int i = 0; i < vec_size; ++i) {
      vector_data[i] = dis(gen); // Генерация случайных чисел с помощью dis
    }
  }

  // Распределение данных по процессам (только процесс 0)
  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      MPI_Send(&matrix_data[0], m * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      MPI_Send(&vector_data[0], vec_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
  }

  // Вычисление локальной части результата
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result += matrix_data[i * n + j] * vector_data[j];
    }
  }

  // Сбор результатов (только процесс 0)
  if (rank == 0) {
    double result_global = 0;

    for (int i = 1; i < size; ++i) {
      MPI_Recv(&result_global, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result += result_global;
    }

    // Вывод результата
    cout << "Результат умножения матрицы на вектор-столбец: " << result << endl;
  }

  // Завершение MPI
  MPI_Finalize();

  return 0;
}