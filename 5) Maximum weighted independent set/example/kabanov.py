import numpy as np # для быстрой работы с массивами
import pandas as pd # для вывода таблицы (v 2.1.1)
import random # для рандомизированного алгоритма
import time # для подсчёта времени работы
import csv # для сохранения ответов
import math # для округления чисел в большую сторону (ceil)

# Так как "независимое множество графа G = клика в дополнении графа G", то задачу можно переформулировать в поиск максимальной взвешенной клики в дополнении графа.

# настройки (для выполнения кода может потребоваться изменение data_path и solutions_path)
runs = 10 # число запусков для усреднения времени
iterations = 1000 # сколько итераций должно пройти без улучшения ответа, чтобы алгоритм вернул текущий лучший
eps = 0.0001 # с какой погрешностью считать, что числа одинаковые
impact_degree = 10 # степень влиянися количества соседей у вершины на вероятность её выбора в клику
impact_weight = 1 # степень влиянися веса вершины на вероятность её выбора в клику
data_path = "./data/" # путь до папки с входными данными
solutions_path = "./solutions/" # путь, куда будут сохраняться посчитанные решения
files = ["brock200_1", "brock200_2", "brock200_3", "brock200_4", "c-fat200-1", "c-fat200-2", "c-fat200-5", "c-fat500-1", "c-fat500-10", "c-fat500-2", "c-fat500-5", "C125.9", "gen200_p0.9_44", "gen200_p0.9_55",  "johnson8-2-4",  "johnson8-4-4", "johnson16-2-4", "hamming6-2", "hamming6-4", "hamming8-2", "hamming8-4", "keller4", "MANN_a9", "MANN_a27", "MANN_a45", "p_hat300-1", "p_hat300-2", "p_hat300-3", "san200_0.7_1", "san200_0.7_2", "san200_0.9_1", "san200_0.9_2", "san200_0.9_3", "sanr200_0.7"] # все файлы, на которых должен быть протестирован код


# вспомогательные функции, не участвующие в работе алгоритма
def get_complement_edges(edges: dict) -> dict:
    """
    Функция, возвращающая список смежности дополнения графа.\n
    Parameters:
        * edges: словарь смежных вершинам вершин, описывающий граф (по i-му индексу находится set со смежными i-ой вершине вершинами)\n
    Returns:
        * dict: словарь смежных вершинам вершин, описывающий дополнение графа
    """
    num_vertices = len(edges.keys()) # число вершин в графе
    all_vertices = set(range(num_vertices)) # set из всех вершин

    edges_complement = {} # словарь как список смежности дополнения графа
    for v in edges.keys(): # идём по вершинам-ключам в списке смежности для изначального графа
        edges_complement[v] = all_vertices - edges[v] - set([v]) # оставляем только те вершины, которые ранее не были смежны с вершиной v (саму вершину v тоже убираем)
    return edges_complement # возвращаем словарь смежных вершинам вершин, описывающий дополнение графа

def check_solution(edges_complement: dict, weights: list, solution, eps=0.0001) -> None:
    """
    Функция для проверка полученного лучшего решения.\n
    Parameters:
        * edges_complement: словарь смежных вершинам вершин, описывающий дополнение графа
        * weights: список весов вершин, где index — номер вершины
        * solution: решение в формате [вес независимого множества, [вершины независимого множества]]
        * eps: с какой погрешностью считать, что числа одинаковые\n
    Returns:
        * None: выкидывает исключение, если в решении есть ошибки
    """
    check_weight_sum = 0 # проверочная сумма весов вершин
    for v in solution[1]: # идём по номерам вершин независимого множества
        check_weight_sum += weights[v] # увеличиваем проверочную сумму весов вершин на значение веса рассматриваемой вершины

    if not (check_weight_sum - eps <= solution[0] <= check_weight_sum + eps): # проверка соответствия веса, полученного алгоритмом
        raise Exception("Weight is incorrect!") # выкидываем ошибку, если веса не сошлись

    for i in range(len(solution[1])-1): # идём по вершинам и проверяем, смежны ли она со всеми остальными вершинами в найденной клике ДОПОЛНЕНИЯ графа
        for j in range(i+1, len(solution[1])): # идём по последующим вершинам в клике
            if solution[1][j] not in edges_complement[solution[1][i]]: # проверяем наличие ребра между вершинами
                raise Exception(f"Nodes {i} and {j} connected!") # выкидываем ошибку, если вершины в изначальном графе были смежными (в дополнении между ними не должно быть ребра)

def transform_solution(solution, round_numbers=2) -> list:
    """
    Функция для преобразования ответа, чтобы номера вершин шли не с 0, а с 1 и по порядку.\n
    Parameters:
        * solution: решение в формате [вес независимого множества, [вершины независимого множества]]
        * round_numbers: число сохраняемых чисел после запятой\n
    Returns:
        * list: решение в формате [вес независимого множества, [отсортированные инкрементированные вершины независимого множества]]
    """
    for i in range(len(solution[1])): # идём по числу вершин в решении
        solution[1][i] += 1 # инкрементируем номер вершины (чтобы они шли не с 0, а с 1)
    return [round(solution[0], round_numbers), sorted(solution[1])] # возвращаем решение, попутно отсортировав инкрементированные вершины и округляя до round_numbers чисел после запятой

def save_solution(dataset, solution: dict) -> None:
    """
    Функция для сохранения лучших ответов.\n
    Parameters:
        * dataset: название тест-кейса
        * solution: словарь для тест-кейса с решениями в формате {"time": время на подсчёт, "weight": вес получившегося независимого множества, "independent_set": [вершины, входящие в независимое множество]}\n
    Returns:
        * None: сохраняет решение
    """
    with open(f"{solutions_path}{dataset}.csv", 'w', newline='') as file: # открываем файл для чистой записи
        writer = csv.writer(file) # создаём объект для записи
        writer.writerow([solution["weight"]]) # сохраняем размер клики (writerow — сохранение одного элемента в строку)
        writer.writerows([solution["independent_set"]]) # сохраняем вершины клики (writerows — сохранение итерационных данных по типу списка в строку)
        writer.writerow([solution["time"]]) # сохраняем время работы  (writerow — сохранение одного элемента в строку)


# получение данных
data = {} 
for file in files:
    data[file] = {"vertex_num": None, "edge_num": None, "edges": {}}
    with open(f"{data_path}{file}.clq", "r") as f: # открываем файл для чтения
        for row in f: # проходим по строкам
            if row[0] == "c": # если строка начинается с буквы "c" - это комментарий, пропускае строку
                continue
            elif row[0] == "p": # если строка начинается с буквы "p" - это описание проблемы, берём из этой строки число вершин и рёбер (последние два числа)
                data[file]["vertex_num"], data[file]["edge_num"] = int(row.split()[-2]), int(row.split()[-1])
            elif row[0] == "e": # если строка начинается с буквы "p" - это вершины, между которыми есть ребро
                v1, v2 = int(row.split()[-2]) - 1, int(row.split()[-1]) - 1 # запоминаем вершины (-1, чтобы не было мороки с индексацией)

                # добавляем связь вершины v1 с v2
                if v1 not in data[file]["edges"].keys(): # если это первое упоминание вершины v1 - создадим для неё set с указанием v2
                    data[file]["edges"][v1] = {v2}
                elif v2 not in data[file]["edges"][v1]: # иначе - просто добавим v2 в set смежных вершин v1
                    data[file]["edges"][v1].add(v2)

                # аналогично, но относительно вершины v2
                if v2 not in data[file]["edges"].keys():
                    data[file]["edges"][v2] = {v1}
                elif v1 not in data[file]["edges"][v2]:
                    data[file]["edges"][v2].add(v1)
        data[file]["edges"] = dict(sorted(data[file]["edges"].items())) # отсортируем вершины в словаре (в set для ключа словаря вершины уже отсортированы)

weights = {}
for dataset in data.keys(): # идём по тест-кейсам
    weights[dataset] = [math.ceil(10*i / data[dataset]["vertex_num"]) * 0.1 for i in range(1, data[dataset]["vertex_num"]+1)] 
# задаём веса вершин по формуле w_i = ceil(10*i / n) * 0.1, где n - число вершин, i - номер вершины (начиная с 1)
# То есть первые 10% вершин имеют вес 0.1, следующие 10% - вес 0.2, ..., последние 10% - вес 1.0.
    

# алгоритм для решения задачи
def randomized_max_weighted_clique(edges: dict, weights: list, impact_degree, impact_weight, iterations: int=10) -> list:
    """
    Функция для получения рандомизированного решения задачи о максимальной взвешенной клике.\n
    Parameters:
        * edges: словарь смежных вершинам вершин
        * weights: список с весами вершин
        * impact_degree: степень влиянися количества соседей у вершины на вероятность её выбора в клику
        * impact_weight: степень влиянися веса вершины на вероятность её выбора в клику
        * iterations: через сколько попыток без улучшения решения выходить из алгоритма\n
    Returns:
        * list: данные о найденной взвешенной клике в формате [вес, [вершина в клике 1, ..., вершина в клике k]]
    """
    num_vertices = len(weights) # == len(edges.keys()), число вершин в графе
    original_candidates = set(edges.keys()) # set вершин (изначально все являются кандидатами в клику)
    original_candidates_degrees = [len(edges[v]) for v in original_candidates] # создаём список степеней вершин (индекс - номер вершины, так как ожидается, что на входе edges остортирован в порядке увеличения номера вершины)

    probability_weights = [impact_degree*original_candidates_degrees[v] + impact_weight*weights[v]  for v in edges.keys()] # создаём список с вероятностями выбора вершины в клику в зависимости от (число смежных вершин)*(коэффициент степени вершины) + (вес вершины)*(коэффициент веса вершины)

    best_weighted_clique = [] # текущая лучшая взвешенная клика
    best_weight = 0 # вес лучшей взвешенной клики

    attempts = 0 # текущее число попыток без улучшения результата
    while attempts < iterations: # запускаем алгоритм, пока число попыток без изменения результата не превысит счётчик iterations
        weighted_clique = [] # создаём "пустую" клику
        weight = 0 # вес клики на текущей попытке

        candidates = original_candidates.copy() # копируем set всех кандидатов
        while len(candidates) != 0: # пока есть кандидаты — пытаемся добавить их в клику в зависимости от их probability_weights
            candidates_probability_weights = [probability_weights[i] for i in candidates] # обновляем вероятности попадания кандидатов в клику (оставляем вероятности только допустимых вершин) для итерациии случайного выбора
            
            v = random.choices(population=list(candidates), weights=candidates_probability_weights, k=1)[0] # случайным образом выбираем вершину в клику в соответствии с её вероятнотью (чем больше степень и вес относительно других вершин — тем выше вероятность) (переводим candidates в список для случайного выбора)
            weighted_clique.append(v) # добавляем выбранную вершину в клику
            weight += weights[v] # увеличиваем вес рассматриваемой клики

            candidates = candidates.intersection(edges[v]) # среди кандидитов оставляем только тех, кто смежен со всеми вершинами в текущей клике (итеративно этот список постоянно уменьшается с добавлением новых вершин в клику)

        if weight > best_weight: # если нашли новую лучшую взвешенную клику, то запоминаем её
            best_weighted_clique = weighted_clique.copy() # сохраняем содержимое лучшей взвешенной клики
            best_weight = weight # обновляем лучший вес
            # print(f"attempt {attempts} with new best: {best_weighted_clique}")
            attempts = 0 # обнуляем число итераций без улучшения решения
        else:
            attempts += 1 # увеличиваем число итераций без улучшения решения

    return [best_weight, best_weighted_clique] # возвращаем вес лучшей взвешенной клики и её содержимое


# запуск алгоритма
solutions = {} 
for dataset in data.keys(): # идём по тест-кейсам
    time_start = time.perf_counter() # замеряем время начала выполнения
    for i in range(runs): # делаем runs запусков для усреднения времени
        edges_complement = get_complement_edges(data[dataset]["edges"]) # переходим к дополнению графа (от независимого множества к клике) (в цикле для честного подсчёта времени)
        solution = randomized_max_weighted_clique(edges=edges_complement, weights=weights[dataset], impact_degree=impact_degree, impact_weight=impact_weight, iterations=iterations) # запускаем рандомизированный алгоритм
        # print(f"run: {i}, solution: {solution}")
    time_working = time.perf_counter() - time_start # считаем сколько времени работал алгоритм

    check_solution(edges_complement=edges_complement, weights=weights[dataset], solution=solution) # проверка решения (последнего полученного за runs запусков, оно может быть не лучшим)
    solution = transform_solution(solution) # сортирует вершины взвешенной клики в порядке возрастания их номера и возвращает нумерацию с единицы
    solutions[dataset] = {"time": time_working/runs, "weight": solution[0], "independent_set": solution[1]} # добавление ответа в словарь с ответами
    # save_solution(dataset, solutions[dataset]) # сохранение полученного ответа
    print(f"{dataset}: {solutions[dataset]}")