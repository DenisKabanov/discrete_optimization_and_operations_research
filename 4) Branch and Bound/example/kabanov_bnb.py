import numpy as np # для быстрой работы с массивами
import random # для рандомизированного алгоритма
import time # для подсчёта времени работы
import csv # для сохранения ответов
import pandas as pd # для вывода таблицы (v 2.0.0)
import networkx as nx # для поиска независимых множеств (сильные ограничения для BnB в задаче максимальной клики)
import os # для проверки на уже посчитанные решения
import cplex # импортирование cplex модели


# настройки (для выполнения кода может потребоваться изменение data_path и solutions_path)
runs = 1 # число запусков для усреднения времени
iterations = 10000 # сколько попыток делать для поиска хорошего начального решения
random_iters = 75 # число итерация для поиска случайной раскраски графа
recalculate = False # ключ — пересчитывать ли решения для тест-кейсов (если True — решения будут пересчитываться; False — возьмутся из сохранений, если они там есть)
data_path = "./data/" # путь до папки с входными данными
solutions_path = "./solutions/" # путь, куда будут сохраняться посчитанные решения
files = ["brock200_1", "brock200_2", "brock200_3", "brock200_4", "c-fat200-1", "c-fat200-2", "c-fat200-5", "c-fat500-1", "c-fat500-10", "c-fat500-2", "c-fat500-5", "C125.9", "gen200_p0.9_44", "gen200_p0.9_55",  "johnson8-2-4",  "johnson8-4-4", "johnson16-2-4", "hamming6-2", "hamming6-4", "hamming8-2", "hamming8-4", "keller4", "MANN_a9", "MANN_a27", "MANN_a45", "p_hat300-1", "p_hat300-2", "san200_0.7_1", "san200_0.7_2", "san200_0.9_1", "san200_0.9_2", "san200_0.9_3", "sanr200_0.7"] # все файлы, на которых должен быть протестирован код
# skipped: p_hat300-3


# вспомогательные функции, не участвующие в работе алгоритма
def check_solution(edges: dict, solution) -> list:
    """
    Функция для проверка полученного лучшего решения.\n
    Parameters:
        * edges: словарь смежных вершинам вершин, описывающий граф
        * solution: решение в формате [размер клики, [вершины в клике]]\n
    Returns:
        * list — массив с ошибками (пуст, если ошибок нет)
    """
    if solution[0] != len(solution[1]): # проверка соответствия значения целевой функции количеству вершин в клике
        print("Solution objective function does not match variables values")

    errors = [] # массив для ошибок решения

    for i in range(solution[0]-1): # идём по вершинам и проверяем, смежны ли она со всеми остальными вершинами в найденной клике (solution[0] - число вершин в найденной клике)
        for j in range(i+1, solution[0]): # идём по последующим вершинам в клике
            if solution[1][j] not in edges[solution[1][i]]: # проверяем наличие ребра между вершинами
                errors.append([solution[1][i], solution[1][j]]) # если ребра нет, а вершины идут вместе — добавляем ошибку в список ошибок
    return errors # возвращаем ошибки

def transform_solution(solution) -> list:
    """
    Функция для преобразования ответа, чтобы номера вершин шли не с 0, а с 1 и по порядку.\n
    Parameters:
        * solution: решение в формате (размер клики, [вершины в клике])\n
    Returns:
        * list: решение в формате (размер клики, [отсортированные инкрементированные вершины клики])
    """
    for i in range(len(solution[1])): # идём по числу вершин в решении
        solution[1][i] += 1 # инкрементируем номер вершины (чтобы они шли не с 0, а с 1)
    return [solution[0], sorted(solution[1])] # возвращаем решение, попутно отсортировав инкрементированные вершины

def save_solution(dataset, solution: dict) -> None:
    """
    Функция для сохранения лучших ответов.\n
    Parameters:
        * dataset: название тест-кейса
        * solution: словарь для тест-кейса с решениями в формате {"time": время на подсчёт, "clique_size": размер клики, "clique": [вершины, входящие в клику]}\n
    Returns:
        * None: сохраняет решение
    """
    with open(f"{solutions_path}{dataset}.csv", 'w', newline='') as file: # открываем файл для чистой записи
        writer = csv.writer(file) # создаём объект для записи
        writer.writerow([solution["clique_size"]]) # сохраняем размер клики (writerow — сохранение одного элемента в строку)
        writer.writerows([solution["clique"]]) # сохраняем вершины клики (writerows — сохранение итерационных данных по типу списка в строку)
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


# вспомогательные функции, участвующие в работе алгоритма
def randomized_greedy_max_clique(edges:dict, iterations=10) -> list:
    """
    Функция для получения рандомизированного решения задачи о максимальной клике.\n
    Parameters:
        * edges: словарь смежных вершинам вершин
        * iterations: через сколько попыток без улучшения решения выходить из алгоритма\n
    Returns:
        * list: данные о найденной клике в формате [размер найденной клики, [вершина в клике 1, ..., вершина в клике k]]
    """
    original_candidates = set(edges.keys()) # set вершин (изначально все являются кандидатами в клику)
    original_candidates_degrees = [len(edges[v]) for v in original_candidates] # создаём список степеней вершин (индекс - номер вершины, так как ожидается, что на входе edges остортирован в порядке увеличения номера вершины)

    attempts = 0 # текущее число попыток
    best_clique = [] # текущая лучшая клика

    while attempts < iterations: # запускаем алгоритм, пока число попыток без изменения результата не превысит счётчик iterations
        clique = [] # создаём "пустую" клику
        candidates = original_candidates.copy() # копируем всех кандидатов
        while len(candidates) != 0: # пока есть кандидаты — пытаемся добавить их в клику
            candidates_degrees = [original_candidates_degrees[i] for i in candidates] # пересчитываем степени кандидатов (оставляем степени только рассматриваемых вершин) для итерациии случайного выбора
            
            v = random.choices(population=list(candidates), weights=candidates_degrees, k=1)[0] # случайным образом выбираем вершину в клику в соответствии с её степенью (чем больше степень относительно других вершин — тем выше вероятность) (переводим candidates в список для случайного выбора)
            clique.append(v) # добавляем её в клику
            
            candidates = candidates.intersection(edges[v]) # среди кандидитов оставляем только тех, кто смежен со всеми вершинами в текущей клике (итеративно этот список постоянно уменьшается с добавлением новых вершин в клику)

        if len(clique) > len(best_clique): # если нашли новую лучшую клику, то запоминаем её
            best_clique = clique.copy()
            attempts = 0 # обнуляем число итераций без улучшения решения
        else:
            attempts += 1 # увеличиваем число итераций без улучшения решения

    return [len(best_clique), best_clique] # возвращаем размер лучшей клики и её саму


def get_independent_sets(edges: dict, random_iters: int=100, min_size_for_ind_set: int=3) -> list:
    """
    Функция для поиска независимых множеств в графе.\n
    Parameters:
        * edges: словарь смежных вершинам вершин
        * random_iters: число итераций для рандомизированного поиска независимых множеств
        * min_size_for_ind_set: минимальный размер для искомых независимых множеств (если равен 1, то сами вершины будут независимыми множествами)\n
    Returns:
        * list: список найденных независимых множеств 
    """
    vertices_num = len(edges) # число вершин в графе
    adj_matrix = np.zeros(shape=(vertices_num, vertices_num), dtype=np.int8) # создание заготовки под матрицу смежности, изначально полностью заполнена нулями

    for v1 in edges.keys(): # идём по вершинам графа
        for v2 in edges[v1]: # идём по смежным v1 вершинам
            adj_matrix[v1][v2] = 1 # отмечаем, что между вершинами есть ребро (из v2 в v1 будет добавлено при рассмотрении вершины v2 как v1)

    G = nx.from_numpy_array(adj_matrix) # создаём объект графа из матрицы смежности

    independent_sets = set() # изначально — сет для tuple-ов, где tuple — независимое множество (в конце конвертируется в список)

    strategies = [nx.coloring.strategy_largest_first, nx.coloring.strategy_independent_set, nx.coloring.strategy_connected_sequential_bfs, nx.coloring.strategy_saturation_largest_first, nx.coloring.strategy_random_sequential] # рассматриваемые стратегии поиска независимых множеств

    for strategy in strategies:
        if strategy == nx.coloring.strategy_random_sequential: # если стратегия — случайное окрашивание
            iters = random_iters # число итераций будет взято из переданных параметров
        else: # иначе — одна итерация, так как другие стратегии всегда будут выдавать один и тот же ответ
            iters = 1

        for _ in range(iters): # запускаем поиск iters раз в зависимости от стратегии 
            coloring_dict = nx.coloring.greedy_color(G, strategy=strategy) # жадно окрашиваем граф по рассматриваемой стратегии
            color2nodes = dict() # словарь для соотнесения цвета с окрашенным им вершинами
            for node, color in coloring_dict.items(): # идём по вершинам и номеру цвета
                if color not in color2nodes.keys(): # если цвета ещё нет в словаре 
                    color2nodes[color] = [] # добавляем под этот цвет пустой массив
                color2nodes[color].append(node) # добавляем в массив соответствующий цвету color вершину, в него окрашенную
            for color, colored_nodes in color2nodes.items(): # идём по цветам и вершинам, окрашенным в этот цвет
                if len(colored_nodes) >= min_size_for_ind_set: # если число вершин цвета color больше заданного порога min_size_for_ind_set
                    colored_nodes = tuple(sorted(colored_nodes)) # сортируем список вершин и конвертируем в tuple (чтобы добавить в set, где не могут дублироваться элементы)
                    independent_sets.add(colored_nodes) # добавление независимого множества (если оно уже было в independent_sets, то оно не добавиться снова)
    independent_sets = [ind_set for ind_set in independent_sets] # конвертируем set в список tuple-ов
    return independent_sets # возвращаем список с независимыми множествами


# определение класса для решения задачи
class BranchAndBound():
    def __init__(self, edges: dict, heuristic_for_init_sol: callable, check_solution: callable, vars_lb: float=0.0, vars_ub: float=1.0, target: str="max", eps: float=0.0001) -> None: # инициализация модели для решения задачи
        """
        Конструктор для модели.\n
        Parameters:
            * edges: словарь смежных вершинам вершин
            * heuristic_for_init_sol: функция, что будет использоваться для нахождения первичного решения
            * check_solution: функция для проверки правильности решения (должна возвращать ошибки решения в виде массива; если решение правильно — этот список должен быть пуст)
            * vars_lb: минимальное значение, что могут принимать все переменные
            * vars_ub: максимальное значение, что могут принимать все переменные
            * target: на что будет целевая функция ("max" или "min"), по стандарту — "max"
            * eps: с какой погрешностью считать, что переменная целая\n
        Returns:
            * None: создаёт модель
        """
        self.heuristic_for_init_sol = heuristic_for_init_sol # эвристическая функция для поиска начального решения
        self.check_solution = check_solution # функция для проверки правильности решения
        self.edges = edges # данные о изначальном графе (словарь смежных вершинам вершин)
        self.num_vars = len(edges) # число переменных в задаче (вершин в графе)
        self.eps = eps # с какой погрешностью считать, что переменная целая
        self.best_solution = [0, []] # текущее лучшее решение в формате [значение целевой функции, [вершина в клике 1, ..., вершина в клике k]]

        self.model = cplex.Cplex() # создание объекта для модели

        obj = [1.0] * self.num_vars # коэффициенты переменных для целевой функции (их количество равно числу переменных)
        lb = [vars_lb] * self.num_vars # lower bound-ы для переменных модели (их количество равно числу переменных)
        ub = [vars_ub] * self.num_vars # upper bound-ы для переменных модели (их количество равно числу переменных)
        names = [f"x{i}" for i in range(self.num_vars )] # имена для переменных модели (их количество равно числу переменных)
        types = ["C"] * self.num_vars # типы для переменных модели ("C" - непрерывная, "B" — бинарная, "I" — целочисленная) (их количество равно числу переменных)
        self.model.variables.add(obj=obj, lb=lb, ub=ub, names=names, types=types) # добавление переменных в модель

        self.model.objective.set_name("Linear Program") # название модели (опционально)
        if target == "max": # тип целевой функции (max или min)
            self.model.objective.set_sense(self.model.objective.sense.maximize) # тип целевой функции — максимизация
        else:
            self.model.objective.set_sense(self.model.objective.sense.minimize) # тип целевой функции — минимизация

        self.model.set_log_stream(None) # отключение логирования у модели
        self.model.set_error_stream(None) # отключение оповещений об ошибках у модели (они handle-ятся иначе)
        self.model.set_warning_stream(None) # отключение предупреждений у модели (они handle-ятся иначе)
        self.model.set_results_stream(None) # отключение оповещений о результатах решения у модели 


    def calc_initial_solution(self, data) -> None:
        """
        Функция для поиска начального решения.\n
        Parameters:
            * data: дополнительные параметры, что будут отправлены в функцию поиска начального решения\n
        Returns:
            * None: обновляет "лучшее" решение
        """
        self.best_solution = self.heuristic_for_init_sol(self.edges, *data) # находим начальное решение эвристикой (запоминаем размер найденной клики и её вершины)
        

    def add_constraints(self, constraints: list, senses: list, rhs: list) -> None:
        """
        Функция для добавления дополнительных ограничений (суммы переменных ? значение) в модель.\n
        Parameters:
            * constraints: список с добавляемыми в модель ограничениями в формате [[[номера/имена переменных в первом ограничении], [коэффициенты для переменных в первом ограничении]], ..., [[номера/имена переменных в n-ом ограничении], [коэффициенты для переменных в n-ом ограничении]]]
            * senses: список со знаками добавляемых ограничений, может быть "G"~">="   "L"~"<="   "E"~"==" в формате [знак ограничения 1, ..., знак ограничения n]
            * rhs: список с границами (правыми частями) для ограничений в формате [ограничение 1, ..., ограничение n]\n
        Returns:
            * None: добавляем ограничения в модель
        """
        self.model.linear_constraints.add(lin_expr=constraints, senses=senses, rhs=rhs) # добавляем ограничения (без имён) в модель


    def recursive_bnb(self) -> None:
        """
        Функция для запуска точного алгоритма Branch and Bound.\n
        Returns:
            * None: обновляет best_solution
        """
        try: # пытаемся получить решение
            self.model.solve() # считаем решение от cplex solver-а
            solution = [int(self.model.solution.get_objective_value()+0.01), self.model.solution.get_values()] # создаём solution в формате [int значение целевой функции +0.01 на случай дробного решения чуть меньше целого числа, [значение переменных модели]]
        except cplex.exceptions.CplexSolverError as error: # если получили ошибку solver-а (например — неправильно наложившиеся ограничения из-за которых задача не решается)
            return # возвращаемся в DFS на уровень выше
        
        if solution[0] <= self.best_solution[0]: # проверка на UB (перестаём рассматривать ветку, если она в любом случае не улучшит целевую функцию)
            return # возвращаемся в DFS на уровень выше

        # ищем дробную вершину с наибольшим значением, по которой будем ветвиться (ближайшая к наличию в клике)
        var_fraction = [-1, -1] # дробная переменная в формате [номер переменной, её значение], по которой будем ветвиться (если таковая будет в решении)
        for var_id, var_value in enumerate(solution[1]): # идём по переменным в ответе
            if not ((1.0 - self.eps <= var_value <= 1.0 + self.eps) or (-self.eps <= var_value <= self.eps)): # проверяем, дробная ли переменная
                if var_value > var_fraction[1]: # если переменная — дробная и её значение самое наибольшее, то будем ветвиться по ней
                    var_fraction = [var_id, var_value] # запоминаем выбранную переменную


        if var_fraction[0] != -1: # если была найдена дробная переменная
            self.model.variables.set_types(var_fraction[0], "B") # переводим переменную в бинарный тип (значения 0 или 1)
            
            self.model.linear_constraints.add(lin_expr=[[[var_fraction[0]], [1.0]]], senses=["E"], rhs=[1.0], names=[f"x{var_fraction[0]}==1"]) # добавляем ограничение для переменной ==1 (в клике)
            self.recursive_bnb() # вызываем BnB на ветку, где дробная переменная >= 1 (с запоминанием текущего ответа)
            self.model.linear_constraints.delete(f"x{var_fraction[0]}==1") # убираем добавленное ограничение по имени после того, как DFS пройдёт по его ветке

            self.model.linear_constraints.add(lin_expr=[[[var_fraction[0]], [1.0]]], senses=["E"], rhs=[0.0], names=[f"x{var_fraction[0]}==0"]) # добавляем ограничение для переменной ==0 (не в клике)
            self.recursive_bnb() # вызываем BnB на ветку, где дробная переменная <= 0 (с запоминанием текущего ответа)
            self.model.linear_constraints.delete(f"x{var_fraction[0]}==0") # убираем добавленное ограничение по имени после того, как DFS пройдёт по его ветке

            self.model.variables.set_types(var_fraction[0], "C") # переводим переменную в непрерывный тип (значения от lb до ub, что были заданы ранее)
        #===================== v1 с проверкой решения прямо в модели (если не были переданы слабые ограничения)
        else: # если дробных переменных нет — проверяем решение на корректность
            solution = [solution[0], [var_id for var_id, var_value in enumerate(solution[1]) if (1.0 - self.eps <= var_value <= 1.0 + self.eps)]] # в solution заменяем переменные на вершины, что "находятся в клике"
            # в [var_id for var_id, var_value in enumerate(solution[1]) if (1.0 - self.eps <= var_value <= 1.0 + self.eps)] пронумеровываем все вершины в решении и берём номера только тех вершин, у которых значение = 1 (состоят в клике)

            errors = self.check_solution(self.edges, solution) # получаем список ошибок решения (если решение без ошибок, то check_solution должен вернуть пустой список)
            if len(errors) != 0: # если в решении есть ошибки (если ошибок не было — список будет пустым)
                constraints = list(zip(errors, [[1.0, 1.0] for i in range(len(errors))])) # конвертируем список пар ошибочно связанных вершин xi и xj в формат [([xi, xj], [1.0, 1.0]), ...], где [1.0, 1.0] — веса переменных в ограничениях
                senses = ["L"] * len(errors) # тип ограничения "L"~"<=" для всех найденных ошибок
                rhs = [1.0] * len(errors) # значение ограничений для ошибок
                self.add_constraints(constraints=constraints, senses=senses, rhs=rhs) # добавляем ограничение "xi + xj <= 1"

                self.recursive_bnb() # вызываем BnB для пересчёта текущего решения
            elif solution[0] > self.best_solution[0]: # если значение целевой функции улучшилось (все переменные не дробные и решение корректное)
                self.best_solution = solution.copy() # сохраняем полученное лучшее решение
        #--------------------- v2 без проверки решения, если у модели имеются все необходимые данные (были переданы слабые ограничения)
        # elif solution[0] > self.best_solution[0]: # если значение целевой функции улучшилось (все переменные не дробные и решение корректное)
        #     self.best_solution = solution.copy() # сохраняем полученное лучшее решение
        #=====================
        return # выходим из рекурсии (возвращаемся на уровень выше в DFS)
        

    def get_best_solution(self, eps: float=0.0001) -> list:
        """
        Функция, возвращающая лучшее найденное решение.\n
        Parameters:
            * eps: с какой погрешностью считать, что переменная целая\n
        Returns:
            * list: решение вида [значение целевой функции, [вершина в клике 1, ..., вершина в клике k]]
        """
        for v in self.best_solution[1]: # идём по переменным (либо номерам вершин в клике)
            if isinstance(v, float): # если в ответе есть не целые числа ~ переменные (ещё не сконвертированные в номера вершин)
                self.best_solution[1] = [var_id for var_id, var_value in enumerate(self.best_solution[1]) if (1.0 - eps <= var_value <= 1.0 + eps)] # в best_solution заменяем переменные на вершины, что "находятся в клике"
                break # выходим из цикла
        return self.best_solution
    

# запуск алгоритма
solutions_cplex = {} 

for dataset in data.keys(): # идём по тест-кейсам
    if os.path.exists(f"{solutions_path}{dataset}.csv") and not recalculate: # если ответ для тест-кейса уже был посчитан ранее — загружаем его
        with open(f"{solutions_path}{dataset}.csv", 'r') as file: # открываем файл для чтения (не бинарного)
            reader = csv.reader(file) # создаём объект для чтения
            reader = list(reader) # конвертируем объект _csv.reader в список для лёгкой итерации
            solutions_cplex[dataset] = {} # создаём пустой словарь под тест-кейс
            solutions_cplex[dataset]["time"] = float(reader[2][0]) # конвертируем элемент третьей строки (время работы алгоритма) в float
            solutions_cplex[dataset]["clique_size"] = int(reader[0][0]) # конвертируем элемент первой строки (размер клики) в int
            solutions_cplex[dataset]["clique"] = [int(node) for node in reader[1]] # конвертируем вторую строку (с вершинами клики) в массив int-ов
    else: # если кейс ещё не решён — решаем
        edges = data[dataset]["edges"] # смежные вершины в тест-кейсе
        time_start = time.perf_counter() # замеряем время начала выполнения
        for i in range(runs): # делаем runs запусков для усреднения времени
            model_cplex = BranchAndBound(edges=edges, heuristic_for_init_sol=randomized_greedy_max_clique, check_solution=check_solution) # создаём BnB модель
            model_cplex.calc_initial_solution([iterations]) # запускаем поиск начального решения с передачей следующих параметров - (iterations - число запусков рандомизированного алгоритма поиска клики)
            print(f"{dataset} solution from heuristic: {model_cplex.get_best_solution()}") # вывод начального эвристического решения
            
        
            independent_sets = get_independent_sets(edges=edges, random_iters=random_iters) # считаем независимые множества в графе (без них модель будет работать медленно на слабых ограничениях)
            for i in range(len(independent_sets)): # идём по независимым множествам и конвертируем их в данные для неравенств
                independent_sets[i] = (independent_sets[i], [1.0]*len(independent_sets[i])) # ((номера переменных), [коэффициенты переменных для ограничений])
            senses = ["L"] * len(independent_sets) # тип ограничения "L"~"<=" для всех найденных независимых множеств
            rhs = [1.0] * len(independent_sets) # значение ограничений для ошибок
            model_cplex.add_constraints(constraints=independent_sets, senses=senses, rhs=rhs) # добавляем сильные ограничения в модель

            #===================== v1 с проверкой решения прямо в модели (если не были переданы слабые ограничения)
            #--------------------- v2 без проверки решения ~ передаём в модель все необходимые данные (слабые ограничения)
            # simple_restrictions = check_solution(edges=edges, solution=(len(edges.keys()), list(range(len(edges.keys()))))) # получаем все слабые ограничения для графа с помощью фиктивного решения из всех элементов этого графа
            # simple_restrictions = list(zip(simple_restrictions, [[1.0, 1.0] for i in range(len(simple_restrictions))])) # конвертируем список пар ошибочно связанных вершин xi и xj в формат [([xi, xj], [1.0, 1.0]), ...], где [1.0, 1.0] — веса переменных в ограничениях
            # senses = ["L"] * len(simple_restrictions) # тип ограничения "L"~"<=" для всех найденных ошибок
            # rhs = [1.0] * len(simple_restrictions) # значение ограничений для ошибок
            # model_cplex.add_constraints(constraints=simple_restrictions, senses=senses, rhs=rhs) # добавляем ограничение "xi + xj <= 1"
            #=====================

            model_cplex.recursive_bnb() # запускаем рекурсивный BnB алгоритм
        time_working = time.perf_counter() - time_start # считаем сколько времени работал алгоритм

        solution = model_cplex.get_best_solution() # получаем точный ответ алгоритма (на ошибки он уже проверен в ходе выполнения BnB)
        if len(check_solution(edges=edges, solution=solution)) != 0: # проверка на то, что в решении не нашлось ошибок (check_solution возвращает список с ошибками, если их нет — список пустой)
            raise Exception("Solution is incorrect") # выбрасываем исключение
        solution = transform_solution(solution) # сортирует вершины клики в порядке возрастания их номера и возвращает нумерацию с единицы
        solutions_cplex[dataset] = {"time": time_working/runs, "clique_size": solution[0], "clique": solution[1]} # добавление ответа в словарь с ответами
        save_solution(dataset=dataset, solution=solutions_cplex[dataset]) # сохраняем решение для рассматриваемого кейса
    print(f"{dataset}: {solutions_cplex[dataset]}")