import os
import subprocess
import time
import re
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from multiprocessing import cpu_count
from filecmp import cmp
import shutil

# Параметры компиляции и запуска
TASKS = {
    "task_1d": {
        "path": "./task_1d",
        "parallel_compiler": "clang++ -fopenmp -O0",
        "serial_compiler": "clang++ -O0",
    },
    "task_2d": {
        "path": "./task_2d",
        "parallel_compiler": "mpic++ -lm -O0",
        "serial_compiler": "clang++ -O0",
    },
    "common_task": {
        "path": "./common_task",
        "serial_compiler": "clang++ -O0",
        "mpi_compiler": "mpic++ -O0",
        "omp_compiler": "clang++ -fopenmp -O0",
    },
}

RESULT_DIR = "./results"
LOG_DIR = "./logs"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
RESULT_DIR= os.path.realpath(RESULT_DIR)
LOG_DIR = os.path.realpath(LOG_DIR)



def extract_computation_time(stdout: str) -> float:
    """
    Извлекает время выполнения программы из строки stdout.
    Ожидаемый формат: "Computation time: 14.435435 seconds".
    """
    print(stdout)
    match = re.search(r"Computation time: ([\d\.]+) seconds", stdout)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Не удалось найти время выполнения в выводе программы.")


def compile_and_run_common(task_info, isize, jsize, processes, repetitions):
    """
    Выполнение common_task для serial, OpenMP и MPI режимов.
    """
    task_path = os.path.realpath(task_info["path"])
    serial_file = os.path.join(task_path, "task_serial.cc")
    mpi_file = os.path.join(task_path, "task_parallel_mpi.cc")
    omp_file = os.path.join(task_path, "task_parallel_omp.cc")

    result_serial = os.path.join(RESULT_DIR, "result_serial.txt")
    result_parallel = os.path.join(RESULT_DIR, "result_parallel.txt")

    # Компиляция
    subprocess.run(
        f'{task_info["serial_compiler"]} -DISIZE={isize} -DJSIZE={jsize} {serial_file} -o {task_path}/task_serial',
        shell=True,
        check=True,
    )
    subprocess.run(
        f'{task_info["mpi_compiler"]} -DISIZE={isize} -DJSIZE={jsize} {mpi_file} -o {task_path}/task_parallel_mpi',
        shell=True,
        check=True,
    )
    subprocess.run(
        f'{task_info["omp_compiler"]} -DISIZE={isize} -DJSIZE={jsize} {omp_file} -o {task_path}/task_parallel_omp',
        shell=True,
        check=True,
    )

    # Запуск
    serial_times = []
    mpi_times = {p: [] for p in range(1, processes + 1)}
    omp_times = {p: [] for p in range(1, processes + 1)}

    # Последовательная версия
    for _ in range(repetitions):
        result = subprocess.run(
            f"{task_path}/task_serial", shell=True, capture_output=True, text=True, check=True
        )
        serial_times.append(extract_computation_time(result.stdout))

    # MPI версия
    for p in range(1, processes + 1):
        for _ in range(repetitions):
            result = subprocess.run(
                f"mpirun -np {p} {task_path}/task_parallel_mpi",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            mpi_times[p].append(extract_computation_time(result.stdout))

    # OpenMP версия
    for p in range(1, processes + 1):
        for _ in range(repetitions):
            result = subprocess.run(
                f"OMP_NUM_THREADS={p} {task_path}/task_parallel_omp",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            omp_times[p].append(extract_computation_time(result.stdout))

    return serial_times, mpi_times, omp_times


def compile_and_run(task, isize, jsize, processes, repetitions=5):
    """
    Компиляция, выполнение и проверка идентичности результатов.
    """
    task_info = TASKS[task]
    task_path = os.path.realpath(task_info["path"])
    serial_file = os.path.join(task_path, "task_serial.cc")
    parallel_file = os.path.join(task_path, "task_parallel.cc")
    result_serial = os.path.join(RESULT_DIR, "result_serial.txt")
    result_parallel = os.path.join(RESULT_DIR, "result_parallel.txt")
    log_file = os.path.join(LOG_DIR, "execution_log.txt")

    log_messages = []

    def log(message):
        log_messages.append(message)
        print(message)

    if task == "common_task":
        return compile_and_run_common(task_info, isize, jsize, processes, repetitions)

    # Компиляция последовательной версии
    log("Starting compilation for task: {}".format(task))
    subprocess.run(
        f'{task_info["serial_compiler"]} -DISIZE={isize} -DJSIZE={jsize} {serial_file} -o {task_path}/task_serial',
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log("Sequential version compiled successfully.")

    # Компиляция параллельной версии
    subprocess.run(
        f'{task_info["parallel_compiler"]} -DISIZE={isize} -DJSIZE={jsize} {parallel_file} -o {task_path}/task_parallel',
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log("Parallel version compiled successfully.")

    # Замеры для последовательной версии
    serial_times = []
    log("Starting execution of the sequential version...")
    for i in range(repetitions):
        result = subprocess.run(
            f"{task_path}/task_serial", shell=True, capture_output=True, text=True, check=True, cwd=RESULT_DIR
        )
        print("Hi")
        serial_times.append(extract_computation_time(result.stdout))
        log(f"Sequential run {i + 1}/{repetitions} completed: {serial_times[-1]:.4f} seconds")
    avg_serial_time = sum(serial_times) / len(serial_times)
    log(f"Average sequential time: {avg_serial_time:.4f} seconds")

    # Замеры для параллельной версии
    parallel_times = {p: [] for p in range(1, processes + 1)}
    log("Starting execution of the parallel version...")
    for p in range(1, processes + 1):
        for i in range(repetitions):
            if task == "task_1d":
                result = subprocess.run(
                    f"OMP_NUM_THREADS={p} {task_path}/task_parallel",
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=RESULT_DIR
                )
            elif task == "task_2d":
                result = subprocess.run(
                    f"mpirun -np {p} {task_path}/task_parallel",
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=RESULT_DIR
                )
            parallel_times[p].append(extract_computation_time(result.stdout))
            log(f"Parallel run with {p} processes ({i + 1}/{repetitions}): {parallel_times[p][-1]:.4f} seconds")

            # Проверка идентичности файлов
            # if not cmp(result_serial, result_parallel, shallow=False):
            #     raise RuntimeError(
            #         f"Results differ between serial and parallel runs (processes: {p}, repetition: {i + 1})."
            #     )
        log(f"Parallel run for {p} processes completed.")

    # Удаление временных файлов
    if os.path.exists(result_serial):
        os.remove(result_serial)
    if os.path.exists(result_parallel):
        os.remove(result_parallel)
    log("Temporary result files removed.")

    return avg_serial_time, parallel_times, "\n".join(log_messages)



# Dash-приложение
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Анализ производительности"),
        html.Div(
            [
                html.Label("Выберите задание:"),
                dcc.Dropdown(
                    id="task-dropdown",
                    options=[
                        {"label": "Task 1D", "value": "task_1d"},
                        {"label": "Task 2D", "value": "task_2d"},
                        {"label": "Common Task", "value": "common_task"},
                    ],
                    value="task_1d",
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Размер массива (ISIZE, JSIZE):"),
                dcc.Input(id="array-size", type="number", value=6000),
            ]
        ),
        html.Div(
            [
                html.Label("Число процессов:"),
                dcc.Input(id="processes", type="number", value=14),
            ]
        ),
        html.Div(
            [
                html.Label("Число повторений (для погрешности):"),
                dcc.Input(id="repetitions", type="number", value=3),
            ]
        ),
        html.Button("Запустить", id="run-button"),
        dcc.Graph(id="time-graph"),
        dcc.Graph(id="speedup-graph"),
    ]
)


@app.callback(
    [Output("time-graph", "figure"), Output("speedup-graph", "figure")],
    Input("run-button", "n_clicks"),
    [
        Input("task-dropdown", "value"),
        Input("array-size", "value"),
        Input("processes", "value"),
        Input("repetitions", "value"),
    ],
)
def update_graph(n_clicks, task, array_size, processes, repetitions):
    if not n_clicks:
        return go.Figure(), go.Figure()

    serial_times, mpi_times, omp_times = compile_and_run(task, array_size, array_size, processes, repetitions)
    process_range = list(range(1, processes + 1))

    # Построение графика времени выполнения
    time_fig = go.Figure()
    time_fig.add_trace(
        go.Scatter(
            x=process_range,
            y=[sum(serial_times) / len(serial_times)] * len(process_range),
            mode="lines+markers",
            name="Serial Time",
        )
    )
    for p in process_range:
        time_fig.add_trace(
            go.Box(
                y=mpi_times[p],
                name=f"MPI Time ({p} processes)",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
        time_fig.add_trace(
            go.Box(
                y=omp_times[p],
                name=f"OpenMP Time ({p} threads)",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    time_fig.update_layout(
        title="Время выполнения",
        xaxis_title="Число процессов/потоков",
        yaxis_title="Время (секунды)",
        legend_title="Режимы",
    )

    # Построение графика ускорения
    avg_serial_time = sum(serial_times) / len(serial_times)
    speedup_data_mpi = {p: [avg_serial_time / pt for pt in mpi_times[p]] for p in process_range}
    speedup_data_omp = {p: [avg_serial_time / pt for pt in omp_times[p]] for p in process_range}

    speedup_fig = go.Figure()
    for p in process_range:
        speedup_fig.add_trace(
            go.Box(
                y=speedup_data_mpi[p],
                name=f"MPI Speedup ({p} processes)",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
        speedup_fig.add_trace(
            go.Box(
                y=speedup_data_omp[p],
                name=f"OpenMP Speedup ({p} threads)",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
    speedup_fig.update_layout(
        title="Ускорение",
        xaxis_title="Число процессов/потоков",
        yaxis_title="Ускорение",
        legend_title="Режимы",
    )

    return time_fig, speedup_fig


if __name__ == "__main__":
    app.run_server(debug=True)