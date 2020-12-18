from typing import Callable, Dict, List

import ray
from ray.util.iter import _NextValueNotReady, LocalIterator, SharedMetrics
from .reader import SourceReader


class Task:
    def __init__(self, task_fn: Callable):
        self.task_fn = task_fn

    def execute(self, task_input: LocalIterator):
        raise NotImplementedError


class SerialTask(Task):
    def __init__(self, task_fn: Callable):
        super(SerialTask, self).__init__(task_fn)

    def execute(self, task_input: LocalIterator):
        return self.task_fn(task_input)


class ParallelTask(Task):
    def __init__(self,
                 task_fn: Callable,
                 max_parallel: int,
                 resources: Dict):
        super(ParallelTask, self).__init__(task_fn)
        self.max_parallel = max_parallel
        self.resources = resources

    def execute(self, task_input_fn: Callable) -> Callable:
        def execute_fn(task_input):
            task_input = task_input_fn(task_input)
            return self.task_fn(task_input)
        return execute_fn


def task_equal(a: Task, b: Task):
    if type(a) != type(b):
        return False

    if isinstance(a, SerialTask):
        return True
    elif isinstance(a, ParallelTask):
        if (a.max_parallel == b.max_parallel and
            a.resources == b.resources):
            return True
    return False


class SerialTaskSet:
    def __init__(self, tasks: List[SerialTask]):
        self.tasks = tasks

    def execute(self, input_it: LocalIterator):
        for task in self.tasks:
            input_it = task.execute(input_it)
        return input_it


class ParallelTaskSet:
    def __init__(self,
                 tasks: List[ParallelTask],
                 max_parallel: int,
                 resources: Dict):
        task = tasks.pop(0)
        for t in tasks:
            task = t.execute(task)

        def remote_fn(task_input):
            return task(task_input)

        self.max_parallel = max_parallel
        self.resources = resources or {}
        self.remote_task = ray.remote(remote_fn).options(**self.resources)

    def execute(self, input_it: LocalIterator):
        cur = []
        for item in iter(input_it):
            if isinstance(item, _NextValueNotReady):
                yield item
            else:
                if len(cur) >= self.max_parallel:
                    finished, cur = ray.wait(cur)
                    yield from ray.get(finished)
                cur.append(self.remote_task.remote(item))
        while cur:
            finished, cur = ray.wait(cur)
            yield from ray.get(finished)


class TaskQueue:
    def __init__(self,
                 source_reader: SourceReader,
                 tasks: List[Task]):
        self.source_reader = source_reader
        self.tasks = tasks or []
        self.task_sets = []

        self.add_task(self._create_source_reader_task())

    def _create_source_reader_task(self):
        def read(task_input):
            return LocalIterator(lambda: iter(self.source_reader), SharedMetrics())
        if self.source_reader.max_parallel() > 0:
            return ParallelTask(read, self.source_reader.max_parallel(),
                                self.source_reader.resources())
        else:
            return SerialTask(read)

    def add_task(self, task: Task):
        self.tasks.append(task)

    def _set_up(self):
        self.task_sets = []
        cur = []
        for task in self.tasks:
            if not cur or task_equal(cur[0], task):
                cur.append(task)
            else:
                if isinstance(cur[0], SerialTask):
                    self.task_sets.append(SerialTaskSet(cur))
                else:
                    self.task_sets.append(ParallelTaskSet(cur))
                cur = []

        if not cur:
            return

        if isinstance(cur[0], SerialTask):
            self.task_sets.append(SerialTaskSet(cur))
        else:
            self.task_sets.append(ParallelTaskSet(cur))

    def execute(self, input_it: LocalIterator):
        self._set_up()
        for task_set in self.task_sets:
            input_it = task_set.execute(input_it)
        return input_it
