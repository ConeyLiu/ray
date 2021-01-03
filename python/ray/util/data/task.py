from typing import Any, Callable, Dict, List, TypeVar

import ray
from ray.util.iter import _NextValueNotReady, LocalIterator, SharedMetrics
from .reader import ReaderVar, SourceReaderVar


class Task:
    def execute(self, task_input: Any) -> Any:
        raise NotImplementedError


T = TypeVar("T", bound=Task)


class SerialTask(Task):
    def __init__(self, task_fn: Callable):
        self.task_fn = task_fn

    def execute(self, task_input: LocalIterator) -> LocalIterator:
        return self.task_fn(task_input)


class ParallelTask(Task):
    def __init__(self,
                 task_fn: Callable,
                 max_parallel: int,
                 resources: Dict):
        self.task_fn = task_fn
        self.max_parallel = max_parallel
        self.resources = resources

    def execute(self, task_input_fn: Callable) -> Callable:
        def execute_fn(task_input):
            task_input = task_input_fn(task_input)
            return self.task_fn(task_input)

        return execute_fn


class NoopParallelTask(ParallelTask):
    def __init__(self,
                 max_parallel: int,
                 resources: Dict):
        super(NoopParallelTask, self).__init__(
            lambda x: x, max_parallel, resources)

    def __call__(self, task_input):
        return task_input


class UnresolvedTask(Task):
    def __init__(self, serial_fn: Callable, parallel_fn: Callable):
        self.serial_fn = serial_fn
        self.parallel_fn = parallel_fn

    def execute(self, task_input: Any) -> Any:
        raise ValueError("UnresolvedTask does not support execute")


def task_equal(a: T, b: T):
    if type(a) != type(b):
        return False

    if isinstance(a, SerialTask):
        return True
    elif isinstance(a, ParallelTask):
        if (a.max_parallel == b.max_parallel and
           a.resources == b.resources):
            return True
    return False


class TaskSet:
    def __init__(self, tasks: List[T]):
        self._tasks = tasks
        self.is_parallel = False

        self.max_parallel = 1
        self.resources = None
        self.remote_task = None

        self.set_up()

    def set_up(self):
        assert len(self._tasks) > 0
        assert all([task_equal(self._tasks[0], t) for t in self._tasks])
        if isinstance(self._tasks[0], ParallelTask):
            task = self._tasks.pop(0)
            self.is_parallel = True
            self.max_parallel = task.max_parallel
            self.resources = task.resources or {}

            for t in self._tasks:
                task = t.execute(task)

            def remote_fn(task_input):
                if callable(task_input):
                    task_input = task_input()
                return task(task_input)

            self.remote_task = ray.remote(remote_fn).options(**self.resources)

    def execute(self, input_it: LocalIterator) -> LocalIterator:
        if self.is_parallel:
            return LocalIterator(lambda timeout: self._execute_parallel(input_it), SharedMetrics())
        else:
            return self._execute_serial(input_it)

    def _execute_serial(self, input_it: LocalIterator):
        for task in self._tasks:
            input_it = task.execute(input_it)
        return input_it

    def _execute_parallel(self, input_it: LocalIterator):
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


class ExecutionTask(Task):
    def __init__(self, task_sets: List[TaskSet]):
        self.task_sets = task_sets

    def execute(self, input_it: LocalIterator):
        for task_set in self.task_sets:
            input_it = task_set.execute(input_it)
        return input_it


class TaskQueue:
    def __init__(self,
                 reader: ReaderVar,
                 tasks: List[T]):
        self.reader = reader
        self.tasks = tasks

    def num_shards(self):
        return self.reader.num_shards()

    def with_task(self, task: T) -> "TaskQueue":
        return TaskQueue(self.reader, self.tasks + [task])

    def with_reader(self, reader: ReaderVar) -> "TaskQueue":
        return TaskQueue(reader, self.tasks)

    def _resolve_tasks(self):
        resolved_tasks = []
        for task in self.tasks:
            if isinstance(task, UnresolvedTask):
                if self.reader.max_parallel() > 1:
                    task = ParallelTask(
                        task.parallel_fn, self.reader.max_parallel(),
                        self.reader.resources())
                else:
                    task = SerialTask(task.serial_fn)
            resolved_tasks.append(task)
        return resolved_tasks

    def create_execution_task(self):
        task_sets = []
        cur = []
        for task in self._resolve_tasks():
            if not cur or task_equal(cur[0], task):
                cur.append(task)
            else:
                task_sets.append(cur)
                cur = []

        if cur:
            task_sets.append(TaskSet(cur))

        if self.reader.max_parallel() > 1:
            if (len(task_sets) == 0 or not task_sets[0].is_parallel or
                self.reader.max_parallel() != task_sets[0].max_parallel or
                    self.reader.resources() != task_sets[0].resources):
                # prepend NoopParallelTask for reading source data parallel
                noop_set = TaskSet([NoopParallelTask(
                    self.reader.max_parallel(), self.reader.resources())])
                task_sets.insert(0, noop_set)

        def create_init_fn(source_reader):
            def init_fn(timeout):
                return source_reader
            return init_fn

        execution_task = {}
        for i in range(self.num_shards()):
            source_reader = self.reader.get_shard(i)
            local_it = LocalIterator(create_init_fn(source_reader), SharedMetrics())
            execution_task[i] = (local_it, ExecutionTask(task_sets.copy()))
        return execution_task
