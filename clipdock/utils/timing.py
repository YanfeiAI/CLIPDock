import time
from collections import namedtuple

try:
    import resource
except ImportError:
    resource = None


TimingSnapshot = namedtuple(
    "TimingSnapshot", "wall_seconds parent_cpu_seconds children_cpu_seconds"
)


def _children_cpu_seconds():
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def take_timing_snapshot():
    return TimingSnapshot(
        time.perf_counter(),
        time.process_time(),
        _children_cpu_seconds(),
    )


def timing_delta(start):
    end = take_timing_snapshot()
    wall_seconds = end.wall_seconds - start.wall_seconds
    cpu_seconds = (
        end.parent_cpu_seconds - start.parent_cpu_seconds
        + end.children_cpu_seconds - start.children_cpu_seconds
    )
    return max(wall_seconds, 0.0), max(cpu_seconds, 0.0)


def format_timing(wall_seconds, cpu_seconds):
    return f"Wall Time Cost: {wall_seconds:.4f}s, CPU Time Cost: {cpu_seconds:.4f}s"
