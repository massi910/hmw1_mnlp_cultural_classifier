import logging
import threading
import time
from threading import Lock, local
from typing import Dict


class Debugger:

    def __init__(self):

        self._local = local()  # Thread-local storage for timers
        self._execution_times = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self.debug = self._logger.isEnabledFor(10)
        self._lock = Lock()  # Protect shared execution times

        self.info: Dict[str, any] = {}
        self._info_lock = Lock()

        self.counters: Dict[str, int] = {}
        self._counter_lock = Lock()

        self._file_locks = {}  # one lock per filename
        self._file_locks_guard = Lock()  # protects the dict itself


    def increment_counter(self, key):
        with self._counter_lock:
            self.counters[key] = self.counters.get(key, 0) + 1

    def get_counter(self, key):
        with self._counter_lock:
            return self.counters.get(key, 0)

    def get_all_counters(self):
        with self._counter_lock:
            return self.counters.copy()

    def set_info(self, key, value):
        with self._info_lock:
            self.info[key] = value

    def get_info(self, key):
        with self._info_lock:
            return self.info.get(key)

    def get_all_info(self):
        with self._info_lock:
            return self.info.copy()

    def _get_timers(self):
        """Ensure each thread has its own timer dictionary."""
        if not hasattr(self._local, "timers"):
            self._local.timers = {}
        return self._local.timers

    def log(self, message):
        if self.debug:
            self._logger.warning(message)

    def start_timer(self, label):
        timer_id: str = self._get_timer_id(label)
        if not self.debug:
            return
        timers = self._get_timers()
        timers[timer_id] = time.time()


    def _get_timer_id(self, label: str) -> str:
        thread_name = threading.current_thread().name
        thread_id = threading.get_ident()
        return f"{label} (Thread: {thread_name} | ID: {thread_id})"

    def stop_timer(self, label, items_num: int = 1):
        if not self.debug:
            return
        timers = self._get_timers()
        timer_id: str = self._get_timer_id(label)
        if timer_id not in timers:
            self._logger.error(f"Timer {timer_id} was not started.")
            return

        end_time = time.time()
        elapsed_time = end_time - timers.pop(timer_id)

        # Record execution time safely
        with self._lock:
            if items_num <= 1:
                self._execution_times.setdefault(timer_id, []).append(elapsed_time)
            else:
                # get the average time per item
                avg = elapsed_time / items_num
                for _ in range(items_num):
                    self._execution_times.setdefault(timer_id, []).append(avg)

    def _get_average_time(self, label):

        if not self.debug:
            return None
        with self._lock:
            if label not in self._execution_times or not self._execution_times[label]:
                self._logger.error(f"No execution times found for label: {label}.")
                return None
            return sum(self._execution_times[label]) / len(self._execution_times[label])

    def log_time_every_x_items(self, label, x=10000):
        """Log the average execution time for a label every x items, with thread identifier."""

        timer_id: str = self._get_timer_id(label)
        if not self.debug:
            return
        with self._lock:
            if timer_id not in self._execution_times or not self._execution_times[timer_id]:
                self._logger.error(f"No execution times found for label: {timer_id}.")
                return
            count = len(self._execution_times[timer_id])
            if count % x == 0:
                avg_time = sum(self._execution_times[timer_id]) / count
                self._logger.warning(
                    f"Average execution time for {timer_id} after {count} recordings: {avg_time:.4f} seconds"
                )


    def get_last_timer(self, label):

        timer_id: str = self._get_timer_id(label)

        if not self.debug:
            return None
        with self._lock:
            if timer_id not in self._execution_times or not self._execution_times[timer_id]:
                self._logger.error(f"No execution times found for label: {timer_id}.")
                return None
            return self._execution_times[timer_id][-1]

    def log_all_info(self):
        if not self.debug:
            return
        if not self._execution_times:
            self._logger.warning("No execution times recorded.")
            return
        for label, times in self._execution_times.items():
            avg_time = self._get_average_time(label)
            if avg_time is not None:
                self._logger.warning(f"Average execution time for {label}: {avg_time:.4f} seconds")

            # get total time
            total_time = sum(times)
            self._logger.warning(f"Total execution time for {label}: {total_time:.4f} seconds | Count: {len(times)}")

        for key, value in self.get_all_counters().items():
            self._logger.warning(f"Counter {key}: {value}")

        for key, value in self.get_all_info().items():
            self._logger.warning(f"Info {key}: {value}")

    def _get_file_lock(self, file_name: str) -> Lock:
        # Ensure creation is thread-safe
        with self._file_locks_guard:
            if file_name not in self._file_locks:
                self._file_locks[file_name] = Lock()
            return self._file_locks.get(file_name)

    def write_line_to_file(self,
                           content: str,
                           file_name: str,
                           write_mode: str = "a"):

        file_lock = self._get_file_lock(file_name)

        try:
            with file_lock:  # <-- only blocks writes to SAME file
                with open(file_name, write_mode) as file:
                    file.write(content)
                    file.write("\n")
        except Exception as e:
            self._logger.error(f"Failed to write to file {file_name}: {e}")

    def empty_file(self, file_name: str):
        file_lock = self._get_file_lock(file_name)

        try:
            with file_lock:  # <-- only blocks writes to SAME file
                with open(file_name, "w") as file:
                    file.truncate(0)
        except Exception as e:
            self._logger.error(f"Failed to empty file {file_name}: {e}")


    def dump_report(self):
        print("No report setup!")
