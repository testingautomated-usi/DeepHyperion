from datetime import datetime


class Timer:
    start = datetime.now()

    @staticmethod
    def get_time():
        return datetime.now()

    @staticmethod
    def get_timestamps():
        now = datetime.now()
        elapsed = now - Timer.start
        return now, elapsed

    @staticmethod
    def get_elapsed_time():
        elapsed_time = datetime.now() - Timer.start
        return elapsed_time