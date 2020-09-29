import os


def normabspath(basedir: str, filename: str):
    return os.path.normpath(os.path.join(basedir, filename))


def show_ellapsed_time(start: float, end: float) -> None:
    ellapsed_secs: int = round(end - start)

    if ellapsed_secs > 59:
        minutes: int = int(ellapsed_secs / 60)
        seconds_msg: str = ' and ' + str(ellapsed_secs % 60) + ' seconds' if ellapsed_secs % 60 != 0 else ''
        print(f"Ellapsed time: {minutes} {'minutes' if minutes > 1 else 'minute'}{seconds_msg}")
    else:
        print(f"Ellapsed time: {ellapsed_secs} seconds")
