


def time2number(time: str):
    h, m, s = [int(t) for t in time.split(':')]
    return s + 60*m + 3600*h
