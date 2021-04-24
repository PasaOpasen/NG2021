

from datetime import datetime as dt


def time2number(time: str):
    h, m, s = [int(t) for t in time.split(':')]
    return s + 60*m + 3600*h


def time2number_iso(time:str):
    return (dt.fromisoformat(time)-dt(1970,1,1)).total_seconds()