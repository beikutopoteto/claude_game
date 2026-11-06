import sys,os,time; sys.argv=[sys.argv[0]]
from fetch import fetch
d=os.path.dirname(os.path.abspath(__file__)); end=int(time.time()*1000); start=end-3*365*24*3600_000
for s in ["SOLUSDT","ADAUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","LTCUSDT","BNBUSDT","DOTUSDT","TRXUSDT"]:
    if not os.path.exists(f"{d}/data/{s}_1h.csv"): fetch(s,start,end).to_csv(f"{d}/data/{s}_1h.csv")
