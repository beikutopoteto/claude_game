import requests, pandas as pd, time, sys, os
BASE="https://data-api.binance.vision/api/v3/klines"
COLS=["open_time","open","high","low","close","volume","close_time","qvol","trades","tb_base","tb_quote","ignore"]
def fetch(sym, start_ms, end_ms):
    out=[]; t=start_ms
    while t<end_ms:
        r=requests.get(BASE,params=dict(symbol=sym,interval="1h",startTime=t,limit=1000),timeout=30).json()
        if not r: break
        out+=r; t=r[-1][0]+3600_000
        time.sleep(0.1)
    df=pd.DataFrame(out,columns=COLS).drop(columns=["ignore","close_time"])
    for c in df.columns[1:]: df[c]=df[c].astype(float)
    df["open_time"]=pd.to_datetime(df.open_time,unit="ms",utc=True)
    return df.drop_duplicates("open_time").set_index("open_time")
if __name__=="__main__":
  end=int(time.time()*1000); start=end-3*365*24*3600_000
  d=os.path.dirname(os.path.abspath(__file__))
  for s in ["XRPUSDT","BTCUSDT","ETHUSDT"]:
      df=fetch(s,start,end); df.to_parquet(f"{d}/data/{s}_1h.parquet") if False else df.to_csv(f"{d}/data/{s}_1h.csv")
      print(s,len(df),df.index[0],df.index[-1])
