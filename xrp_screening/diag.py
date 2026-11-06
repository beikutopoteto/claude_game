import warnings; warnings.filterwarnings("ignore")
import runpy, numpy as np, pandas as pd, io, contextlib
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path("screen.py")
ALL,base,cut=g["ALL"],g["base"],g["cut"]
for k in ["T02","T05","T18","T12","R52","T25"]:
    name=[n for n in ALL if n.startswith(k)][0]; s=ALL[name][0].replace([np.inf,-np.inf],np.nan).loc[base.index]
    o=base.index>=cut; m=s.notna()&o; ss,yy=s[m],base.y[m]
    q=pd.qcut(ss.rank(method="first"),10,labels=False)
    prof=(yy.groupby(q).mean()*1e4).round(2).tolist(); med=(yy.groupby(q).median()*1e4).round(2).tolist()
    yw=yy.clip(yy.quantile(.01),yy.quantile(.99))
    print(k,"Pearson OOS=%.4f  winsorPearson=%.4f"%(np.corrcoef(ss,yy)[0,1],np.corrcoef(ss.clip(ss.quantile(.01),ss.quantile(.99)),yw)[0,1]))
    print("   decile mean bps  ",prof); print("   decile median bps",med)
