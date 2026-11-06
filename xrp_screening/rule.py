"""Trend-continuation rule from T12 & T18; calibrated on XRP in-sample only, then frozen."""
import numpy as np, pandas as pd, os, warnings; from scipy import stats; warnings.filterwarnings("ignore")
D=os.path.dirname(os.path.abspath(__file__))
FEE=float(os.environ.get("FEE","0.003"))  # round-trip; default = bitFlyer Lightning spot 0.15%x2 (30d vol <100k JPY)
def load(s): return pd.read_csv(f"{D}/data/{s}_1h.csv",index_col=0,parse_dates=True).iloc[:-1]
def feats(x):
    C,H,L,V=x.close,x.high,x.low,x.volume; r1=np.log(C/C.shift())
    rv24=r1.rolling(24).std(); f12=(C/C.rolling(24).mean()-1)/rv24            # = -T12
    d=C.diff(); up=d.clip(lower=0).ewm(alpha=1/14).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/14).mean()
    rsi=100-100/(1+up/(dn+1e-12)); lv=np.log(V+1); vz=((lv-lv.rolling(24).mean())/lv.rolling(24).std()).clip(-5,5)
    f18=(rsi-50)*np.exp(vz)                                                      # = -T18
    W=720  # causal 30-day rolling percentile -> coin-agnostic thresholds
    p12=f12.rolling(W).rank(pct=True); p18=f18.rolling(W).rank(pct=True)
    fh=H.rolling(24).max().shift(-24); fl=L.rolling(24).min().shift(-24)
    return pd.DataFrame(dict(p12=p12,p18=p18,A=(fh>=1.05*C).astype(float).where(fh.notna()),
                             B=(fl<=0.95*C).astype(float).where(fl.notna()),C=C,H=H,L=L))
def nw_var(u,lag=48):
    u=u-u.mean(); m=len(u); s=(u@u)/m
    for l in range(1,lag+1): s+=2*(1-l/(lag+1))*(u[l:]@u[:-l])/m
    return s/m
def ab_test(df,sig):
    sel=sig.astype(float).values; su,sd=sel*df.A.values,sel*df.B.values; mu,md=su.mean(),sd.mean()
    if mu==0 or md==0: return mu/sel.mean(),md/sel.mean(),np.nan,np.nan
    g=su/mu-sd/md; z=np.log(mu/md)/np.sqrt(nw_var(g)); return mu/sel.mean(),md/sel.mean(),z,2*stats.norm.sf(abs(z))
def trades(df,sig,tp=.05,sl=.05,hold=24):
    """non-overlapping long: entry at close of signal bar, TP/SL vs entry, same-bar both -> SL (conservative), else exit at 24h close"""
    out=[]; i=0; C,H,L,s=df.C.values,df.H.values,df.L.values,sig.values; n=len(df)
    while i<n-hold:
        if s[i]:
            e=C[i]; ret=None
            for j in range(i+1,i+hold+1):
                hs,ls=H[j]>=e*(1+tp),L[j]<=e*(1-sl)
                if ls: ret=-sl; break
                if hs: ret=tp; break
            if ret is None: ret=C[i+hold]/e-1
            out.append((df.index[i],ret-FEE)); i=j+1
        else: i+=1
    return pd.DataFrame(out,columns=["t","r"]).set_index("t").r
def evaluate(sym,rule,period=None):
    df=feats(load(sym)).dropna()
    if period is not None: df=df[period(df.index)]
    sig=rule(df); A,B,z,p=ab_test(df,sig); tr=trades(df,sig)
    wins=(tr>0).sum(); bp=stats.binomtest(wins,len(tr),0.5).pvalue if len(tr) else np.nan
    tp=stats.ttest_1samp(tr,0).pvalue if len(tr)>2 else np.nan
    return dict(sym=sym,n_hours=len(df),sig_rate=sig.mean(),A=A,B=B,AB=A/B if B else np.nan,z=z,p_AB=p,
                base_AB=df.A.mean()/df.B.mean(),n_trades=len(tr),win=wins/len(tr) if len(tr) else np.nan,p_win=bp,
                mean_ret_pct=tr.mean()*100,p_ret=tp,total_ret_pct=(np.prod(1+tr)-1)*100)
# ---- 1) calibration on XRP in-sample (first 70%) : 3 candidate thresholds only ----
xi=feats(load("XRPUSDT")).dropna().index; cut=xi[int(len(xi)*.7)]; gap=cut+pd.Timedelta(hours=24)
IS=lambda ix: ix<cut; OOS=lambda ix: ix>=gap
cands={q:(lambda q: lambda d: (d.p12>=q)&(d.p18>=q))(q) for q in (0.7,0.8,0.9)}
cal=pd.DataFrame([dict(q=q,**evaluate("XRPUSDT",r,IS)) for q,r in cands.items()])
pd.set_option("display.width",250)
print("[XRP IS calibration]"); print(cal[["q","sig_rate","A","B","AB","p_AB","n_trades","win","mean_ret_pct"]].round(3).to_string(index=False))
qbest=cal.loc[cal.AB.idxmax(),"q"]; RULE=cands[qbest]
print(f"\n=> 凍結ルール: 30日ローリング百分位で  MA24乖離/rv24 >= {qbest:.0%}  かつ  (RSI14-50)*exp(出来高z) >= {qbest:.0%}  → ロング, TP+5% / SL-5% / 24h時間切れ, 手数料0.1%")
# ---- 2) frozen-rule tests ----
res=[dict(set="XRP OOS",**evaluate("XRPUSDT",RULE,OOS))]
for s in ["SOLUSDT","ADAUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","LTCUSDT","BNBUSDT","DOTUSDT","TRXUSDT","ETHUSDT","BTCUSDT"]:
    res.append(dict(set="他通貨(全期間)",**evaluate(s,RULE)))
res=pd.DataFrame(res)
pp=res.p_AB.fillna(1).values; o=np.argsort(pp); q=pp[o]*len(pp)/np.arange(1,len(pp)+1)
res.loc[o,"q_AB"]=np.minimum(np.minimum.accumulate(q[::-1])[::-1],1)
res.to_csv(f"{D}/results_rule_fee{FEE*100:.2f}pct.csv",index=False,encoding="utf-8-sig")
print(res[["set","sym","sig_rate","A","B","AB","base_AB","p_AB","q_AB","n_trades","win","p_win","mean_ret_pct","p_ret","total_ret_pct"]].round(3).to_string(index=False))
# pooled across other coins (each coin equally; trade-level)
o=res.iloc[1:]; print("\n他通貨: A/B>1 の数 %d/%d, A/B>=2 の数 %d, p_AB<.05(A>B) の数 %d, 平均トレード収益>0 の数 %d"%(
 (o.AB>1).sum(),len(o),(o.AB>=2).sum(),((o.p_AB<.05)&(o.AB>1)).sum(),(o.mean_ret_pct>0).sum()))
print("符号検定(A/B>1 が偶然か) p=%.4f"%stats.binomtest(int((o.AB>1).sum()),len(o),0.5).pvalue)
