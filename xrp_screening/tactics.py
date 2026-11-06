"""Low-frequency uses of the frozen trend signal (q=0.9):
 (1) Short filter: baseline RSI>=70 short; skip shorts when trend signal fired in last 6h.
 (2) Signal-timed weekly DCA vs plain weekly DCA.
Fees: bitFlyer Lightning spot 0.15%/side (FEE env = round trip)."""
import os, numpy as np, pandas as pd, warnings; from scipy import stats; warnings.filterwarnings("ignore")
from rule import feats, load
FEE=float(os.environ.get("FEE","0.003")); Q=0.9; D=os.path.dirname(os.path.abspath(__file__))
SYMS=["XRPUSDT","BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","LTCUSDT","BNBUSDT","DOTUSDT","TRXUSDT"]
def rsi(C):
    d=C.diff(); u=d.clip(lower=0).ewm(alpha=1/14).mean(); v=(-d.clip(upper=0)).ewm(alpha=1/14).mean(); return 100-100/(1+u/(v+1e-12))
def short_trades(df,entry,tp=.05,sl=.05,hold=24):
    C,H,L,s=df.C.values,df.H.values,df.L.values,entry.values; out=[]; i=0; n=len(df)
    while i<n-hold:
        if s[i]:
            e=C[i]; r=None
            for j in range(i+1,i+hold+1):
                if H[j]>=e*(1+sl): r=-sl; break        # SL first if both (conservative)
                if L[j]<=e*(1-tp): r=tp; break
            if r is None: r=1-C[i+hold]/e
            out.append(r-FEE); i=j+1
        else: i+=1
    return np.array(out)
def summ(r): return dict(n=len(r),win=(r>0).mean(),mean=r.mean()*100,total=(np.prod(1+r)-1)*100)
S,Dc=[],[]
for sym in SYMS:
    x=load(sym); df=feats(x).dropna(); df["rsi"]=rsi(x.close).loc[df.index]
    sig=(df.p12>=Q)&(df.p18>=Q); recent=sig.rolling(6,min_periods=1).max().astype(bool)
    # ---- (1) short filter ----
    base=(df.rsi>=70) if os.environ.get("BASE","daily")=="rsi" else pd.Series(df.index.hour==0,df.index)  # daily 00:00 UTC short (signal-independent)
    a=short_trades(df,base); b=short_trades(df,base&~recent); c=short_trades(df,base&recent)  # c = shorts that were skipped
    p=stats.ttest_ind(b,c,equal_var=False).pvalue if len(c)>2 else np.nan
    sa,sb,sc=summ(a),summ(b),summ(c)
    S.append(dict(sym=sym,base_n=sa["n"],base_mean=sa["mean"],base_total=sa["total"],filt_n=sb["n"],filt_mean=sb["mean"],filt_total=sb["total"],
                  skipped_n=sc["n"],skipped_mean=sc["mean"],skipped_win=sc["win"],p_kept_vs_skipped=p))
    # ---- (2) weekly DCA ----
    wk=df.index.to_period("W-SUN"); g=pd.DataFrame({"C":df.C,"sig":sig,"wk":wk})
    rows=[]
    for w,gg in g.groupby("wk"):
        if len(gg)<160: continue
        plain=gg.C.iloc[0]; hit=gg[gg.sig]; timed=(hit.C.iloc[0] if len(hit) else gg.C.iloc[-1]); rows.append((plain,timed,len(hit)>0))
    w=pd.DataFrame(rows,columns=["plain","timed","had_sig"]); d=np.log(w.timed/w.plain)
    units_ratio=(1/w.timed).sum()/(1/w.plain).sum()-1          # same JPY each week; fees identical both sides
    t=stats.ttest_1samp(d,0); 
    Dc.append(dict(sym=sym,weeks=len(w),pct_weeks_signal=w.had_sig.mean(),mean_price_diff_pct=d.mean()*100,
                   units_gain_pct=units_ratio*100,p_two_sided=t.pvalue,
                   diff_sig_weeks=d[w.had_sig].mean()*100,diff_nosig_weeks=d[~w.had_sig].mean()*100))
S,Dc=pd.DataFrame(S),pd.DataFrame(Dc)
S.to_csv(f"{D}/tactic_short_filter_{os.environ.get('BASE','daily')}.csv",index=False); Dc.to_csv(f"{D}/tactic_dca.csv",index=False)
pd.set_option("display.width",250)
print(f"FEE round-trip {FEE*100:.2f}%\n[1] ベースショート(BASE={os.environ.get('BASE','daily')}) vs トレンドシグナル(直近6h)時は見送り")
print(S.round(3).to_string(index=False))
print("改善した銘柄(平均):",(S.filt_mean>S.base_mean).sum(),"/",len(S)," 見送ったショートの平均<残したショート:",(S.skipped_mean<S.filt_mean).sum(),
      " 符号検定p=%.4f"%stats.binomtest(int((S.filt_mean>S.base_mean).sum()),len(S),.5).pvalue)
print("\n[2] 週次積立: 週初に買う vs 週内で最初のシグナル時に買う(無ければ週末)  ※price_diff<0 / units_gain>0 が有利")
print(Dc.round(3).to_string(index=False))
print("有利だった銘柄:",(Dc.units_gain_pct>0).sum(),"/",len(Dc)," 符号検定p=%.4f"%stats.binomtest(int((Dc.units_gain_pct>0).sum()),len(Dc),.5).pvalue)
