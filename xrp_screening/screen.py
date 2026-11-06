"""XRP 1h-ahead feature-combination screening (25 theory-based + 75 random combos)."""
import numpy as np, pandas as pd, os, json
from scipy import stats
D=os.path.dirname(os.path.abspath(__file__))
rd=lambda s: pd.read_csv(f"{D}/data/{s}_1h.csv",index_col=0,parse_dates=True).iloc[:-1]  # drop in-progress bar
x,b,e=rd("XRPUSDT"),rd("BTCUSDT"),rd("ETHUSDT")
idx=x.index.intersection(b.index).intersection(e.index); x,b,e=x.loc[idx],b.loc[idx],e.loc[idx]
EPS=1e-12
C,O,H,L,V,QV=x.close,x.open,x.high,x.low,x.volume,x.qvol
lr=lambda s,n: np.log(s/s.shift(n))
r1=lr(C,1)
# ---------- base features (all use info up to close of bar t) ----------
B={}
for n in [1,3,6,12,24,72]: B[f"r{n}"]=lr(C,n)
for n in [6,24,168]: B[f"rv{n}"]=r1.rolling(n).std()
B["rng"]=np.log(H/L)
B["park24"]=np.sqrt((B["rng"]**2/(4*np.log(2))).rolling(24).mean())
B["body"]=np.log(C/O)
B["uwick"]=(H-np.maximum(O,C))/C
B["lwick"]=(np.minimum(O,C)-L)/C
B["clv"]=((2*C-H-L)/(H-L+EPS))
lv=np.log(V+1); B["vz"]=(lv-lv.rolling(24).mean())/lv.rolling(24).std()
B["lqv"]=np.log(QV+1)
B["tbr"]=x.tb_base/(V+EPS)-0.5
B["tbr6"]=B["tbr"].rolling(6).mean()
lt=np.log(x.trades+1); B["trd"]=lt-lt.rolling(24).mean()
la=np.log(V/(x.trades+1)+EPS); B["ats"]=la-la.rolling(24).mean()
vwap=QV.rolling(24).sum()/V.rolling(24).sum(); B["vwap_dev"]=C/vwap-1
B["ma24"]=C/C.rolling(24).mean()-1; B["ma168"]=C/C.rolling(168).mean()-1
d=C.diff(); up=d.clip(lower=0).ewm(alpha=1/14).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/14).mean()
B["rsi14"]=100-100/(1+up/(dn+EPS))
B["pos24"]=(C-L.rolling(24).min())/(H.rolling(24).max()-L.rolling(24).min()+EPS)-0.5
B["amihud"]=r1.abs()/(QV+EPS)*1e6
B["er24"]=lr(C,24)/(r1.abs().rolling(24).sum()+EPS)
br1=lr(b.close,1); er1=lr(e.close,1)
B["btc_r1"]=br1; B["btc_r6"]=lr(b.close,6); B["btc_r24"]=lr(b.close,24); B["eth_r1"]=er1
B["btc_rv24"]=br1.rolling(24).std()
B["corr168"]=r1.rolling(168).corr(br1)
beta=(r1.rolling(168).cov(br1)/br1.rolling(168).var()); B["beta168"]=beta
B["resid1"]=r1-beta*br1
B["rel24"]=lr(C,24)-B["btc_r24"]
B["btc_tbr"]=b.tb_base/(b.volume+EPS)-0.5
hr=idx.hour; B["hsin"]=pd.Series(np.sin(2*np.pi*hr/24),idx); B["hcos"]=pd.Series(np.cos(2*np.pi*hr/24),idx)
B=pd.DataFrame(B)
# rolling (causal) z-score for random combos
Z=((B-B.rolling(168,min_periods=168).mean())/B.rolling(168,min_periods=168).std()).clip(-5,5)
Z[["hsin","hcos"]]=B[["hsin","hcos"]]
y=np.log(C.shift(-1)/C).rename("y")    # target: 1h-ahead log return
# ---------- 25 theory-based combinations ----------
f=B; z=Z
T={
"T01 リスク調整24hモメンタム: r24/rv24":                    (f.r24/f.rv24, "時系列モメンタム(Moskowitz+2012)をボラで正規化"),
"T02 ボラ調整短期リバーサル: -r1/rv24":                      (-f.r1/f.rv24, "流動性供給/過剰反応の1h反転(Jegadeesh 1990系)"),
"T03 出来高確認モメンタム: r1*vz":                            (f.r1*f.vz, "Campbell-Grossman-Wang(1993):高出来高の変化は反転しにくい"),
"T04 符号付き出来高フロー: tbr*exp(vz)":                      (f.tbr*np.exp(f.vz.clip(-5,5)), "オーダーフロー不均衡×出来高(Kyle 1985の情報トレード)"),
"T05 非流動性×リターン: r1*amihud":                           (f.r1*f.amihud, "Amihud(2002):薄い板での価格変動は反転しやすい"),
"T06 フロー価格乖離: z(tbr6)-z(r6)":                         (z.tbr6-z.r6, "買いフローが価格に未反映な分の遅行反映"),
"T07 BTCリードラグ: btc_r1-r1":                              (f.btc_r1-f.r1, "大型資産→アルトへの情報伝播の遅れ(リードラグ)"),
"T08 ベータ調整キャッチアップ: beta*btc_r1-r1":              (f.beta168*f.btc_r1-f.r1, "市場ベータ相当の未反映分"),
"T09 市場ファクター追随: (btc_r1+eth_r1)/2-r1":              ((f.btc_r1+f.eth_r1)/2-f.r1, "暗号資産市場ファクターへの収束"),
"T10 対BTC相対強度(ボラ調整): rel24/rv24":                   (f.rel24/f.rv24, "クロスセクション・相対モメンタム"),
"T11 VWAP回帰: -vwap_dev/rv24":                              (-f.vwap_dev/f.rv24, "VWAPからの乖離は執行アルゴで回帰"),
"T12 24hMA乖離リバージョン: -ma24/rv24":                     (-f.ma24/f.rv24, "短期平均回帰(ボリンジャー型)"),
"T13 長期トレンド強度: ma168/rv168":                         (f.ma168/f.rv168, "1週間トレンドの持続"),
"T14 チャイキン型蓄積: clv*vz":                              (f.clv*f.vz, "Accumulation/Distribution:高出来高時の引け位置"),
"T15 ヒゲ非対称性: (lwick-uwick)/rng":                       ((f.lwick-f.uwick)/(f.rng+EPS), "価格拒否(下ヒゲ=買い支え)"),
"T16 ボラ・レジーム×モメンタム: r1*rv6/rv24":                (f.r1*f.rv6/f.rv24, "ボラ上昇局面で短期モメンタムが変化"),
"T17 大レンジ足の反転: -r1*rng/rv24":                        (-f.r1*f.rng/f.rv24, "ノイズ大の足はオーバーシュート→反転"),
"T18 RSI逆張り×出来高: (50-rsi14)*exp(vz)":                  ((50-f.rsi14)*np.exp(f.vz.clip(-5,5)), "売られ過ぎ+出来高=投げ売りの反発"),
"T19 取引頻度×不均衡: tbr*trd":                              (f.tbr*f.trd, "情報トレーダーは約定数急増時に到来(Easley-O'Hara)"),
"T20 平均約定サイズ×不均衡: tbr*ats":                        (f.tbr*f.ats, "大口(クジラ)フローの方向"),
"T21 レンジ内位置×出来高: pos24*vz":                         (f.pos24*f.vz, "ブレイクアウトの出来高確認"),
"T22 効率比モメンタム: er24*rv24/rv168":                     (f.er24*f.rv24/f.rv168, "Kaufman効率比:一方向トレンドの質×ボラ水準"),
"T23 BTCボラ波及キャッチアップ: (btc_r1-r1)*btc_rv24/rv24":  ((f.btc_r1-f.r1)*f.btc_rv24/f.rv24, "BTC相対ボラ高時の情報伝播"),
"T24 相関加重キャッチアップ: (btc_r6-r6)*corr168":           ((f.btc_r6-f.r6)*f.corr168, "相関が高いほど乖離は収束"),
"T25 マルチ期間モメンタム-短期反転: r3/rv6+r24/rv24+r72/rv168-r1/rv6": (f.r3/f.rv6+f.r24/f.rv24+f.r72/f.rv168-f.r1/f.rv6, "期間分散モメンタム+短期反転(5特徴量)"),
}
# ---------- 75 random combinations (z-scored bases, 2-5 features, + - * /) ----------
rng_=np.random.default_rng(20261106); names=[c for c in Z.columns]
OPS=["+","-","*","/"]
def apply(a,op,c):
    if op=="+": return a+c
    if op=="-": return a-c
    if op=="*": return a*c
    return a/(c.abs()+1)        # protected division (sign of numerator kept)
R={}; seen=set()
while len(R)<75:
    k=int(rng_.integers(2,6)); fs=list(rng_.choice(names,k,replace=False)); ops=list(rng_.choice(OPS,k-1))
    key=tuple(fs)+tuple(ops)
    if key in seen: continue
    seen.add(key)
    s=Z[fs[0]]; expr=fs[0]
    for op,fn in zip(ops,fs[1:]): s=apply(s,op,Z[fn]); expr=f"({expr}{op}{fn})"
    R[f"R{len(R)+1:02d} {expr}"]=(s,f"ランダム{k}特徴量")
ALL={**T,**R}
# ---------- evaluation ----------
def nw_t(u,lag=24):
    u=u-u.mean(); n=len(u); g0=(u@u)/n; s=g0
    for l in range(1,lag+1): s+=2*(1-l/(lag+1))*(u[l:]@u[:-l])/n
    return np.sqrt(s/n)
valid=y.notna()&B.notna().all(axis=1)&Z.notna().all(axis=1)
base=pd.DataFrame({"y":y}).loc[valid]
n=len(base); cut=base.index[int(n*0.7)]
rows=[]
for name,(s,why) in ALL.items():
    s=s.replace([np.inf,-np.inf],np.nan).loc[base.index]
    m=s.notna(); yy=base.y[m]; ss=s[m]
    ry=stats.rankdata(yy); rs=stats.rankdata(ss)
    ry=(ry-ry.mean())/ry.std(); rs=(rs-rs.mean())/rs.std()
    prod=ry*rs; ic=prod.mean(); se=nw_t(prod); t=ic/se; p=2*stats.norm.sf(abs(t))
    IS=ss.index<cut
    ic_is=stats.spearmanr(ss[IS],yy[IS])[0]; ic_os=stats.spearmanr(ss[~IS],yy[~IS])[0]
    mon=pd.DataFrame({"s":ss,"y":yy}).groupby(ss.index.to_period("M")).apply(lambda g: stats.spearmanr(g.s,g.y)[0])
    sg=np.sign(ic_is) if ic_is!=0 else 1
    # OOS: quintile long-short (direction fixed from IS) & hit-rate
    so,yo=ss[~IS]*sg,yy[~IS]
    q=pd.qcut(so.rank(method="first"),5,labels=False)
    ls=(yo[q==4].mean()-yo[q==0].mean())*1e4
    hit=(np.sign(so-so.median())==np.sign(yo)).mean()
    # OOS simple strategy: position=sign(IS-oriented z), cost ignored
    pos=np.sign(so-so.median()); strat=pos*yo
    sharpe=strat.mean()/strat.std()*np.sqrt(24*365)
    rows.append(dict(id=name.split()[0],name=name,type="理論" if name[0]=="T" else "ランダム",rationale=why,
        IC_full=ic,t_NW=t,p=p,IC_IS=ic_is,IC_OOS=ic_os,sign_stable=np.sign(ic_is)==np.sign(ic_os),
        ICIR_monthly=mon.mean()/mon.std(),pct_months_same_sign=(np.sign(mon)==np.sign(ic)).mean(),
        OOS_LS_bps=ls,OOS_hit=hit,OOS_sharpe_nocost=sharpe,turnover=(pos.diff().abs()/2).mean(),n=len(ss)))
res=pd.DataFrame(rows)
# BH-FDR over the 100 tests
pp=res.p.values; o=np.argsort(pp); q=pp[o]*len(pp)/np.arange(1,len(pp)+1); q=np.minimum.accumulate(q[::-1])[::-1]
res.loc[o,"q_BH"]=np.minimum(q,1); res["p_bonf"]=np.minimum(res.p*len(res),1)
res=res.sort_values("IC_full",key=abs,ascending=False)
res.to_csv(f"{D}/results.csv",index=False,encoding="utf-8-sig")
meta=dict(start=str(base.index[0]),end=str(base.index[-1]),n=n,split=str(cut),
          y_mean_bps=base.y.mean()*1e4,y_std_bps=base.y.std()*1e4)
json.dump(meta,open(f"{D}/meta.json","w"),ensure_ascii=False,indent=1)
print(meta)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
print(res[["name","IC_full","t_NW","q_BH","IC_IS","IC_OOS","sign_stable","pct_months_same_sign","OOS_LS_bps","OOS_hit","OOS_sharpe_nocost","turnover"]].head(30).round(4).to_string(index=False))
print("\n理論25:"); print(res[res.type=="理論"][["name","IC_full","t_NW","q_BH","IC_IS","IC_OOS","OOS_LS_bps","OOS_sharpe_nocost"]].round(4).to_string(index=False))
print("\nsummary by type:"); print(res.groupby("type").apply(lambda g: pd.Series(dict(mean_absIC=g.IC_full.abs().mean(),n_q05=(g.q_BH<.05).sum(),n_bonf05=(g.p_bonf<.05).sum(),n_sign_stable=g.sign_stable.sum(),med_absIC_OOS=g.IC_OOS.abs().median()))))
