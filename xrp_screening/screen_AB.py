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
import warnings; warnings.filterwarnings("ignore")
fh=H.rolling(24).max().shift(-24); fl=L.rolling(24).min().shift(-24)
up=(fh>=1.05*C).astype(float).where(fh.notna()); dn=(fl<=0.95*C).astype(float).where(fl.notna())
valid=up.notna()&dn.notna()&B.notna().all(axis=1)&Z.notna().all(axis=1)
U,Dn=up[valid],dn[valid]; ix=U.index; n=len(ix)
cut=ix[int(n*0.7)]; gap=cut+pd.Timedelta(hours=24); IS=ix<cut; OS=ix>=gap
def nw_var(u,lag=48):
    u=u-u.mean(); m=len(u); s=(u@u)/m
    for l in range(1,lag+1): s+=2*(1-l/(lag+1))*(u[l:]@u[:-l])/m
    return s/m
def test(sel,u,d):
    # H0: A=B (A/B=1) in selected bucket. Ratio estimator A/B = mean(sel*u)/mean(sel*d); delta method w/ HAC on joint series
    su,sd=(sel*u).values,(sel*d).values; mu,md=su.mean(),sd.mean()
    g=su/mu-sd/md if md>0 and mu>0 else None        # influence fn of log(A/B)
    if g is None: return np.nan,np.nan,np.nan,np.nan
    se=np.sqrt(nw_var(g)); lr=np.log(mu/md); z=lr/se
    return mu/md, z, 2*stats.norm.sf(abs(z)), (np.exp(lr-1.96*se),np.exp(lr+1.96*se))
rows=[]
bA,bB=U[OS].mean(),Dn[OS].mean()
for name,(s,why) in ALL.items():
    s=s.replace([np.inf,-np.inf],np.nan).loc[ix]
    for sg in (1,-1):                       # both tails/directions; threshold from IS only
        ss=s*sg; th=ss[IS].quantile(.9)
        for per,msk in (("IS",IS),("OOS",OS)):
            sel=(ss[msk]>=th).astype(float).fillna(0)
            A=(sel*U[msk]).sum()/sel.sum(); Bp=(sel*Dn[msk]).sum()/sel.sum()
            r,z,p,ci=test(sel,U[msk],Dn[msk])
            rows.append(dict(name=name,dir="高" if sg==1 else "低",period=per,n_sel=int(sel.sum()),A=A,B=Bp,AB=r,z=z,p=p,
                             ci_lo=ci[0] if isinstance(ci,tuple) else np.nan,ci_hi=ci[1] if isinstance(ci,tuple) else np.nan))
r=pd.DataFrame(rows)
w=r.pivot_table(index=["name","dir"],columns="period",values=["n_sel","A","B","AB","z","p","ci_lo","ci_hi"]); w.columns=[f"{a}_{b}" for a,b in w.columns]; w=w.reset_index()
w["type"]=np.where(w.name.str[0]=="T","理論","ランダム")
# OOS p BH over all 200 (feature×direction) tests, for reference
pp=w.p_OOS.fillna(1).values; o=np.argsort(pp); q=pp[o]*len(pp)/np.arange(1,len(pp)+1); w.loc[w.index[o],"q_BH_OOS"]=np.minimum(np.minimum.accumulate(q[::-1])[::-1],1)
w["pass"]=(w.AB_IS>=2)&(w.AB_OOS>=2)&(w.p_OOS<0.05)
w=w.sort_values("AB_OOS",ascending=False); w.to_csv(f"{D}/results_AB.csv",index=False,encoding="utf-8-sig")
print("OOS無条件: A=%.3f B=%.3f A/B=%.2f | IS無条件 A/B=%.2f"%(bA,bB,bA/bB,U[IS].mean()/Dn[IS].mean()))
c=["name","dir","type","n_sel_OOS","A_IS","B_IS","AB_IS","A_OOS","B_OOS","AB_OOS","ci_lo_OOS","ci_hi_OOS","z_OOS","p_OOS","q_BH_OOS"]
pd.set_option("display.width",300); pd.set_option("display.max_colwidth",55)
print("A/B>=2(OOS)候補数:",(w.AB_OOS>=2).sum()," IS&OOS>=2:",((w.AB_IS>=2)&(w.AB_OOS>=2)).sum()," +p<.05:",w["pass"].sum()," +q<.05:",(w["pass"]&(w.q_BH_OOS<.05)).sum())
print(w[w.AB_OOS>=2][c].round(3).to_string(index=False))
