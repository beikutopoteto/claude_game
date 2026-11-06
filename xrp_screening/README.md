# XRP 1時間先リターン × 組合せ特徴量スクリーニング（100通り）

- データ: Binance XRPUSDT / BTCUSDT / ETHUSDT 1h足 2023-11-21〜2026-11-05（n=25,943）
- ターゲット: y = log(C[t+1]/C[t])（1時間後の価格の変化率。価格水準そのものは非定常で相関が見かけだけになるため不採用）
- 特徴量: ベース40個（リターン/ボラ/出来高/テイカー買い比率/約定数/VWAP/RSI/BTC・ETH連動 など、すべて時点tまでの情報）
  - 理論25: 文献・市場構造に基づく組合せ（screen.py の T01〜T25）
  - ランダム75: 168h ローリングz化したベースから2〜5個・四則演算をランダム生成（seed固定、除算は a/(|b|+1) で保護）
- 評価: Spearman IC、Newey-West(24h) t値、BH-FDR q値・Bonferroni（100本補正）、IS(前70%)/OOS(後30%, 2025-12-16〜)、月次IC符号一致率、OOS十分位L/S(bps)、ヒット率、コスト抜きシャープ
- 結果: results.csv（|IC|順）。再実行: `python3 fetch.py && python3 screen.py`
