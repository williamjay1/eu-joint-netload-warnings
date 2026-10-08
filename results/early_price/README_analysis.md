# 早期目标与价格竞争解释：实际运行结果

完成日期2026-10-08。新增分析是已查看2024–2025结果后的大修再分析，不是新封存测试。脚本为`scripts/early_price_analysis.py`；训练、派生特征、预测与结果均在D盘，不修改既有原始/公开稿。以下点估计与不确定性均来自实际执行，未使用模拟替代观察数据。

## 主地区：目标对齐检查

目标early_hour=真实联合事件小时，且位于按shared_eval统一规则形成的过程首次事件起点后六小时内；间隙允许一完整无事件日。该标签只用于拟合和评分，任何特征与审核分配均不接触真实起点/过程编号/真实滞后结果。

共同特征为已发布M0风险及logit、同次UTC日内未来/之前六小时风险均值和最大值、整日风险均值/最大值、上升差、预测共同天气均值/spread、季节/小时周期及weekend。未来目标预测只取同一已发布24小时向量，绝不读取翌日尚未发布的预测。相同22列特征分别拟合early_hour与hour目标的L2 logistic，以隔离目标差异。

2022训练，2023以Brier优先、logloss次序、较小C破平局选择C∈{0.1,1,10}，两种主地区模型均选0.1。最终成熟定义严格从区间终点计算：主区hour_end+7天≤issue，丹麦hour_end+15天≤issue。2023最初预测仅使用2022-12-24 12UTC以前**结束**的主区标签，共8508训练小时。选择C也执行成熟约束：验证标签hour_end≤2023-12-31 12UTC−7天（主区）或−15天（丹麦）。主区2023全年支持8472小时，成熟C验证子集8292小时。准确选择分数见regularization_validation.csv，early验证Brier约0.010942/logloss约0.049659；hour约0.038954/0.147897；**两个分数对应不同目标，不能相互当作模型优劣比较。** 2023全年资源计数只使用2022拟合管线产生的已发布预测，因此可使用末期预测而不读取其未成熟标签。2024–2025每季重新拟合，C固定；2024Q1主区拟合16980小时/159个early阳性，随后允许使用已成熟的过去测试季度标签。quarterly_refits.csv给出每次真实训练样本、区间终点与成熟切点。七/十五天规则是标签成熟假设，不能证明供应商所有历史修订版本可恢复。

原始支持：2022有25过程/61早期事件小时，2023有28过程/98早期事件小时；2024有39过程/132早期事件小时，2025有32过程/91早期事件小时。全年early prior分别0.007021、0.011568、0.015152、0.010474。全测试17400小时/2105一般事件小时/223 early阳性/71过程。未知边界保留原共同定义并在early_target_support.csv注明。

| 概率/规则 | 审核N | 一般事件命中H | 初期过程C/71 |
|---|---:|---:|---:|
| 原M0 Top2 | 1450 | 341 | 35 |
| 同特征hour-logit Top2 | 1450 | 349 | 38 |
| 同特征early-logit Top2 | 1450 | 340 | 43 |
| 原M0 2023 validation count gate=.23（主合同） | 642 | 324 | 32 |
| hour-logit 2023 validation count gate=.05（主合同） | 823 | 341 | 37 |
| early-logit 2023 validation count gate=.02（主合同） | 735 | 324 | 41 |
| 原M0 2022 count gate=.25 | 618 | 322 | 32 |
| hour-logit 2022 count gate=.07 | 738 | 335 | 36 |
| early-logit 2022 count gate=.03 | 631 | 313 | 40 |
| early-logit硬套gate=.25 | 75 | 59 | 10 |

主要count gate以共同GRID={0,.01,.02,…,.95,.99,1}，在2023验证预测上选择满足平均≤.5审核/预测完整日的最大审核数阈值，不使用测试标签/工作量。新classifier只在2022拟合，C在2023选择，故这些概率在参数拟合样本外而仍是调参验证资料；没有称2023为独立测试。2023有357预测完整日，资源目标floor(.5×357)=178，原M0/early/hour实际计数175/175/173。测试分别642/725=.8855、735/725=1.0138、823/725=1.1352，均漂移，不能称保证.5/day。2022 count合同保留为次要；新模型2022概率为训练内预测分布，绝不称2022真实在线留出审核。

模型early_target测试Brier约0.012313/logloss约0.052709；一般hour目标测试Brier为early模型约0.109771、hour模型约0.062902、原M0 0.0601906。准确值以更新后的probability_scores.csv为准。early模型牺牲一般hour概率任务；它是目标对齐的选择实验，不能替代原M0的一般联合事件预测。early_calibration.csv按固定概率区间报告2024/2025预测均值和真实频率，不声称条件校准已经完美。逐季Brier、区间、logloss/prior及成对分数差在quarterly_probability_scores.csv/quarterly_brier_contrasts.csv。

### 必须报告的paired不确定性和年度差异

主要配对比较及季度分数使用2000次配对14天moving blocks，按year-quarter分层，共同保存的概率流与选择名单保持固定。一般cost_curves.csv曲线仍为1000次，replicates字段明确区分。**这些区间条件于已经拟合的滚动预测，不含重新拟合/选C的不确定性。** C用原共同过程起点所在日计数，每过程一次；不是逐小时IID。

| 对比 | ΔN [95%] | ΔH [95%] | ΔC [95%] |
|---|---|---|---|
| early 2023 count vs原2023 count（主） | +93 [54,130] | 0 [−25,17] | +9 [−1,18] |
| early 2023 count vs同特征hour 2023 count（主） | −88 [−113.025,−70.975] | −17 [−39,−5] | +4 [−4,13] |
| early 2023 count vs原.25 gate | +117 [76,150] | +2 [−24,20] | +9 [−1,18] |
| early 2022 count vs原.25 gate（次要） | +13 [−21,42.025] | −9 [−37,10.025] | +8 [−2,17] |
| early Top2 vs同特征hour Top2 | 0 [0,0] | −9 [−31,4] | +5 [−3,13] |
| early Top2 vs原M0 Top2 | 0 [0,0] | −1 [−25,15] | +8 [−1,17] |

2024主要2023-gate early覆盖25/39（397审核），原2023 gate15/39（366审核）、同特征hour19/39（437审核）；2025分别16/32（338审核）、17/32（276审核）、18/32（386审核）。旧2022-gate early在2024为24/39、2025为16/32。Top2 early为2024 27/39、2025 16/32；hour为20/39、18/32。收益集中2024，**不支持稳定普遍优越的结论**。它支持目标对齐值得单独评价，以及固定计数预算/目标漂移会改变筛选价值。

归一化过程效用U=C−rN。主要2023-gate early对原2023 gate差=9−93r，对同特征hour gate差=4+88r。后者r=.05时差8.4，pointwise95%[.75,17]；前者同r差4.35，[−5.95,13.45]。不能事后挑r=.05作为预先指定主阈值；完整声明成本网格、pointwise及gridband均在primary_paired_cost_contrasts.csv。r是审核成本/一次早期过程价值比例，不是真实欧元或工时。次数相同的Top2初期差区间仍跨零。

## 丹麦：价格竞争解释

DK1/DK2不是独立重复，因为两个组合均包含DE/FR。原始价格2022–2025全部EUR/MWh，2025-10之后使用四个完整15分钟区间均值，与Elspot小时价接续。首次/最终历史发布版本不存在：总体市场公布规则不证明每个历史日Energinet API在12UTC就可访问，故全部价格比较为**assumed-availability retrospective benchmark**。

进一步约束当地拍卖日：在issue的Europe/Copenhagen当地日期+1之前的交付价格才暴露。UTC目标日晚间1–2小时属于当地后天、需要下一次尚未发生的拍卖，设缺价标记，以训练内median/缺失指示填补；日均/极值价格只使用可暴露小时。每区2024–2025共1151小时因此不暴露实际价格（是否完整评分另行按共同规则）；四年共2315小时。raw-price Top2只排这些可暴露小时，原值排序保持严格单调。价格排序分数不是概率，不作概率评分或将其数值用作成本概率gate。

price/calendar、base-p/logit/calendar、combined-price/base/calendar使用共同日历项，base与combined的风险控制相同。两种目标都以2022训练、2023选C，2024–2025季度严格hour_end+15天成熟标签重拟合；没有用测试效果决定C。旧七天运行已被本批十五天结果替代。

| 组合及Top2 | N | H | C/全部过程 |
|---|---:|---:|---:|
| DK1 原M0 | 1450 | 290 | 35/71 |
| DK1 raw价格排名 | 1450 | 206 | 24/71 |
| DK1 base hour-logit | 1450 | 292 | 35/71 |
| DK1 combined hour-logit | 1450 | 316 | 40/71 |
| DK2 原M0 | 1450 | 321 | 40/75 |
| DK2 raw价格排名 | 1450 | 217 | 22/75 |
| DK2 base hour-logit | 1450 | 321 | 41/75 |
| DK2 combined hour-logit | 1450 | 341 | 48/75 |

combined hour vs同控制base hour的初期过程差：DK1+5[−1,12]、DK2+7[2,13]，条件配对区间。多目标/区域为补充分析，不能把单个未校正正区间当作普遍独立证实。价格单独不能替代风险模型；与风险联合有条件增量，仍受假定可见性限制。

丹麦特有事件pivotal逐小时标签另外评价，不再拟合新目标。原始过程初期含pivotal机会45/54；Top2原M0嵌套覆盖22/45、29/54，combined hour为25/45、35/54。pivotal事件自行重聚类后过程63/76，原M0早期覆盖26/63、36/76，combined hour30/63、40/76。两套不同起点/分母必须分别命名，全部审核成本仍按完整名单计入。pivotal_policy_metrics.csv不把晚期小时反推为原始过程初期机会。

## 交付输出及核验

- `policy_metrics.csv`、`annual_policy_metrics.csv`：真实N/H/C/分母，主图可直接读。
- `regularization_validation.csv`、`quarterly_refits.csv`、`early_target_support.csv`：选择过程、训练/验证阳性支持、真实训练区间终点/成熟切点、每年支持数量。
- `probability_scores.csv`、`early_calibration.csv`、`quarterly_probability_scores.csv`、`quarterly_brier_contrasts.csv`：明确区分hour与early目标，给出季度评分与配对区间。
- `primary_paired_contrasts.csv`、`primary_paired_cost_contrasts.csv`：上述主要配对比较及成本全网格；`paired_cost_contrasts.csv`为所有策略对原Top2的补充。
- `cost_curves.csv`、`cost_gate_policies.csv`：固定规则效用和共同成本gate规则；后一文件是N/H/C点估计，不声称含模型重拟合区间。
- `prediction_*.parquet`、`development_prediction_*.parquet`、`review_masks_*.parquet`、`targets_*.parquet`：逐小时测试/2022–2023开发概率、issue、实际标签、选择与过程编号；开发概率文件标识training_distribution_prediction。
- `features_*.parquet`、`price_exposure_*.parquet`、`raw_price_*.parquet`：可追溯输入及价格可见性限制。
- `analysis_manifest.json`、`verification_checks.json`：实际运行范围和检查。

已核验17条测试概率流落在[0,1]、索引唯一有序、119套选择每日不超过2、全部112个季度拟合interval_end≤相应7/15天成熟cutoff、24小时分组只有共同且正确issue、不可见当地后天价格均为空、原M0的1450/341/35和618/322/32与既有共同输出一致。CI固定概率/名单并不意味着已验证历史数据版本或真实人工审核收益。
