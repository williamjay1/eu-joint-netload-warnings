# ASMBI大修输入与实现可行性（只读核查，2026-10-08）

本次未训练、未下载、未修改F归档或既有公开仓库。`input_map.json`记录16个实际读取文件的字段、行数、时间范围、SHA256与口径核查；下文给出能够启动分析的材料与边界。

## 1. 可以马上进行的策略分析

公开可复现输入目录：`D:\MLWork\eu_joint_netload_EYENG7621_20261006\repository\data\hourly`。

`DE_LU_FR_BE.parquet`、`DK1.parquet`、`DK2.parquet`各35,064行，完整UTC小时索引为2022-01-01 00至2025-12-31 23。字段含`issue_time`、`period`、`M0_q90`至`M3_q90`及q95、`y_q90/q95`、`policy_p_q90/q95`、`policy_event_q90/q95`、P0/P1/P2选择标记、gap0/1/2过程编号。原区还含独立与直接逻辑回归概率；DK含独立概率。

M0=原边际Gaussian；M1=额外条件边际校准+同Gaussian；M2=原边际+同R选中t；M3=额外条件边际校准+同R选中t。四概率流在每一个区域组合内非缺失掩码完全相同。原区完整24h共同评分日2022/23/24/25为362/353/363/362；DK两个组合均362/356/363/362。测试期共同域均725日、17,400小时。q90测试事件小时原区2105、DK1=1641、DK2=1826；q95为1001/785/859。

`policy_event`为不完整观察日全日屏蔽后的标签，不能以逐小时`y`非缺失替代；两者不同位置原区483、DK414小时。分配必须先仅由预测完整性生成完整路径，再在完整标签域上评分。原区`policy_p`与M0最多2e-12差异来自原CDF裁剪，DK完全一致。

规则入口：`repository\scripts\upgrade_warning_policies_20261002.py::policy/process_masks/summarise`；原过程实现`repository\scripts\evaluation.py::event_processes`；复现入口`repository\analysis\reproduce_hourly_evaluation.py::run`。Top1/Top2、成本阈值、固定gate、滚动预测分位数gate、共同阈值冷却与风险上升消融、预测模型×策略交叉、零审核、成本曲线、年度季度工作量、审核类别分解均可使用这些已有概率流，无需重训。

## 2. 原规则参数、并列与过程边界

实际冻结参数文件：`F:\AcademicData\eu_joint_netload\archive\eu_joint_netload_20261002\config\upgrade_warning_policies_20261002.json`。2022设平均每日0.5名额目标、概率阈值网格0/.01–.95/.99/1；P1=.25；风险上升冷却P2=.19+6小时。6/12/24小时候选在2023按标准化早期覆盖选择6小时；每日最大2名额，允许零名额。当前规则的dev选择并非统一效用成本的事前合同。

候选需`p>=gate`。排序为风险上升priority降序、p降序、UTC时间升序；普通排序的priority就是p。风险上升`p_t*(1-max(previous six issued hourly risks))`，前日仍用已发布预测；没有真实事件状态输入。冷却拒绝与同日已选择或前日仍在状态中的目标时点距离严格小于6小时的候选，正好6小时允许。名单日24小时预测不完整则不分配；真实标签缺失不能重置冷却。全路径从2022开始，再截2024–25评分。

过程按UTC完整观察日形成：当日至少一个事件则活跃，默认最多桥接1个完整非活跃日，缺失/不完整观察日断开。起点为过程第一个真实事件小时，不是当日零点；终点为最后事件小时。早期窗口仅含`onset<=t<onset+6h`且真实事件的时点，六小时端点排除。邻近gap+1天有缺失则`uncertain_boundary=True`。这是事件日聚集代理，不能声称严格气象天气过程。

## 3. 全量边际、依赖及天气输入

F归档根：`F:\AcademicData\eu_joint_netload\archive\eu_joint_netload_20261002`。

原区：`results\upgrade_margin_dependence_20261002\upgrade_predictions.parquet`，43,824行，2021–2025。含`v0_q90_ZONE`、`v1_q90_ZONE`及q95（原/额外校准后的F(q)）、区`pit/raw_pit/calibrated_pit`、区物理threshold/net_load/exceed、6天气cell/stratum、rho_01/02/12、copula_origin、df_event_selected、24个nu候选概率与M0–M3。

丹麦：`results\upgrade_external_transfer_20261002\local_weather_strict15d\DK1\upgrade_predictions.parquet`（DK2类似），各43,824行2021–25。含三地F(q)/PIT/exceed、rho/nu、M0–3联合事件概率、all3概率、`*_DK_pivotal`概率与`y_*_DK_pivotal`。固定nu=3来自原区开发选择，未用丹麦选择。应以local_weather_strict15d为当前主要迁移流，macro_strict15d只为历史天气输入敏感性；不能混用旧macro/早期非严格延迟路径。

共同原始边际：`results\shared_frozen_history_test_20261001\prequential_combined_shared_margins.parquet`，同43,824行。含原/raw F(q)、季节阈值、17个边际分位数、观测净负荷、raw与每日校准PIT以及发报时点。

天气与时点：`datasets\joint_auxiliary_frozen_20261001\joint_features_lean.parquet`，同43,824行13列（5日历列、共同温度/陆风/海风/辐射的预测均值及spread）；同目录`timing.parquet`与`label_availability.parquet`。边际训练特征：`datasets\weather_panel_gefs_frozen_through_20260101\panel.parquet`61,368行2019–25，75特征+3区净负荷。前69为日历/天气/确定交互，再加入各区14/21天总负荷滞后。`margin_features.json`保留字段顺序及版本。

20天气成员不是20个成员特定边际预测后混合：成员地域风/温度/辐射先形成均值和标准差，进入GAM均值/尺度预测和动态相关天气基；共同均值为三区均值的等权平均，共同spread为各区集合SD的均值，不是地理均值的SD。`member_weather.parquet`1,227,360行（61,368×20）包含target/init/issue/member_id及成员地域变量。GEFS固定common1degree网格、成员1–20、6小时端点24/30/36/42/48，插值到交付24小时。代码：`weather_panel_builder.py`、`gefs_regional_features.py`。

DK天气：`results\upgrade_data_audit_20261002\DK_local_weather_summary_20181231_20251230.parquet`61,368行2019–25，包括DK1/2地域均值与spread。DK输入替换原边际BE天气槽并加丹麦日历，21/28天gross load滞后；DE/FR边际模块保持已有预测。DK新joint天气基使用DE/FR/相应DK三地均值。边界为固定地理proxy，不是逐像素电网边界。

## 4. 训练、校准、数值计算及真实可见性

原模型不是一次固定2019–2021训练后完全锁定系数。`shared...\prequential_manifest.json`记录20个季度，F从2019-01-01扩展历史，每季度更新，排除随后60天calibration窗；每日G只用发行日前7天之前成熟的60天PIT。最早2021Q1的F原始fit截至2020-10-26，2022–23开发选择策略，2024–25按冻结程序继续更新参数。现有2024–25结果已查看，ASMBI大修必须称再分析，不能重新宣称封存新验证。

GaussianAdditiveMargin是alpha10、5knots、三次样条的regularized location-scale：原始CDF为Gaussian标准化残差；均值与log残差方差分别拟合，scale有训练variance_factor。每日Beta PIT映射G，另有6固定weathercell每周Beta映射H。H每周first_actual_issue−7天、365天窗、至少28天及168小时、shape penalty4，否则identity。模型代码：`margins.py`、`prequential_margins.py`；额外map/4cell代码`upgrade_margin_dependence_20261002.py`。

F模型pickle已实际存在，例如`results\prequential_gefs_frozen_test_v1_20240101\prequential_20240101_additive_DE_LU.pickle`与FR/BE；开发季度在`prequential_gefs_daily_lag_matched_v3_additive_raw60_daily60_YYYYMMDD`。无须加载pickle即可做现概率流决策或原F(q)扰动；需要模型重训才恢复对应panel与模型配置到D盘。

相关矩阵通过动态Gaussian copula的天气基和部分相关tanh参数化保证正定；R原issued PIT季度拟合，四配置共用同一R。nu候选3/5/8/15/30/∞，由原区开发q90跨两边际平均loss选；测试选中3。事件概率代码`dependence.py::elliptical_event_probabilities`，fast路径`elliptical_event_fast.py`用确定性相关路径Gauss–Legendre积分；t的identity点用quad_vec积分，数值加阶差是收敛proxy而非认证上界。当前tolerance1e-6。已有复现差≤2e-12只证明一致性，不能代替提高积分阶数/更小容差后重新比较BS和gate临界名单；全F(q)、R允许独立复算。

`timing.available_time=initialization+12h`是模型约定，不是历史真实公开版本证明。`label_availability.label_available_time=index+1h`仅存间隔末端；7/15天成熟延迟由cutoff执行，不可把该字段直接当成熟。原区R与物理阈值保留quarter origin−7d，即首发前6.5天；每日G/额外map/nu严格first_issue−7d。DK新F/threshold/R严格first_issue−15d，DE/FR仍保留原module。全部电力历史snapshot可能已修订，缺乏原publicationvintage。重新训练仍不恢复当年可见历史版本。

## 5. 损失的三个原区早期过程

直接用公开P0/P1和同一gap1过程重建，少覆盖过程全部2024年：

| onset UTC | end UTC | 事件小时 | 前6h真实事件小时 | max early p | P1过程后续事件命中 |
|---|---|---:|---:|---:|---:|
| 2024-01-09 14:00 | 2024-01-19 10:00 | 129 | 2 | 0.107293 | 9 |
| 2024-04-08 06:00 | 2024-04-11 17:00 | 4 | 1 | 0.221027 | 1 |
| 2024-10-07 15:00 | 同时点 | 1 | 1 | 0.088034 | 0 |

三者均无uncertain boundary。第一过程温度预测−1.80℃，cool_higher_wind格，前6h法国最大阈值超越5705MW，德国1376MW；因此丢掉3个过程不能解释为都不严重。gate仍在该长过程后续命中9小时；损失是早期覆盖而非全过程完全漏检。具体字段和区超越值入input_map.json。

## 6. 丹麦特有迁移子集

测试完整域各17,400h，原DE/FR共同q90超越均924h，DK1全事件1641h、特有717h；DK2全事件1826h、特有902h。逐小时恒等式已验证：联合事件=DE/FR共同超越 + DK_pivotal（互斥）。有真实`y_q90_DK_pivotal`和模型对应概率，故可以将特有子集重新形成过程边界并报告早期覆盖、效用；应明示子集过程重新形成会与全事件过程起点不同。两个丹麦组合共享DE/FR证据，不能称两次完全独立复制。

## 7. 价格增量比较：可做与需补

现成`results\ese_revision_operational_20261002\dk_hourly_prices_2024_2025.parquet`35,088行=17,544小时×2区，字段target_time/PriceArea/price_eur_mwh/count/source_dataset；两区测试完整评分域各17,400价格无缺失。原价2025-10前Elspotprices，之后DayAheadPrices的4完整15分钟价格算术均值。`price_association_hourly_join.parquet`34,800行两区共同域，含真实特有标签、共同标签、区price。

2022–23开发价格不在上述已恢复资产；价格排序无需训练可直接作为情景比较，price+calendar/price+probability模型需先获取2022–23价格、在开发训练后2024–25评价。原官方获取逻辑在F归档`scripts\ese_revision_price_association_20261002.py`，manifest已保存官方meta及下载receipt。可沿现有接口请求`https://api.energidataservice.dk/dataset/Elspotprices`，start=2021-12-31，end=2024-01-02，filter为PriceArea=[DK1,DK2]，sort=HourUTC ASC，limit=100000。这个新增日期请求尚未执行或验证。若下载，新增raw遵守F:\AcademicData\eu_joint_netload\raw\新版本，不能覆盖旧材料。

价格current final snapshot不具历史首次可见版本。已有名单发布时间D-1 12UTC；日前拍卖CET/CEST日历时点可能早于名单，但必须核查最终auction publication而非仅gate closure。未核验时只称假定相应日前价格当时已可得的敏感性，不能真运营增量价值。

## 8. 最小恢复与环境

策略计算只需复制三个公开parquet（约数MB）和规则来源。校准决策稳定性只需F原区/DK3个概率parquet（约62MB）及13天气列；原始GAM不需重训。早期窗口模型可用2021–25全概率、对应天气、日历、按原边界重算早期目标进行简单Logistic/GAM按时间训练，保留7/15天成熟标签，不使用未实现的真实lags。2022/2023初训练与滚动origin评价可在D进行；全现有测试都已查看，诚实标再分析。

实际Python：`C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -B`，numpy2.5.1、pandas3.0.5、scipy1.18.0、scikit-learn1.9.0、pyarrow25.0.0均import成功。为训练设BLAS线程上限可由新D任务环境决定；旧版本pickle不必加载。保持F只读，训练/输出只写D的新ASMBI目录。