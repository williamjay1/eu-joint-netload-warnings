# ASMBI大修：核心文献核验与贡献定位（2026-10-08）

本次为有边界的定向文献审查，不宣称系统综述。问题拆为四项：成熟联合能源预测已经解决什么；预测质量怎样转成带成本/容量的决策价值；怎样评价时间聚集事件的初期发现；校准误差怎样影响选择名单而不只影响平均分数。检索使用agent-reach的Exa、Jina Reader与内置Web搜索两条检索路径，只以论文原文、作者/大学存储库、出版方及官方数据说明支持结论。现稿23条bib先去重，新增9条经过题名、作者、出处/DOI核对；已有Gioia2025、Gneiting2007a/b、Ziegel2014、Browell2021/2022、Demarta2005保留。

## 核心判断

相对于最接近研究，本文可信的新增能力应是：**在同一已发布概率流上，把小时命中和过程初期发现分开，比较有容量上限、允许零审核的实际选择规则，并将边际概率扰动传播到gate、排名和冷却路径的稳定性。** 成本阈值本身、时间范围评价、动态copula都已有成熟研究；新意须由公平策略比较、机制模拟与失败边界支撑，不能仅由新标题或不显著的Gaussian–t差异推出。[Gioia等](https://doi.org/10.1080/01621459.2024.2412361)，[Hirsch与Ziel](https://doi.org/10.1002/asmb.2837)，[Ehm等](https://doi.org/10.1111/rssb.12154)，[Franc等](https://jmlr.org/papers/v24/21-0048.html)，[Tatbul等](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)

## 最接近研究对照

| 研究 | 已解决的核心任务 | 原文证据及读深 | 本文须提供的具体增量 |
|---|---|---|---|
| Gioia等，JASA 2025 | 英国14区域日前净需求的天气条件动态协方差；修正Cholesky/GAM及选择；以预测分数评价，还检验宏区域聚合及区域差。 | Glasgow出版接受稿37页，正文与结论深读；2024-10-11接受，最终卷120(549):107–119。 | 不争“首先天气条件相关建模”；增加有明确每日容量/成本的选择行为、一次过程价值及校准扰动后的决策稳定性。 |
| Hirsch与Ziel，ASMBI 2024 | 德国日内市场跨产品价格变化的高维模拟预测；零膨胀Johnson SU边际和copula依赖，建模交易结构，CRPS/energy score评价。 | 最终期刊书目信息核验；全文读作者2023 arXiv v1 27页，不冒充最终发表全文。 | 数据不是本文净负荷，目标不是有限审核；本文关注冻结预测流怎样转化为资源约束筛选，及分数差异何时不改变决策。 |
| Ehm等，JRSSB 2016 | 一致评分函数的基本评分混合表示；Murphy diagrams阈值决策解释，预测排序依赖损失偏好。 | 官方出版信息及作者preprint v2 34页，方法与讨论深读。 | 引用“平均分数不能替代具体决策损失”的既有基础；有容量耦合和过程价值时需重放共同合同，不能自动套单点Murphy dominance。 |
| Franc等，JMLR 2023 | reject option下成本、覆盖与选择风险三类约束；最优选择以条件风险阈值并处理边界随机化。 | JMLR全文49页，问题、三种合同与最优选择章节深读。 | 本文是高风险小时触发人工审核，不是分类器拒绝不确定预测；借其合同清晰性，不冒称p>c/v加top-K为新定理。 |
| Tatbul等，NeurIPS 2018 | 时间范围而非独立点的precision/recall；存在、重叠、重复警告与前端位置偏好。 | 官方会议页及MIT作者全文11页，定义和结论深读。 | 六小时初期、跨空日合并天气过程和一次覆盖价值是本应用合同，不是首次提出时间范围评价。 |
| Morris等，Stat Med 2019 | ADEMP组织模拟，模拟指标本身的MCSE、共同随机数与可复核报告。 | PMC发表全文深读模拟设计及MCSE章节。 | 明确机制场景与真值，成对比较审核收益/漏检/名单变化，记录重复次数和MCSE，不能把模拟画成现实电网证据。 |

近邻原文：[Gioia接受稿](https://eprints.gla.ac.uk/342345/1/342345.pdf)，[Hirsch作者稿](https://arxiv.org/abs/2306.13419)，[Ehm作者稿](https://arxiv.org/abs/1503.08195)，[Franc发表全文](https://jmlr.org/papers/volume24/21-0048/21-0048.pdf)，[Tatbul发表全文](https://people.csail.mit.edu/tatbul/publications/NeurIPS18.pdf)，[Morris发表全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/)。

## 应进入正文的知识链

### 1. 预测质量与经济价值之间已经有成熟桥梁

Murphy的cost-loss分析与Richardson的集合预报经济价值研究表明，用户成本/损失不同，预测价值和标准技巧分数可能不同。本文不能将这种一般区别本身包装成创新。可将它落实为审核成本c、真实目标命中的归一化价值v、每日上限K的明确合同，报告成本区间下的规则排序，并证明哪些评分差异实际改变了入选小时。[Murphy 1977](https://doi.org/10.1175/1520-0493(1977)105%3C0803:TVOCCA%3E2.0.CO;2)，[Richardson 2000](https://doi.org/10.1002/qj.49712656313)

Ehm等给出阈值决策与评分排序的理论背景。本文若画归一化效用随c/v变化的曲线，称cost-value curve即可；除非实际计算了对应基本一致评分，勿把任意曲线命名为Murphy diagram。日容量下跨小时竞争、过程初期一次覆盖和冷却路径，使其决策目标超出无耦合单点评分函数。[Ehm等2016](https://doi.org/10.1111/rssb.12154)

### 2. “可不审核”是选择合同，不自动是新算法

Chow和Franc等证明成本/拒绝/覆盖约束的选择阈值思想已知。在各小时收益可加且命中价值相同时，期望净值最大化导致p>c/v及容量内最高概率排序，是一个基础决策推论。若本文有过程初期一次价值，目标为非可加覆盖，基础阈值规则仅是强基线；冷却或rising priority需要与同一阈值、同容量的替代规则比较，不能从其名称推断最优。[Chow 1970](https://doi.org/10.1109/TIT.1970.1054406)，[Franc等2023](https://jmlr.org/papers/v24/21-0048.html)

### 3. 平均校准与选择附近校准需要区分

已有Gneiting等2007区分概率校准、锐度等概念，Ziegel与Gneiting2014研究copula calibration。均匀PIT或若干总体频率接近，不能自行保证给定天气状态、阈值附近及每日排名的每一种校准。Dimitriadis等2021展示传统分箱可靠性图对实现选择不稳定，并以PAV给出稳定诊断；若本稿没有运行CORP，不声称使用了CORP。[Gneiting等2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x)，[Ziegel与Gneiting2014](https://doi.org/10.1214/14-EJS964)，[Dimitriadis等2021](https://doi.org/10.1073/pnas.2016191118)

本文边际扰动研究必须说明扰动是设定的敏感性情景还是有概率保证的估计集合。条件格均值残差不是逐小时真概率的置信半径。固定copula下耦合/union bound给出的事件概率变化界是基础不等式；可评估的工作在于这一界或实际重算如何传播到选择门槛、K与K+1排名间隙及跨日冷却历史，而不是重新命名定理。[既有概率校准基础](https://doi.org/10.1111/j.1467-9868.2007.00587.x)，[选择风险与合同基础](https://jmlr.org/papers/v24/21-0048.html)

### 4. 时间聚集会改变审核的边际价值

Tatbul等已将事件存在、覆盖比例、告警重复和事件前端偏好纳入时间范围评价。本文应明确每个天气过程初始六小时的事件小时集合，过程只记一次命中。自然分子、过程分母、boundary uncertain标记及空日合并规则全部可追溯；标准化稀释公式仅是解析暴露对齐，不能充当在线可执行控制策略。[Tatbul等2018](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)

Gioia等接受稿已指出缺乏时间连贯轨迹是其未来方向。本文不能宣称已有成熟联合预测从未考虑运行目标；它们已经检验区域聚合与传输相关差值。本稿的窄而明确贡献是让时间过程目标参与审核合同及性能解释。[Gioia接受稿](https://eprints.gla.ac.uk/342345/1/342345.pdf)

## 引用配置及可直接改写的英文素材

主线新增引用建议Hirsch2024、Richardson2000、Ehm2016、Franc2023、Tatbul2018、Dimitriadis2021和Morris2019。Murphy1977/Chow1970是可选奠基背景，不为凑数量双重展开。既有Gioia2025是最重要联合能源预测近邻；Gneiting2007a/b与Ziegel2014保留作预测质量基础，Demarta2005作t属性基础。新bib仅含新增9条，供主代理合并。

候选引言素材（须与最终实验相符，不预写未经验证的优势）：

> A joint forecast can describe simultaneous regional demand without determining which hours deserve review. Dynamic covariance models already provide competitive joint net-demand distributions, and multivariate simulation models capture dependence in electricity markets. Our question concerns the decision made from such distributions: which forecast hours should be selected when review is costly, daily capacity is limited, and repeated selections within a prolonged episode have different value from its early detection? Forecast economic value and selective prediction provide the general decision foundations. We evaluate this contract using a common probability stream and trace marginal probability perturbations through thresholds, ranks and cooldown histories.

建议引文分配：第一句后无须堆引文；第二句引Gioia2025/Hirsch2024；决策基础引Richardson2000/Ehm2016/Franc2023；过程评价引Tatbul2018；模拟段引Morris2019。以上定位是本文应检验的新增能力，不等于当前结果已经确证。

## 原文获取与书目信息边界

| 引用 | 书目信息核验 | 原文身份/访问深度 |
|---|---|---|
| Gioia2025 | JASA120(549):107–119，DOI 10.1080/01621459.2024.2412361 | Glasgow接受稿37页全文；arXiv v3较早版本47页；Bristol下载链接返回Cloudflare HTML，不作论文使用。 |
| Hirsch2024 | ASMBI40(6):1571–1595，DOI 10.1002/asmb.2837；Crossref及大学作者出版页交叉核验 | arXiv 2306.13419 v1全文27页；最终Wiley全文访问受限。 |
| Ehm2016 | JRSSB78(3):505–562，DOI 10.1111/rssb.12154；当前出版方OUP核验 | 作者arXiv v2全文34页。 |
| Richardson2000 | QJRMS126(563):649–667，DOI 10.1002/qj.49712656313；官方出版摘要核验 | ECMWF 1998 Technical Memorandum262前身PDF26页下载成功但文本字库抽取乱码；不声称该PDF正文已逐页阅读。观点依据发表摘要。不要混用memo DOI与2000文章DOI。 |
| Murphy1977 | MWR105(7):803–816，DOI在新增bib中保留尖括号原形式 | 官方摘要/书目信息，未获可读取发表全文。 |
| Franc2023 | JMLR24(11):1–49 | 官方49页发表全文；无核实DOI，不编造。 |
| Chow1970 | IEEE IT16(1):41–46，DOI10.1109/TIT.1970.1054406 | IBM作者官方摘要及Franc原文引用核验，未深读发表全文。 |
| Tatbul2018 | NeurIPS31，五位作者Tatbul/Lee/Zdonik/Alam/Gottschlich | 官方会议记录、作者MIT全文11页；不要误引同年另一个range-based草稿及不同作者顺序。 |
| Dimitriadis2021 | PNAS118(8):e2016191118，DOI10.1073/pnas.2016191118 | KIT存储的发表全文10页，定义/方法深读。 |
| Morris2019 | Stat Med38(11):2074–2102，DOI10.1002/sim.8086 | PMC发表全文，ADEMP/MCSE/共同随机数段落深读。 |

新增原始文献保存于F盘独立修订raw目录，分析报告和bib存D盘；未修改旧F归档。原材料路径仅用于后台报告，不进入主文。细节及可访问性参见各链接：[Hirsch作者出版记录](https://dsee.wiwi.uni-due.de/en/research/publications/multivariate-simulation-based-forecasting-for-intraday-power-markets-modeling-cross-product-price-effects-17077/)，[Ehm出版记录](https://academic.oup.com/jrsssb/article-abstract/78/3/505/7040984)，[Richardson发表摘要](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.49712656313)，[Richardson ECMWF前身报告](https://www.ecmwf.int/en/elibrary/76153-skill-and-relative-economic-value-ecmwf-ensemble-prediction-system)，[Chow IBM摘要](https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff)，[Dimitriadis发表PDF](https://publikationen.bibliothek.kit.edu/1000130510/106591216)。

## 新增价格竞争解释的边界

价格模型可检验“已有市场信号是否已经解释风险排序”的竞争解释，但2026下载的Energinet历史价没有逐记录首发/修订时间；D−1 noon UTC可见性只能假设。Elspotprices小时序列在2025-09-30终止，其后DayAheadPrices为15分钟，转小时需要四期完整均值。Nord Pool2022正常公布时间12:45 CET与Energinet当前13–14 CET获取建议均不能代替每天真实发布日志。故辅助价格比较是assumed-availability retrospective benchmark，不是历史实时部署。[Nord Pool 2022官方指南](https://support.nordpoolgroup.com/support/solutions/articles/8000077444-about-the-day-ahead-auction-in-nordics-baltics)，[Elspotprices元数据](https://api.energidataservice.dk/meta/dataset/Elspotprices)，[DayAheadPrices元数据](https://api.energidataservice.dk/meta/dataset/DayAheadPrices)

价格为日前出清价格，不是不平衡结算价，也不直接测量备用不足。具体辅助模型与规则仍需保持2022拟合、2023选择、2024–2025再分析的明确时间顺序；新的检验属于大修时追加分析，不能伪称原封存期前预注册。历史时点与外部比较限制已写入journal_requirements.md。[官方数据集说明](https://api.energidataservice.dk/meta/dataset/Elspotprices)
