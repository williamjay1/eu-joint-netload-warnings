# 既有归档与补充材料托管状态（2026-10-06）

作者已完成代码和派生评价数据的 Zenodo 1.0.0 归档，正式版本 DOI 为 [10.5281/zenodo.23100589](https://doi.org/10.5281/zenodo.23100589)。GitHub 标签 `v1.0.0` 保留原版本。

本次新加入 `supplemental_materials/` 的补充 PDF、Fig. S1、26 CSV 导出及10支持表单独托管在 GitHub；不能把它们说成已包含在旧 Zenodo DOI 中。期刊稿件使用具体提交版本链接定位这些文件。本次没有创建新的 GitHub release 或 Zenodo DOI。

后续若作者希望将新补充材料归档到 Zenodo，应由作者建立新的版本并核对完整文件，再使用真实的新版本 DOI。不要覆盖原1.0.0归档。

## 2026-10-02 首次归档准备流程（历史记录）

以下为已完成的首次归档流程，保留作记录，不应再次创建 `v1.0.0`。

# 由作者完成 Zenodo 归档

仓库：[williamjay1/eu-joint-netload-warnings](https://github.com/williamjay1/eu-joint-netload-warnings)。已准备 `CITATION.cff` 和 `.zenodo.json`；此准备没有创建 Zenodo 记录、DOI 或 GitHub release。

1. 登录正式 [Zenodo](https://zenodo.org/)，连接拥有此仓库的 GitHub 账户。在个人菜单的 GitHub 页面点击 **Sync now**，找到仓库并开启归档开关。先启用仓库，再创建新 release。[官方连接说明](https://help.zenodo.org/docs/github/enable-repository/)。
2. 核对仓库元数据：作者 **Junjie Zhang**；ORCID **0009-0004-8821-4018**；上海外国语大学两个单位；版本 **1.0.0**。代码使用 MIT，派生研究数据使用 CC BY 4.0，源数据和 SciencePlots 的例外见 `DATA_LICENSES.md`。两份元数据文件同时存在时，Zenodo 采用 `.zenodo.json`。[官方元数据说明](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)。
3. 在 GitHub 的 **Releases → Draft a new release** 中，从核对完的 `main` 提交创建标签 **v1.0.0**，填写版本说明并发布正式 release。版本说明注明这次归档支持逐时预测结果的评估重算，不是所有原始数据的完整重新拟合。[官方 release 归档说明](https://help.zenodo.org/docs/github/archive-software/github-upload/)。
4. 等 Zenodo 处理成功后，打开归档记录并核对文件、作者、版本和许可说明。保存这一个明确版本的 **version DOI**；论文引用可复现版本时用这个 DOI。概念 DOI 对应整个版本系列，应与版本 DOI 区分。
5. 将真实版本 DOI 加入主稿的 Data and code availability、补充材料 S14 和引用信息。保留已归档的 v1.0.0；后续代码或数据修改应产生新版本，不覆盖已归档版本。文章 DOI 只有在期刊实际分配后再填入。

这些操作由作者亲自执行。归档 DOI 表示研究产物已保存，不代表论文已投稿或被期刊录用。
