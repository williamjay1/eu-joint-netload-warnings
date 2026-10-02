# 由作者完成 Zenodo 归档

仓库：[williamjay1/eu-joint-netload-warnings](https://github.com/williamjay1/eu-joint-netload-warnings)。已准备 `CITATION.cff` 和 `.zenodo.json`；此准备没有创建 Zenodo 记录、DOI 或 GitHub release。

1. 登录正式 [Zenodo](https://zenodo.org/)，连接拥有此仓库的 GitHub 账户。在个人菜单的 GitHub 页面点击 **Sync now**，找到仓库并开启归档开关。先启用仓库，再创建新 release。[官方连接说明](https://help.zenodo.org/docs/github/enable-repository/)。
2. 核对仓库元数据：作者 **Junjie Zhang**；ORCID **0009-0004-8821-4018**；上海外国语大学两个单位；版本 **1.0.0**。代码使用 MIT，派生研究数据使用 CC BY 4.0，源数据和 SciencePlots 的例外见 `DATA_LICENSES.md`。两份元数据文件同时存在时，Zenodo 采用 `.zenodo.json`。[官方元数据说明](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)。
3. 在 GitHub 的 **Releases → Draft a new release** 中，从核对完的 `main` 提交创建标签 **v1.0.0**，填写版本说明并发布正式 release。版本说明注明这次归档支持逐时预测结果的评估重算，不是所有原始数据的完整重新拟合。[官方 release 归档说明](https://help.zenodo.org/docs/github/archive-software/github-upload/)。
4. 等 Zenodo 处理成功后，打开归档记录并核对文件、作者、版本和许可说明。保存这一个明确版本的 **version DOI**；论文引用可复现版本时用这个 DOI。概念 DOI 对应整个版本系列，应与版本 DOI 区分。
5. 将真实版本 DOI 加入主稿的 Data and code availability、补充材料 S14 和引用信息。保留已归档的 v1.0.0；后续代码或数据修改应产生新版本，不覆盖已归档版本。文章 DOI 只有在期刊实际分配后再填入。

这些操作由作者亲自执行。归档 DOI 表示研究产物已保存，不代表论文已投稿或被期刊录用。
