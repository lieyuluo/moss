# moss 文档索引

本文档按使用场景组织仓库入口。模块内部的细节文档继续与代码放在一起，便于同步维护。

## 训练指南

### RLVR / KD / Guide-KD

- [模块概览](../openrlhf-kd/README.md)
- [数据格式、核心配置与训练模式](../openrlhf-kd/examples/README.md)
- [Code Eval 指标说明](../openrlhf-kd/examples/code_reward/code_eval/README.md)

### SFT

- [根目录快速开始](../README.md#sft-快速开始)
- [训练入口](../main_train.py)
- [单机示例脚本](../run_example.sh)
- [多机示例脚本](../multinode_run.sh)
- [训练参数](../train_args/)
- [示例数据](../data/sft_data.jsonl)

## 学习笔记

- [学习笔记总览](../llm_tricks/README.md)
- [从零实现 MoE](../llm_tricks/moe/README.md)
- [Transformer 实现说明](../llm_tricks/transformer/README.md)

## 文档维护约定

- 根目录 `README.md` 只保留项目定位、结构和最短上手路径。
- 模块专属文档放在对应模块目录，避免与实现脱节。
- `docs/README.md` 作为统一导航，不重复模块内的大段说明。
- 新增或移动文档时，同步更新本索引及上级模块 README 中的相对链接。
