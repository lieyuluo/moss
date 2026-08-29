# 🥋 moss

轻量级的 RLHF / SFT 实验仓库，当前包含 `RLVR`、`KD`、`Guide-KD`、监督微调和基础模型实现笔记。

## 项目结构

| 路径 | 用途 | 文档 |
| --- | --- | --- |
| `openrlhf-kd/` | 当前 RLVR / KD / Guide-KD 主线 | [模块说明](./openrlhf-kd/README.md) |
| `main_train.py` | SFT 训练入口 | [SFT 快速开始](#sft-快速开始) |
| `train_args/` | SFT 与 DeepSpeed 参数配置 | [文档索引](./docs/README.md) |
| `utils/` | 数据处理、collator 和模型工具脚本 | [文档索引](./docs/README.md) |
| `data/` | 最小训练数据示例 | [`sft_data.jsonl`](./data/sft_data.jsonl) |
| `llm_tricks/` | MoE、Transformer 等学习笔记 | [学习笔记索引](./llm_tricks/README.md) |

完整入口见 [文档索引](./docs/README.md)。

## 环境准备

SFT 入口所需依赖记录在根目录的 `requirements.txt`：

```bash
pip install -r requirements.txt
```

RLVR / KD 训练依赖及运行方式见 [`openrlhf-kd/README.md`](./openrlhf-kd/README.md)。训练依赖中包含对 PyTorch、DeepSpeed 和 vLLM 的版本约束，建议在独立环境中安装。

## RLVR / KD / Guide-KD

`openrlhf-kd` 基于 OpenRLHF 精简并扩展，保留 RLVR 训练主链路，支持：

- 按 `datasource` 路由 reward、teacher、KD 和 Guide-KD；
- 纯 RLVR、纯 KD、纯 Guide-KD，以及多种混合训练模式；
- 本地代码 reward、远程 API reward 和多 teacher 配置。

配置、数据格式与命令示例见 [Reward / KD / Guide-KD 使用指南](./openrlhf-kd/examples/README.md)。

## SFT 快速开始

根目录提供简洁的 SFT 训练入口，支持 DeepSpeed、LoRA、QLoRA、全参数微调和 chat template 适配。

```bash
bash run_example.sh
```

或直接启动：

```bash
deepspeed --include localhost:0,1 main_train.py \
  --train_data_path /path/to/data.jsonl \
  --model_name_or_path /path/to/model \
  --task_type sft \
  --train_mode qlora \
  --output_dir /path/to/output
```

多机启动示例见 [`multinode_run.sh`](./multinode_run.sh)。
