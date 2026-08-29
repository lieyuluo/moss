# OpenRLHF-KD

本目录是仓库的 RLVR / KD / Guide-KD 主线实现。它基于 OpenRLHF 的训练结构进行精简，并增加按 `datasource` 路由的 reward、teacher、KD 与 Guide-KD 能力。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `openrlhf/cli/` | 训练命令入口 |
| `openrlhf/datasets/` | prompt 数据集与数据工具 |
| `openrlhf/models/` | Actor、loss 与模型辅助逻辑 |
| `openrlhf/trainer/` | PPO / KD 训练器、经验生成与 Ray workers |
| `openrlhf/utils/` | DeepSpeed、采样、日志和远程 reward 工具 |
| `examples/` | reward、teacher 与训练配置示例 |

## 使用入口

环境依赖应与目标 CUDA、PyTorch、DeepSpeed 和 vLLM 版本匹配；上游依赖可参考 [OpenRLHF requirements](https://github.com/OpenRLHF/OpenRLHF/blob/main/requirements.txt)。

训练数据格式、核心配置、不同训练模式及完整命令示例见：

- [Reward / KD / Guide-KD 使用指南](./examples/README.md)
- [Code Eval 指标与安全说明](./examples/code_reward/code_eval/README.md)

最小命令形态：

```bash
cd openrlhf-kd
python -m openrlhf.cli.train_ppo_ray \
  --pretrain /path/to/model \
  --prompt_data /path/to/dataset \
  --input_key input \
  --label_key label \
  --remote_rm_url examples/guide-kd-reward.py \
  --advantage_estimator reinforce
```

请在隔离环境中运行会执行模型生成代码的本地 Code Eval reward；具体风险说明见其指标文档。
