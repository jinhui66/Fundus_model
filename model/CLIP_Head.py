import torch
from torch import nn
from Config import parse_args
from typing import List
import torch.nn.functional as F 

args = parse_args()
# 获取运行的设备
if args.device != 'cpu':
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# MoE 参数
class MoeArgs:
    def __init__(self):
        self.num_experts_per_tok = 2  # 每个输入选 2 个专家

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        print(selected_experts, weights)
        results = torch.zeros([inputs.shape[0], 7]).to(device)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # print(results[batch_idx].shape, weights[batch_idx, nth_expert, None].shape, expert(inputs[batch_idx], text[batch_idx]).shape)
            # print(batch_idx, nth_expert)

            # results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx], text[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx], label[batch_idx])
        # print(results)
        return results
