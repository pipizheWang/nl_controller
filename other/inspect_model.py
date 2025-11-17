import os
import torch
import torch.nn.functional as F

# 复用已有实现
from neural_network import load_model


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    # 指定模型权重路径（根据用户提供的绝对路径）
    pth_path = "/home/zhe/px4_ws/src/nl_controller/nl_controller/neural-fly_dim-a-4_v-q-pwm-epoch-950.pth"
    model_folder = os.path.dirname(pth_path)
    model_name = os.path.splitext(os.path.basename(pth_path))[0]

    # 加载模型（使用 neural_network.py 中的 load_model）
    model = load_model(modelname=model_name, modelfolder=model_folder)
    phi_net, h_net, options = model

    print("==== Model Options ====")
    for k, v in options.items():
        print(f"{k}: {v}")

    print("\n==== Phi_Net (feature network) ====")
    print(phi_net)
    print(f"trainable params: {count_parameters(phi_net)}")

    print("\n==== H_Net_CrossEntropy (head/classifier) ====")
    print(h_net)
    print(f"trainable params: {count_parameters(h_net)}")

    # 构造 11 维随机输入（用户要求）
    torch.manual_seed(42)
    dim_x = 11
    x = torch.randn(dim_x)  # shape: [11]
    print("\n==== Random Input ====")
    print(f"x.shape: {tuple(x.shape)}")
    print(x)

    # 前向传播
    with torch.no_grad():
        phi_out = phi_net(x)
        print("\n==== Phi_Net Output ====")
        print(f"phi_out.shape: {tuple(phi_out.shape)}  (expected dim_a)")
        print(phi_out)

        h_out = h_net(phi_out)
        print("\n==== H_Net Output (logits) ====")
        print(f"h_out.shape: {tuple(h_out.shape)}  (expected num_c)")
        print(h_out)

        # 可选：打印 softmax 概率
        probs = F.softmax(h_out, dim=-1)
        print("\n==== H_Net Probabilities (softmax) ====")
        print(probs)


if __name__ == "__main__":
    main()
