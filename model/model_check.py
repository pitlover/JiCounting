import torch
# from model.Depthformer.depthformer import Depthformer
from model.Depthformer.depthformer_v2 import DepthformerV2

model = DepthformerV2.build(opt={"hidden_dim": 512, "num_heads": 4, "img_size": (480, 640)},
                            min_depth=0.001, max_depth=80.0)

dummy_input = torch.empty(1, 3, 480, 640)
dummy_output, dummy_attn_weights = model(dummy_input)
print("Output:", dummy_output.shape)
for weight in dummy_attn_weights:
    print("Attn weight:", weight.shape)

num_params = model.count_params()
print("#Params:", num_params)

encoder_num_params = 0
for v in model.encoder.parameters():
    encoder_num_params += v.numel()
print("#Encoder Params:", encoder_num_params)
print("#New params:", num_params - encoder_num_params)