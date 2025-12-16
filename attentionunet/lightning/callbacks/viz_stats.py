import torch

# Training-set z-score stats

# S1 (VV, VH)
S1_MEAN = torch.tensor([-3.8912460803985596, -7.705480575561523])
S1_STD  = torch.tensor([6.468840599060059, 11.871362686157227])

# S2 (B2, B3, B4, B8)
S2_MEAN = torch.tensor([
    1549.42333984375,
    1361.87548828125,
    1372.6019287109375,
    1037.02490234375
])
S2_STD = torch.tensor([
    2861.3310546875,
    2766.714111328125,
    2694.313720703125,
    2237.964599609375
])

# Optional tone mapping divisor
S2_TONEMAP_DIV = 10000.0
