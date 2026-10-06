def forward(self, L_x_ : torch.Tensor, L_bias_ : torch.Tensor):
    l_x_ = L_x_
    l_bias_ = L_bias_
    add = l_x_ + l_bias_;  l_x_ = l_bias_ = None
    y = torch.nn.functional.silu(add);  add = None
    mul = y * y;  y = None
    mean = mul.mean(dim = -1);  mul = None
    return (mean,)
