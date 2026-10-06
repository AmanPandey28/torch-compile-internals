# ---- unpacked GatedMLP ----
def forward(self, L_self_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_x_ : torch.Tensor, L_self_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_self_modules_gate_proj_parameters_weight_ = L_self_modules_gate_proj_parameters_weight_
    l_x_ = L_x_
    l_self_modules_up_proj_parameters_weight_ = L_self_modules_up_proj_parameters_weight_
    l_self_modules_down_proj_parameters_weight_ = L_self_modules_down_proj_parameters_weight_
    linear = torch._C._nn.linear(l_x_, l_self_modules_gate_proj_parameters_weight_, None);  l_self_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear);  linear = None
    linear_1 = torch._C._nn.linear(l_x_, l_self_modules_up_proj_parameters_weight_, None);  l_x_ = l_self_modules_up_proj_parameters_weight_ = None
    gated = silu * linear_1;  silu = linear_1 = None
    linear_2 = torch._C._nn.linear(gated, l_self_modules_down_proj_parameters_weight_, None);  gated = l_self_modules_down_proj_parameters_weight_ = None
    return (linear_2,)

# ---- packed GatedMLP ----
def forward(self, L_self_modules_gate_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_x_ : torch.Tensor, L_self_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_self_modules_gate_up_proj_parameters_weight_ = L_self_modules_gate_up_proj_parameters_weight_
    l_x_ = L_x_
    l_self_modules_down_proj_parameters_weight_ = L_self_modules_down_proj_parameters_weight_
    linear = torch._C._nn.linear(l_x_, l_self_modules_gate_up_proj_parameters_weight_, None);  l_x_ = l_self_modules_gate_up_proj_parameters_weight_ = None
    chunk = linear.chunk(2, dim = -1);  linear = None
    gate = chunk[0]
    up = chunk[1];  chunk = None
    silu = torch.nn.functional.silu(gate);  gate = None
    mul = silu * up;  silu = up = None
    linear_1 = torch._C._nn.linear(mul, l_self_modules_down_proj_parameters_weight_, None);  mul = l_self_modules_down_proj_parameters_weight_ = None
    return (linear_1,)
