# FX graphs captured with torch.compile(dynamic=None).
# Graph 1 starts specialized; later graphs may introduce symbolic inputs.

# ---- graph 1 ----
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

# ---- graph 2 ----
def forward(self, L_self_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, s27 : torch.SymInt, L_x_ : torch.Tensor, L_self_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
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
