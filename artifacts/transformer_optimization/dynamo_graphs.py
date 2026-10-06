# ---- unpacked transformer block ----
def forward(self, L_self_modules_attention_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_x_ : torch.Tensor, L_self_modules_attention_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_attention_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_attention_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_attention_modules_out_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_self_modules_attention_norm_parameters_weight_ = L_self_modules_attention_norm_parameters_weight_
    l_x_ = L_x_
    l_self_modules_attention_modules_q_proj_parameters_weight_ = L_self_modules_attention_modules_q_proj_parameters_weight_
    l_self_modules_attention_modules_k_proj_parameters_weight_ = L_self_modules_attention_modules_k_proj_parameters_weight_
    l_self_modules_attention_modules_v_proj_parameters_weight_ = L_self_modules_attention_modules_v_proj_parameters_weight_
    l_self_modules_attention_modules_out_proj_parameters_weight_ = L_self_modules_attention_modules_out_proj_parameters_weight_
    l_self_modules_mlp_norm_parameters_weight_ = L_self_modules_mlp_norm_parameters_weight_
    l_self_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_mlp_modules_gate_proj_parameters_weight_
    l_self_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_mlp_modules_up_proj_parameters_weight_
    l_self_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_mlp_modules_down_proj_parameters_weight_
    rms_norm = torch.rms_norm(l_x_, (768,), l_self_modules_attention_norm_parameters_weight_, 1e-06);  l_self_modules_attention_norm_parameters_weight_ = None
    linear = torch._C._nn.linear(rms_norm, l_self_modules_attention_modules_q_proj_parameters_weight_, None);  l_self_modules_attention_modules_q_proj_parameters_weight_ = None
    view = linear.view(4, 128, 12, 64);  linear = None
    query = view.transpose(1, 2);  view = None
    linear_1 = torch._C._nn.linear(rms_norm, l_self_modules_attention_modules_k_proj_parameters_weight_, None);  l_self_modules_attention_modules_k_proj_parameters_weight_ = None
    view_1 = linear_1.view(4, 128, 12, 64);  linear_1 = None
    key = view_1.transpose(1, 2);  view_1 = None
    linear_2 = torch._C._nn.linear(rms_norm, l_self_modules_attention_modules_v_proj_parameters_weight_, None);  rms_norm = l_self_modules_attention_modules_v_proj_parameters_weight_ = None
    view_2 = linear_2.view(4, 128, 12, 64);  linear_2 = None
    value = view_2.transpose(1, 2);  view_2 = None
    attended = torch._C._nn.scaled_dot_product_attention(query, key, value, dropout_p = 0.0, is_causal = True);  query = key = value = None
    transpose_3 = attended.transpose(1, 2);  attended = None
    contiguous = transpose_3.contiguous();  transpose_3 = None
    merged = contiguous.view(4, 128, 768);  contiguous = None
    linear_3 = torch._C._nn.linear(merged, l_self_modules_attention_modules_out_proj_parameters_weight_, None);  merged = l_self_modules_attention_modules_out_proj_parameters_weight_ = None
    hidden = l_x_ + linear_3;  l_x_ = linear_3 = None
    rms_norm_1 = torch.rms_norm(hidden, (768,), l_self_modules_mlp_norm_parameters_weight_, 1e-06);  l_self_modules_mlp_norm_parameters_weight_ = None
    linear_4 = torch._C._nn.linear(rms_norm_1, l_self_modules_mlp_modules_gate_proj_parameters_weight_, None);  l_self_modules_mlp_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear_4);  linear_4 = None
    linear_5 = torch._C._nn.linear(rms_norm_1, l_self_modules_mlp_modules_up_proj_parameters_weight_, None);  rms_norm_1 = l_self_modules_mlp_modules_up_proj_parameters_weight_ = None
    gated = silu * linear_5;  silu = linear_5 = None
    linear_6 = torch._C._nn.linear(gated, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  gated = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
    add_1 = hidden + linear_6;  hidden = linear_6 = None
    return (add_1,)

# ---- packed transformer block ----
def forward(self, L_self_modules_attention_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_x_ : torch.Tensor, L_self_modules_attention_modules_qkv_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_attention_modules_out_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_gate_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_self_modules_attention_norm_parameters_weight_ = L_self_modules_attention_norm_parameters_weight_
    l_x_ = L_x_
    l_self_modules_attention_modules_qkv_proj_parameters_weight_ = L_self_modules_attention_modules_qkv_proj_parameters_weight_
    l_self_modules_attention_modules_out_proj_parameters_weight_ = L_self_modules_attention_modules_out_proj_parameters_weight_
    l_self_modules_mlp_norm_parameters_weight_ = L_self_modules_mlp_norm_parameters_weight_
    l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_mlp_modules_gate_up_proj_parameters_weight_
    l_self_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_mlp_modules_down_proj_parameters_weight_
    rms_norm = torch.rms_norm(l_x_, (768,), l_self_modules_attention_norm_parameters_weight_, 1e-06);  l_self_modules_attention_norm_parameters_weight_ = None
    linear = torch._C._nn.linear(rms_norm, l_self_modules_attention_modules_qkv_proj_parameters_weight_, None);  rms_norm = l_self_modules_attention_modules_qkv_proj_parameters_weight_ = None
    chunk = linear.chunk(3, dim = -1);  linear = None
    query = chunk[0]
    key = chunk[1]
    value = chunk[2];  chunk = None
    view = query.view(4, 128, 12, 64);  query = None
    transpose = view.transpose(1, 2);  view = None
    view_1 = key.view(4, 128, 12, 64);  key = None
    transpose_1 = view_1.transpose(1, 2);  view_1 = None
    view_2 = value.view(4, 128, 12, 64);  value = None
    transpose_2 = view_2.transpose(1, 2);  view_2 = None
    attended = torch._C._nn.scaled_dot_product_attention(transpose, transpose_1, transpose_2, dropout_p = 0.0, is_causal = True);  transpose = transpose_1 = transpose_2 = None
    transpose_3 = attended.transpose(1, 2);  attended = None
    contiguous = transpose_3.contiguous();  transpose_3 = None
    merged = contiguous.view(4, 128, 768);  contiguous = None
    linear_1 = torch._C._nn.linear(merged, l_self_modules_attention_modules_out_proj_parameters_weight_, None);  merged = l_self_modules_attention_modules_out_proj_parameters_weight_ = None
    hidden = l_x_ + linear_1;  l_x_ = linear_1 = None
    rms_norm_1 = torch.rms_norm(hidden, (768,), l_self_modules_mlp_norm_parameters_weight_, 1e-06);  l_self_modules_mlp_norm_parameters_weight_ = None
    linear_2 = torch._C._nn.linear(rms_norm_1, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  rms_norm_1 = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = None
    chunk_1 = linear_2.chunk(2, dim = -1);  linear_2 = None
    gate = chunk_1[0]
    up = chunk_1[1];  chunk_1 = None
    silu = torch.nn.functional.silu(gate);  gate = None
    mul = silu * up;  silu = up = None
    linear_3 = torch._C._nn.linear(mul, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  mul = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
    add_1 = hidden + linear_3;  hidden = linear_3 = None
    return (add_1,)
