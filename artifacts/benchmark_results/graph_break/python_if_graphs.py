# ---- graph 1 ----
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2
    sum_1 = l_x_.sum();  l_x_ = None
    gt = sum_1 > 0;  sum_1 = None
    return (gt, y)


# ---- graph 2 ----
def forward(self, L_y_ : torch.Tensor):
    l_y_ = L_y_
    sin = torch.sin(l_y_);  l_y_ = None
    return (sin,)


# ---- graph 3 ----
def forward(self, L_y_ : torch.Tensor):
    l_y_ = L_y_
    cos = torch.cos(l_y_);  l_y_ = None
    return (cos,)
