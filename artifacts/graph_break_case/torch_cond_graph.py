# ---- graph 1 ----
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2
    sum_1 = l_x_.sum();  l_x_ = None
    gt = sum_1 > 0;  sum_1 = None
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, (y,));  gt = cond_true_0 = cond_false_0 = y = None
    getitem = cond[0];  cond = None
    return (getitem,)
