class EfficientBlendedRotaryEmbedding(nn.Module):
    def __init__(self, base, dims, head, theta_learnable=True, rot_learnable=True,
                 matrix_learnable=False, freq_learnable=True):
        super(EfficientBlendedRotaryEmbedding, self).__init__()
        self.base = base
        self.dims = dims
        self.head = head

        self.h_dim = self.dims // self.head
        self.rot = (self.dims // self.head) // 2

        self.thetas = nn.Parameter(torch.zeros(self.rot))
        self.r_pairs = nn.Parameter(data=torch.rand(self.rot, 2) * self.h_dim)

        self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_learnable)
        self.rot_scale = nn.Parameter(torch.ones(1), requires_grad=rot_learnable)

        self.r_matrix = nn.Parameter(torch.eye(n=self.h_dim), requires_grad=matrix_learnable)

        freq_data = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
        self.inv_freq = nn.Parameter(freq_data, requires_grad=freq_learnable)

        self.orthogonal_reg_weight = 0.01

    def blended_rotation_matrix(self, dims, i, j, theta):
        G = torch.eye(dims).to(theta.device)
        G[i, i] = torch.cos(theta)
        G[i, j] = -torch.sin(theta)
        G[j, i] = torch.sin(theta)
        G[j, j] = torch.cos(theta)

        v = torch.zeros(dims).to(theta.device)
        v[i] = torch.cos(theta)
        v[j] = torch.sin(theta)
        H = torch.eye(dims).to(theta.device) - 2 * torch.outer(v, v) / torch.dot(v, v)

        R = torch.eye(dims).to(theta.device)
        R[i, i] = torch.cos(theta)
        R[i, j] = -torch.sin(theta)
        R[j, i] = torch.sin(theta)
        R[j, j] = torch.cos(theta)

        return (G + H + R) / 3

    def apply_blended_rotation(self, x):
        adjusted_rot = int(torch.round(self.rot_scale * self.rot))
        for k in range(adjusted_rot):
            i, j = self.r_pairs[k].long()
            theta = self.thetas[k] * self.theta_scale
            B = self.blended_rotation_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
            x = torch.matmul(input=x, other=B)
        return x

    def update_base(self, new_base):
        if new_base is not None and new_base != self.base:
            self.base = new_base
            inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
            self.inv_freq.data.copy_(inv_freq)
            self.update_pairs()

    def reset_parameters(self):
        nn.init.orthogonal_(self.r_matrix)
        nn.init.zeros_(self.thetas)

    def orthogonal_regularization_term(self):
        loss = torch.tensor(0.0, device=self.r_matrix.device)
        if self.r_matrix.requires_grad:
            product = torch.matmul(self.r_matrix, self.r_matrix.t())
            identity = torch.eye(self.r_matrix.size(0)).to(self.r_matrix.device)
            loss = ((product - identity) ** 2).sum()
        return self.orthogonal_reg_weight * loss

    def update_pairs(self):
        pairs = []
        while len(pairs) < self.rot:
            i, j = torch.randint(0, self.h_dim - 1, (2,))
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))
        self.r_pairs.data.copy_(torch.tensor(pairs, dtype=torch.float32))

    def forward(self, x, global_step=None):
        if x.dim() not in [3, 4]:
            raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")

        batch_size, seq_len, *rest = x.size()

        if x.dim() == 3:
            dims = rest[0]
            if dims != self.head * self.h_dim:
                raise ValueError(f"Expected dims ({dims}) to be compatible with head ({self.head}) * h_dim ({self.h_dim}={self.head * self.h_dim})")
        else:
            head, h_dim = rest
            if head != self.head or h_dim != self.h_dim:
                raise ValueError(f"For 4D input, expected head {self.head} and h_dim {self.h_dim}, but got head {head} and h_dim {h_dim}")

        x = x.view(batch_size, seq_len, self.head, self.h_dim)
        x = x.reshape(-1, self.h_dim)

        x = self.apply_blended_rotation(x)

        x = torch.matmul(input=x, other=self.r_matrix)

        x = x.view(batch_size, seq_len, self.head, self.h_dim)

        sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device), self.inv_freq.to(device=x.device))
        sin = sinusoid_inp.sin()[None, :, None, :]
        cos = sinusoid_inp.cos()[None, :, None, :]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x = x.view(batch_size, seq_len, self.dims)

        return x
