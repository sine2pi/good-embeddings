#### This enbedding system is set for 3 dimensions. You can edit it for more or less. It relies on the base frequency updates via loss from your models forward pass. 
##### These go in your model class : 
     
         
     class CombinedRotaryEmbedding(nn.Module):
         def __init__(self, n_state: int, n_head: int, n_freq: float,
                      theta_scale_learnable: bool = True,
                      n_rots_scale_learnable: bool = True,
                      r_matrix_learnable: bool = False,
                      inv_freq_learnable: bool = False):
             super().__init__()
             self.n_state = n_state
             self.n_head = n_head
             self.n_freq = n_freq
             assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
             self.h_dim = self.n_state // self.n_head
             assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
             self.n_rots = ((n_state // n_head) // 2)
     
             # --- Learnable Parameters ---
             self.thetas = nn.Parameter(torch.zeros(self.n_rots))
             self.r_pairs = nn.Parameter(data=torch.rand(self.n_rots, 2) * self.h_dim)
     
             # --- Scaling Parameters ---
             self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_scale_learnable)
             self.n_rots_scale = nn.Parameter(torch.ones(1), requires_grad=n_rots_scale_learnable)
     
             # --- R Matrix ---
             self.r_matrix = nn.Parameter(torch.eye(n=self.h_dim), requires_grad=r_matrix_learnable)
     
             # --- Frequency Parameters for RoPE ---
             inv_freq_data = 1.0 / (self.n_freq ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
             self.inv_freq = nn.Parameter(inv_freq_data, requires_grad=inv_freq_learnable)
     
             # --- Regularization ---
             self.orthogonal_reg_weight = 0.1  # Weight for orthogonal regularization
     
         def givens_r_matrix(self, n_state, i, j, theta):
             G = torch.eye(n_state).to(theta.device) # Update device handling
             G[i, i] = math.cos(theta)
             G[i, j] = -math.sin(theta)
             G[j, i] = math.sin(theta)
             G[j, j] = math.cos(theta)
             return G
     
         def update_base(self, new_base):
             if new_base is not None and new_base != self.n_freq:
                 self.n_freq = new_base
                 inv_freq = 1.0 / (self.n_freq ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
                 self.inv_freq.data.copy_(inv_freq)
     
         def reset_parameters(self):
             nn.init.orthogonal_(tensor=self.r_matrix)
             nn.init.zeros_(tensor=self.thetas)
     
         def orthogonal_regularization_term(self):
             """Calculates the orthogonal regularization term for r_matrix."""
             loss = torch.tensor(0.0, device=self.r_matrix.device) # Update device handling
             if self.r_matrix.requires_grad:  # Only calculate if r_matrix is learnable
                 product = torch.matmul(self.r_matrix, self.r_matrix.t())
                 identity = torch.eye(self.r_matrix.size(0)).to(self.r_matrix.device)
                 loss = ((product - identity) ** 2).sum()
             return self.orthogonal_reg_weight * loss
     
         def forward(self, x):
             if x.dim() not in [3, 4]:
                 raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
     
             batch_size, seq_len, *rest = x.size()
     
             if x.dim() == 3:
                 n_state = rest[0]
                 if n_state != self.n_head * self.h_dim:
                     raise ValueError(
                         f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim}={self.n_head * self.h_dim})")
             else:
                 n_head, h_dim = rest
                 if n_head != self.n_head or h_dim != self.h_dim:
                     raise ValueError(
                         f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
     
             x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
             x = x.reshape(-1, self.h_dim)
             adjusted_n_rots = int(torch.round(self.n_rots_scale * self.n_rots))
     
             for k in range(adjusted_n_rots):
                 i, j = self.r_pairs[k].long()
                 theta = self.thetas[k] * self.theta_scale
                 G = self.givens_r_matrix(n_state=self.h_dim, i=i, j=j, theta=theta)
                 x = torch.matmul(input=x, other=G)
     
             x = torch.matmul(input=x, other=self.r_matrix)
             x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
     
             sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device),
                                          self.inv_freq.to(device=x.device))
             sin = sinusoid_inp.sin()[None, :, None, :]
             cos = sinusoid_inp.cos()[None, :, None, :]
     
             x1, x2 = x[..., ::2], x[..., 1::2]
             x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
             x = x.view(batch_size, seq_len, self.n_state)
     
             return x
         
         # loss += embedding_layer.orthogonal_regularization_term()
