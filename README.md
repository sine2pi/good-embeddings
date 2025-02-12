### A mix of:
   ####  Givens Rotation 1/3
   ####  Householder Rotation 1/3
   ####  Orthogonal Rotation  1/3

#### Efficient-ish (relative to the original) version:
(Soon to be dynamically and automatically controlled by the model.)
     
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
     
     class CombinedRotaryEmbedding(nn.Module):
         def __init__(self, base, dims, head, rotation_type='givens', theta_learnable=True,
                      rot_learnable=True, matrix_learnable=False, freq_learnable=True):
             super(CombinedRotaryEmbedding, self).__init__()
             self.base = base
             self.dims = dims
             self.head = head
             self.rotation_type = rotation_type
     
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
     
             if self.rotation_type == 'givens':
                 self.rotation_function = self.givens_rotation
             elif self.rotation_type == 'householder':
                 self.rotation_function = self.householder_rotation
             elif self.rotation_type == 'orthogonal':
                 self.rotation_function = self.orthogonal_rotation
             else:
                 raise ValueError('Invalid rotation type')
     
         def givens_r_matrix(self, dims, i, j, theta):
             G = torch.eye(dims).to(theta.device)
             G[i, i] = torch.cos(theta)
             G[i, j] = -torch.sin(theta)
             G[j, i] = torch.sin(theta)
             G[j, j] = torch.cos(theta)
             return G
     
         def givens_rotation(self, x):
             adjusted_rot = int(torch.round(self.rot_scale * self.rot))
             for k in range(adjusted_rot):
                 i, j = self.r_pairs[k].long()
                 theta = self.thetas[k] * self.theta_scale
                 G = self.givens_r_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
                 x = torch.matmul(input=x, other=G)
             return x
     
         def householder_rotation(self, x):
             adjusted_rot = int(torch.round(self.rot_scale * self.rot))
             for k in range(adjusted_rot):
                 i, j = self.r_pairs[k].long()
                 theta = self.thetas[k] * self.theta_scale
                 v = torch.zeros(self.h_dim).to(theta.device)
                 v[i] = torch.cos(theta)
                 v[j] = torch.sin(theta)
                 H = torch.eye(self.h_dim).to(theta.device) - 2 * torch.outer(v, v) / torch.dot(v, v)
                 x = torch.matmul(input=x, other=H)
             return x
     
         def orthogonal_rotation(self, x):
             adjusted_rot = int(torch.round(self.rot_scale * self.rot))
             for k in range(adjusted_rot):
                 i, j = self.r_pairs[k].long()
                 theta = self.thetas[k] * self.theta_scale
                 R = torch.eye(self.h_dim).to(theta.device)
                 R[i, i] = torch.cos(theta)
                 R[i, j] = -torch.sin(theta)
                 R[j, i] = torch.sin(theta)
                 R[j, j] = torch.cos(theta)
                 x = torch.matmul(input=x, other=R)
             return x
     
         def update_base(self, new_base):
             if new_base is not None and new_base!= self.base:
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
                 if i!= j and (i, j) not in pairs and (j, i) not in pairs:
                     pairs.append((i, j))
             self.r_pairs.data.copy_(torch.tensor(pairs, dtype=torch.float32))
     
         def forward(self, x, global_step=None):
             if x.dim() not in [3, 4]:
                 raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
     
             batch_size, seq_len, *rest = x.size()
     
             if x.dim() == 3:
                 dims = rest[0]
                 if dims!= self.head * self.h_dim:
                     raise ValueError(
                         f"Expected dims ({dims}) to be compatible with head ({self.head}) * h_dim ({self.h_dim}={self.head * self.h_dim})")
             else:
                 head, h_dim = rest
                 if head!= self.head or h_dim!= self.h_dim:
                     raise ValueError(
                         f"For 4D input, expected head {self.head} and h_dim {self.h_dim}, but got head {head} and h_dim {h_dim}")
     
             x = x.view(batch_size, seq_len, self.head, self.h_dim)
             x = x.reshape(-1, self.h_dim)
     
             x = self.rotation_function(x)
     
             x = torch.matmul(input=x, other=self.r_matrix)
     
             x = x.view(batch_size, seq_len, self.head, self.h_dim)
     
             sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device),
                                          self.inv_freq.to(device=x.device))
             sin = sinusoid_inp.sin()[None, :, None, :]
             cos = sinusoid_inp.cos()[None, :, None, :]
     
             x1, x2 = x[..., ::2], x[..., 1::2]
             x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
             x = x.view(batch_size, seq_len, self.dims)
     
             return x
     
     class MultiRotationLayer(nn.Module):
         def __init__(self, input_dim, output_dim, num_scales=3):
             super(MultiRotationLayer, self).__init__()
             self.num_scales = num_scales
             self.rotations = nn.ModuleList([self._create_rotation(input_dim // (2**i)) for i in range(num_scales)])
             self.scale = nn.Parameter(torch.ones(num_scales))
     
         def _create_rotation(self, input_dim):
             return nn.Sequential(
                 GivensRotation(input_dim // 3),
                 HouseholderRotation(input_dim // 3),
                 OrthogonalRotation(input_dim // 3)
             )
     
         def forward(self, x):
             outputs = []
             for i, rotation in enumerate(self.rotations):
                 output = rotation(x[:, ::(2**i)])
                 outputs.append(output)
             x = torch.cat(outputs, dim=1)
             return x
     
         def update(self, loss):
             # Calculate the scale of the rotations based on the loss
             self.scale.data = 1 / (1 + torch.exp(-loss))
     
             # Update the rotations based on the scale
             for i, rotation in enumerate(self.rotations):
                 rotation.scale = self.scale[i]
     
         def get_scale(self):
             return self.scale
