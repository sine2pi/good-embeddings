### Fun with embeddings!  Choose your own rotation type examples : rotation_type='givens'   or household or orthogonal or quaternion. or blend them. Drop in replacment for your standard Facebook rope. Standard embedding take 100 steps to memorize 1 sentence. RoPe takes 50. This takes 5. Good embeddings! (Except for the Householder one that ones pretty bad.)

#### Usage:
####               self.rotary = MixedTape(base=10000, n_state=n_state, n_head=n_head, rotation_type='mixed_tape', theta_learnable=False, rot_learnable=False, matrix_learnable=False, freq_learnable=False)

#### Now with less Housholder!


<img width="683" alt="123" src="https://github.com/user-attachments/assets/10fa2ecd-8aec-46e2-86f1-9e3d48f6f398" />

<img width="388" alt="legend" src="https://github.com/user-attachments/assets/0285b11d-308c-4c72-8084-5dc6fc2eb5ed" />

            
      class MixedTape(nn.Module):
          def __init__(self, base, n_state, n_head, rotation_type='givens', theta_learnable=False,
                       rot_learnable=False, matrix_learnable=False, freq_learnable=False):
              super(MixedTape, self).__init__()
              self.base = base
              self.n_state = n_state
              self.n_head = n_head
              self.rotation_type = rotation_type
      
              self.h_dim = self.n_state // self.n_head
              self.rot = (self.n_state // self.n_head) // 2
      
              self.thetas = nn.Parameter(torch.zeros(self.rot))
              self.r_pairs = nn.Parameter(data=torch.rand(self.rot, 2) * self.h_dim)
      
              self.theta_scale = nn.Parameter(torch.ones(1), requires_grad=theta_learnable)
              self.rot_scale = nn.Parameter(torch.ones(1), requires_grad=rot_learnable)
      
              self.r_matrix = nn.Parameter(torch.eye(n=self.h_dim), requires_grad=matrix_learnable)
      
              freq_data = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
              self.inv_freq = nn.Parameter(freq_data, requires_grad=freq_learnable)
      
              self.orthogonal_reg_weight = 0.01
      
              if self.rotation_type == 'givens':
                  self.rotation_function = self.givens
              elif self.rotation_type == 'orthogonal':
                  self.rotation_function = self.orthogonal
              elif self.rotation_type == 'mixed_tape':
                  self.rotation_function = self.apply_blended_rotation
              else:
                  raise ValueError('Invalid rotation type')
      
          def quaternion_rotation(self, x, theta, u, v):
              u = u / torch.norm(u)
              v = v / torch.norm(v)
              cos_theta = torch.cos(theta / 2)
              sin_theta = torch.sin(theta / 2)
      
              q = torch.tensor([cos_theta, sin_theta * u[0], sin_theta * u[1], sin_theta * u[2]], device=x.device)
              q_conjugate = torch.tensor([cos_theta, -sin_theta * u[0], -sin_theta * u[1], -sin_theta * u[2]], device=x.device)
      
              x_shape = x.shape
              x = x.view(-1, 3)
      
              uv_cross = torch.cross(u.unsqueeze(0), x)
              uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
              x_rot = x + 2 * (q[0] * uv_cross + uuv_cross)
              
              x_rot = x_rot.view(*x_shape)
              return x_rot
      
          def blended_rotation_matrix(self, dims, i, j, theta):
              G = torch.eye(dims).to(theta.device)
              G[i, i] = torch.cos(theta)
              G[i, j] = -torch.sin(theta)
              G[j, i] = torch.sin(theta)
              G[j, j] = torch.cos(theta)
      
              u = torch.eye(dims).to(theta.device)[i]
              v = torch.eye(dims).to(theta.device)[j]
      
              if dims == 3:
                  Q = self.quaternion_rotation(x=torch.eye(dims).to(theta.device), theta=theta, u=u, v=v)
                  return (G + Q) / 2
              return G
      
          def apply_blended_rotation(self, x):
              adjusted_rot = int(torch.round(self.rot_scale * self.rot))
              for k in range(adjusted_rot):
                  i, j = self.r_pairs[k].long()
                  theta = self.thetas[k] * self.theta_scale
                  B = self.blended_rotation_matrix(dims=self.h_dim, i=i, j=j, theta=theta)
                  x = torch.matmul(x, B)
              return x
      
          def givens_r_matrix(self, n_state, i, j, theta):
              G = torch.eye(n_state).to(theta.device)
              G[i, i] = torch.cos(theta)
              G[i, j] = -torch.sin(theta)
              G[j, i] = torch.sin(theta)
              G[j, j] = torch.cos(theta)
              return G
      
          def givens(self, x):
              adjusted_rot = int(torch.round(self.rot_scale * self.rot))
              for k in range(adjusted_rot):
                  i, j = self.r_pairs[k].long()
                  theta = self.thetas[k] * self.theta_scale
                  G = self.givens_r_matrix(n_state=self.h_dim, i=i, j=j, theta=theta)
                  x = torch.matmul(input=x, other=G)
              return x
      
          def householder(self, x):
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
      
          def orthogonal(self, x):
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
                  n_state = rest[0]
                  if n_state!= self.n_head * self.h_dim:
                      raise ValueError(
                          f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim}={self.n_head * self.h_dim})")
              else:
                  n_head, h_dim = rest
                  if n_head!= self.n_head or h_dim!= self.h_dim:
                      raise ValueError(
                          f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
      
              x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
              x = x.reshape(-1, self.h_dim)
              x = self.rotation_function(x)
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
          
