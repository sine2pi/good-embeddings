#### This enbedding system is set for 3 dimensions. You can edit it for more or less. It relies on the base frequency updates via loss from your models forward pass. 
##### These go in your model class : 

     def adjust_base(self, loss, factor=1.0025):  ## optionally adjust to control fine grain updates -- it's not elegant.
         if self.adjust_counter % 25 == 0:  ## updates can be 10x or more frequent than steps
             if loss < self.best_loss:
                 new_base = self.base * factor
             else:
                 new_base = self.base / factor
             self.update_base(new_base)
             self.base = new_base
             self.best_loss = loss
   
         self.adjust_counter += 1
         return self.base

     def update_base(self, new_base):
         self.new_base=new_base
         for name, module in self.encoder.named_modules():
             if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding, AudioEncoder)):
                 module.update_base(self.new_base)
         for name, module in self.decoder.named_modules():
             if isinstance(module, (MultiheadAttention, CombinedRotaryEmbedding)):
                 module.update_base(self.new_base)

     def print_update(self):
         self.adjust_counter += 1
         if self.adjust_counter % 20 == 0:
             print(f"{self.adjust_counter}: Loss: {self.best_loss}  Base: {self.base}, Distance: {self.max_dist}")

##### These go in your models forward pass:

             self.adjust_base(loss.item())
             self.adjust_max_dist(loss.item())
             self.print_update()  

##### Example forward pass : 

     def forward(self, input_features, labels=None, dec_input_ids=None):
         if labels is not None:
             if dec_input_ids is None:
                 dec_input_ids = self.shift_tokens_right(
                     labels, self.config.pad_token_id, self.config.decoder_start_token_id)

         encoded_features = self.encoder(input_features).to(self.device)  
         logits = self.decoder(dec_input_ids, encoded_features)

         loss = None
         if labels is not None:
             loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
             labels = labels.to(logits.device).long()
             loss = loss_fct(logits.view(-1, self.config.n_vocab), labels.view(-1))

             self.adjust_base(loss.item())  ###
             self.adjust_max_dist(loss.item())  ###
             self.print_update()  ###

         return {"loss": loss, "logits": logits}

##### Rotary block

    class CombinedRotaryEmbedding(nn.Module):
        def __init__(self, base, n_state, n_head):
            super().__init__()
            self.base = base
            self.n_state = n_state
            self.n_head = n_head
            assert self.n_state % self.n_head == 0, "n_state must be divisible by n_head"
            self.h_dim = self.n_state // self.n_head
            assert self.h_dim % 2 == 0, "Head dimension must be even for rotary embeddings"
            self.num_rotations = ((n_state // n_head) // 2)
        
            self.thetas = nn.Parameter(torch.zeros(self.num_rotations)) 
            self.rotation_pairs = nn.Parameter(data=torch.rand(self.num_rotations, 2) * self.h_dim)
            self.theta_scale = nn.Parameter(data=torch.ones(1))
            self.rotation_matrix = nn.Parameter(data=torch.eye(n=self.h_dim))
            self.inv_freq = nn.Parameter(data=1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim)))
            self.num_rotations_scale = nn.Parameter(data=torch.ones(1))
    
        def givens_rotation_matrix(self, n_state, i, j, theta):
            G = torch.eye(n_state).to(device)
            G[i, i] = math.cos(theta)
            G[i, j] = -math.sin(theta)
            G[j, i] = math.sin(theta)
            G[j, j] = math.cos(theta)
            return G
            
        def update_base(self, new_base):
            if new_base is not None and new_base != self.base:
                self.base = new_base
                inv_freq = 1.0 / (self.base ** (torch.arange(start=0, end=self.h_dim, step=2).float() / self.h_dim))
                self.inv_freq.data.copy_(inv_freq)
    
        def reset_parameters(self):
            nn.init.orthogonal_(tensor=self.rotation_matrix)
            nn.init.zeros_(tensor=self.thetas)
    
        def forward(self, x, new_base=None):
            self.update_base(new_base) 
    
            if x.dim() not in [3, 4]:
                raise ValueError(f"Expected input tensor to be 3D or 4D, but got {x.dim()}D")
    
            batch_size, seq_len, *rest = x.size()
    
            if x.dim() == 3:
                n_state = rest[0]
                if n_state != self.n_head * self.h_dim:
                    raise ValueError(f"Expected n_state ({n_state}) to be compatible with n_head ({self.n_head}) * h_dim ({self.h_dim} = {self.n_head * self.h_dim})") 
            else: 
                n_head, h_dim = rest
                if n_head != self.n_head or h_dim != self.h_dim:
                    raise ValueError(f"For 4D input, expected n_head {self.n_head} and h_dim {self.h_dim}, but got n_head {n_head} and h_dim {h_dim}")
    
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim) 
            x = x.reshape(-1, self.h_dim)
            adjusted_num_rotations = int(torch.round(self.num_rotations * self.num_rotations_scale))
    
            for k in range(adjusted_num_rotations):
                i, j = self.rotation_pairs[k].long()
                theta = self.thetas[k] * self.theta_scale
                G = self.givens_rotation_matrix(n_state=self.h_dim, i=i, j=j, theta=theta).to(device)
                x = torch.matmul(input=x, other=G).to(device)
    
            x = torch.matmul(input=x, other=self.rotation_matrix)
            x = x.view(batch_size, seq_len, self.n_head, self.h_dim)
    
            sinusoid_inp = torch.einsum('i, j -> i j', torch.arange(end=seq_len, device=x.device), self.inv_freq.to(device=x.device))
            sin = sinusoid_inp.sin()[None, :, None, :]
            cos = sinusoid_inp.cos()[None, :, None, :]
    
            x1, x2 = x[..., ::2], x[..., 1::2]
            x = torch.cat(tensors=[x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            x = x.view(batch_size, seq_len, self.n_state)
            return x
