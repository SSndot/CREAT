import sys
import torch

from utils import *
from tqdm import tqdm
import ot
from matplotlib import pyplot as plt

class PositionEncoder(nn.Module):
    def __init__(self, max_seq_len, pos_emb_size):
        super(PositionEncoder, self).__init__()
        self.pos_emb_size = pos_emb_size
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, pos_emb_size)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, x):
        pos = torch.arange(self.max_seq_len).unsqueeze(0).expand(x.size(0), -1).to(x.device)
        return self.pos_emb(pos)


class GeneratorModel(nn.Module):
    def __init__(self, seq_emb_size, item_emb_size, hidden_size, max_seq_len, pos_emb_size=64, dropout_rate=0.1, num_layers=2, device='cpu'):
        super(GeneratorModel, self).__init__()
        self.num_layers = num_layers
        inp_size = seq_emb_size + item_emb_size + pos_emb_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(inp_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.pos_embedding = PositionEncoder(max_seq_len, pos_emb_size)

        self.device = device

    def forward(self, seq_emb, item_emb, mask):
        batch_size, seq_len, item_emb_dim = item_emb.shape
        seq_emb = seq_emb.unsqueeze(1).expand(-1, seq_len, -1)
        pos_emb = self.pos_embedding(seq_emb)
        inp_emb = torch.cat([seq_emb, item_emb, pos_emb], dim=2)    # batch_size x seq_len x (seq_emb + item_emb + pos_emb)

        hidden = torch.zeros(self.num_layers, seq_emb.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(inp_emb, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.squeeze(-1)
        out = out.masked_fill(mask == 0, float('-inf'))
        out = F.softmax(out, dim=-1)
        return out


class Generator:
    def __init__(self, model, model_helper, max_attack_num, target_item, max_seq_len, log=None, device='cpu'):
        self.model = model
        self.model_helper = model_helper
        self.max_attack_num = max_attack_num

        self.target_item = target_item
        self.max_seq_len = max_seq_len

        self.log = log
        self.device = device

        self.batch_size = 64

        self.group_num = 10
        self.history_seqs, self.history_acts, self.history_probs, self.history_rewards, self.history_constraints = None, None, None, None, None

        self.clip_epsilon = 0.2

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)

    def get_group(self, base_seqs, is_adv=False):
        self.model.eval()
        self.history_seqs, self.history_acts, self.history_probs, self.history_rewards, self.history_constraints = [], [], [], [], []
        for _ in tqdm(range(self.group_num)):
            hist_seq, hist_act, hist_prob, hist_reward, hist_con = [], [], [], [], []
            seqs_len = get_lengths(base_seqs)
            attack_seqs = base_seqs.clone()
            attack_seqs = attack_seqs.to(self.device)
            seqs_len = seqs_len.to(self.device)
            padding_mask = self.gen_padding_mask(seqs_len)

            for i in range(self.max_attack_num):
                gen_seq_emb, _ = self.model_helper(attack_seqs, grads=False)
                gen_item_emb, _ = self.model_helper.model.embedding(attack_seqs)
                gen_seq_emb = gen_seq_emb.detach().to(self.device)
                gen_item_emb = gen_item_emb.detach().to(self.device)
                attacked_mask = self.gen_attacked_mask(attack_seqs)
                mask = torch.logical_and(padding_mask, attacked_mask)
                out = self.model(gen_seq_emb, gen_item_emb, mask)
                target_pos = torch.multinomial(out, 1)
                probs = torch.gather(out, dim=1, index=target_pos)
                tmp_seqs = attack_seqs.clone()
                attack_seqs.scatter_(1, target_pos, self.target_item)

                pattern_rewards = self.get_pattern_reward(attack_seqs, seqs_len)
                dpp_rewards = self.get_dpp_reward(attack_seqs, seqs_len)
                total_rewards = pattern_rewards + dpp_rewards

                hist_seq.append(tmp_seqs.detach())
                hist_act.append(target_pos.detach())
                hist_prob.append(probs.detach())
                hist_reward.append(total_rewards.detach())

                if is_adv:
                    ot_rewards = self.get_ot_reward(tmp_seqs, attack_seqs)
                    hist_con.append(ot_rewards.detach())

            self.history_seqs.append(torch.stack(hist_seq, dim=0))
            self.history_acts.append(torch.stack(hist_act, dim=0))
            self.history_probs.append(torch.stack(hist_prob, dim=0))
            self.history_rewards.append(torch.stack(hist_reward, dim=0))

            if is_adv:
                self.history_constraints.append(torch.stack(hist_con, dim=0))

        self.history_seqs = torch.stack(self.history_seqs, dim=0)
        self.history_acts = torch.stack(self.history_acts, dim=0)
        self.history_probs = torch.stack(self.history_probs, dim=0)
        self.history_rewards = torch.stack(self.history_rewards, dim=0)
        self.history_rewards = get_grpo_rewards(self.history_rewards)

        if is_adv:
            self.history_constraints = torch.stack(self.history_constraints, dim=0)
            self.history_constraints = get_grpo_rewards(self.history_constraints)

    def cal_lambda(self, reward, constraint, target_constraint, eps=1e-8):

        grad_g = torch.autograd.grad(constraint, self.model.parameters(),
                                     retain_graph=True,
                                     create_graph=False)

        grad_f = torch.autograd.grad(reward, self.model.parameters(),
                                     retain_graph=True,
                                     create_graph=False)

        dot_product = sum(torch.sum(gf * gg) for gf, gg in zip(grad_f, grad_g))
        grad_g_norm_sq = sum(torch.sum(gg ** 2) for gg in grad_g)

        phi = min(constraint-target_constraint, grad_g_norm_sq)
        numerator = phi - dot_product
        denominator = grad_g_norm_sq + eps
        lambda_t = torch.clamp(numerator / denominator, min=0.0)

        return lambda_t.detach().item()

    def train(self, k=5, is_adv=False):
        self.model.train()
        loss, reward = 0, 0
        for i in range(self.group_num):
            for j in range(self.max_attack_num):
                with torch.no_grad():
                    base_seqs = self.history_seqs[i][j]
                    seqs_len = get_lengths(base_seqs)
                    attack_seqs = base_seqs.clone()
                    attack_seqs = attack_seqs.to(self.device)
                    seqs_len = seqs_len.to(self.device)
                    padding_mask = self.gen_padding_mask(seqs_len)

                    gen_seq_emb, _ = self.model_helper(attack_seqs, grads=False)
                    gen_item_emb, _ = self.model_helper.model.embedding(attack_seqs)
                    gen_seq_emb = gen_seq_emb.detach().to(self.device)
                    gen_item_emb = gen_item_emb.detach().to(self.device)
                    attacked_mask = self.gen_attacked_mask(attack_seqs)
                    mask = torch.logical_and(padding_mask, attacked_mask)

                out = self.model(gen_seq_emb, gen_item_emb, mask)
                old_pos = self.history_acts[i][j]
                probs = torch.gather(out, dim=1, index=old_pos)

                r_probs = probs / self.history_probs[i][j]
                clip_r_probs = torch.clamp(r_probs, min=1-self.clip_epsilon, max=1+self.clip_epsilon)
                his_r = self.history_rewards[i][j]
                l = -torch.min(r_probs * his_r, clip_r_probs * his_r)

                if is_adv:
                    his_r = self.history_constraints[i][j]
                    l_c = torch.max(r_probs * his_r, clip_r_probs * his_r)
                    values, _ = torch.kthvalue(
                        self.history_constraints[:, j, :, :].squeeze(),
                        k=k,
                        dim=0,
                        keepdim=True
                    )
                    target_c = torch.max(r_probs * values.T, clip_r_probs * values.T)
                    lambda_t = self.cal_lambda(l.mean(), l_c.mean(), target_c.mean())
                    loss += torch.mean(l + lambda_t * l_c)

                else:
                    loss += torch.mean(l)

        self.optimizer.zero_grad()
        loss = loss / (self.max_attack_num * self.group_num)
        loss.backward()
        self.optimizer.step()

    def old_train(self):
        self.model.train()
        loss, reward = 0, 0
        for i in range(self.group_num):
            for j in range(self.max_attack_num):
                with torch.no_grad():
                    base_seqs = self.history_seqs[i][j]
                    seqs_len = get_lengths(base_seqs)
                    attack_seqs = base_seqs.clone()
                    attack_seqs = attack_seqs.to(self.device)
                    seqs_len = seqs_len.to(self.device)
                    padding_mask = self.gen_padding_mask(seqs_len)

                    gen_seq_emb, _ = self.model_helper(attack_seqs, grads=False)
                    gen_item_emb, _ = self.model_helper.model.embedding(attack_seqs)
                    gen_seq_emb = gen_seq_emb.detach().to(self.device)
                    gen_item_emb = gen_item_emb.detach().to(self.device)
                    attacked_mask = self.gen_attacked_mask(attack_seqs)
                    mask = torch.logical_and(padding_mask, attacked_mask)

                out = self.model(gen_seq_emb, gen_item_emb, mask)
                old_pos = self.history_acts[i][j]
                probs = torch.gather(out, dim=1, index=old_pos)

                his_r = self.history_rewards[i][j]
                l = probs * his_r
                loss += torch.mean(l)

        self.optimizer.zero_grad()
        loss = loss / (self.max_attack_num * self.group_num)
        loss.backward()
        self.optimizer.step()

    def test(self, base_seqs):
        self.model.eval()
        seqs_len = get_lengths(base_seqs)
        attack_seqs = base_seqs.clone()
        attack_seqs = attack_seqs.to(self.device)
        seqs_len = seqs_len.to(self.device)
        padding_mask = self.gen_padding_mask(seqs_len)

        pattern_rewards, dpp_rewards, ot_rewards = 0, 0, 0
        for _ in range(self.max_attack_num):
            gen_seq_emb, _ = self.model_helper(attack_seqs, grads=False)
            gen_item_emb, _ = self.model_helper.model.embedding(attack_seqs)
            gen_seq_emb = gen_seq_emb.detach().to(self.device)
            gen_item_emb = gen_item_emb.detach().to(self.device)
            attacked_mask = self.gen_attacked_mask(attack_seqs)
            mask = torch.logical_and(padding_mask, attacked_mask)
            out = self.model(gen_seq_emb, gen_item_emb, mask)
            target_pos = torch.argmax(out, dim=-1).view(-1, 1)
            attack_seqs.scatter_(1, target_pos, self.target_item)

        p_rewards = self.get_pattern_reward(attack_seqs, seqs_len)
        d_rewards = self.get_dpp_reward(attack_seqs, seqs_len)
        o_rewards = self.get_ot_reward(base_seqs, attack_seqs)

        pattern_rewards += p_rewards.detach().mean()
        dpp_rewards += d_rewards.detach().mean()
        ot_rewards += o_rewards.detach().mean()
        return pattern_rewards, dpp_rewards, ot_rewards

    def gen_padding_mask(self, seq_len):
        max_valid_seq_len = self.max_seq_len
        range_tensor = torch.arange(max_valid_seq_len, device=seq_len.device).unsqueeze(0).expand(seq_len.size(0), -1)
        seq_len_expanded = seq_len.unsqueeze(1)
        mask = range_tensor < seq_len_expanded
        return mask

    def gen_attacked_mask(self, attack_seqs):
        mask = ~torch.eq(attack_seqs, self.target_item)
        return mask

    def get_emb_dist(self, seq):
        if len(seq) == 0: return 0
        seq = t_padding(seq, self.max_seq_len).unsqueeze(0) # 1 x emb_size
        seq_emb, _ = self.model_helper(seq)
        item_index = torch.tensor([self.target_item]).to(self.device)
        item_emb = self.model_helper.model.embedding.token(item_index)
        if self.model_helper.model_name == 'narm':
            item_emb = self.model_helper.model.model.b_vetor(item_emb)
        dist = F.cosine_similarity(seq_emb, item_emb)
        return (1 - dist) / 2

    def get_reverse_dist(self, seq, pos, seq_len):
        dist = 0
        for p in pos:
            front_seq = seq[:p]
            front_dist = self.get_emb_dist(front_seq)
            back_seq = seq[p+1:seq_len]
            back_dist = self.get_emb_dist(back_seq)
            dist += front_dist + back_dist
        return dist

    def get_pattern_reward(self, seqs, seqs_valid_len):
        rewards = []
        for idx, seq in enumerate(seqs):
            pos = torch.nonzero(seq == self.target_item).squeeze(1).tolist()
            seq_len = seqs_valid_len[idx]
            dist = self.get_reverse_dist(seq, pos, seq_len)
            reward = dist
            rewards.append(reward)
        rewards = torch.stack(rewards, dim=0)
        return rewards.detach()

    def get_dpp(self, seq, pos, seq_len, eps=1e-8):
        all_seqs = []
        dpp = 0
        lp = 0
        pos += [seq_len]
        for p in pos:
            s = seq[lp:p]
            if len(s) != 0: all_seqs.append(t_padding(s, self.max_seq_len))
            lp = p

        if len(all_seqs) != 0:
            seqs = torch.stack(all_seqs, dim=0).to(self.device)
            seqs_emb, _ = self.model_helper(seqs)
            norms = torch.norm(seqs_emb, p=2, dim=-1, keepdim=True)
            norms = torch.clamp(norms, min=eps)
            norm_seqs_emb = seqs_emb / norms
            kernel = torch.matmul(norm_seqs_emb, norm_seqs_emb.transpose(0, 1))
            dpp = torch.det(kernel)

        return dpp

    def get_dpp_reward(self, seqs, seqs_valid_len):
        rewards = []
        for idx, seq in enumerate(seqs):
            pos = torch.nonzero(seq == self.target_item).squeeze(1).tolist()
            seq_len = seqs_valid_len[idx]
            dpp = self.get_dpp(seq, pos, seq_len)
            reward = dpp
            rewards.append(reward)
        rewards = torch.stack(rewards, dim=0).unsqueeze(-1)
        return rewards.detach()

    def get_ot_reward(self, prev_seqs, curr_seqs):
        prev_seq_emb, _ = self.model_helper(prev_seqs, grads=False)
        curr_seq_emb, _ = self.model_helper(curr_seqs, grads=False)

        prev_seq_emb_np = prev_seq_emb.cpu().numpy()
        curr_seq_emb_np = curr_seq_emb.cpu().numpy()

        M_s = ot.dist(curr_seq_emb_np, prev_seq_emb_np, metric='euclidean')
        M_f = ot.dist(curr_seq_emb_np.T, prev_seq_emb_np.T, metric='euclidean')

        P, Q = ot.gromov.unbalanced_co_optimal_transport(curr_seq_emb_np, prev_seq_emb_np, M_samp=M_s, M_feat=M_f,
                                                         max_iter=20)
        P = torch.Tensor(P).to(self.device)
        Q = torch.Tensor(Q).to(self.device)
        transfer_seq_emb = torch.matmul(P.T, curr_seq_emb)
        transfer_seq_emb = torch.matmul(transfer_seq_emb, Q)
        diff = transfer_seq_emb - curr_seq_emb
        rewards = torch.norm(diff, p=2, dim=-1)
        return rewards.unsqueeze(-1).detach()

    def gen_all_attack_seqs(self, material_seqs, batch_size=256):
        base_seqs = material_seqs.clone()
        train_len = len(base_seqs)
        all_attack_seqs = None
        for i in range(0, train_len, batch_size):
            batch_base_seqs = base_seqs[i:i+batch_size]
            batch_attack_seqs = self.gen_attack_seqs(batch_base_seqs)
            if all_attack_seqs is None:
                all_attack_seqs = batch_attack_seqs
            else:
                all_attack_seqs = torch.concat([all_attack_seqs, batch_attack_seqs], dim=0)
        return all_attack_seqs

    def gen_attack_seqs(self, base_seqs):
        seq_len = get_lengths(base_seqs)
        attack_seqs = base_seqs.clone()
        attack_seqs = attack_seqs.to(self.device)
        seq_len = seq_len.to(self.device)
        padding_mask = self.gen_padding_mask(seq_len)

        for i in range(self.max_attack_num):
            gen_seq_emb, _ = self.model_helper(attack_seqs, grads=False)
            gen_item_emb, _ = self.model_helper.model.embedding(attack_seqs)
            gen_seq_emb = gen_seq_emb.detach().to(self.device)
            gen_item_emb = gen_item_emb.detach().to(self.device)
            attacked_mask = self.gen_attacked_mask(attack_seqs)
            mask = torch.logical_and(padding_mask, attacked_mask)
            out = self.model(gen_seq_emb, gen_item_emb, mask)
            target_pos = torch.argmax(out, dim=-1).view(-1, 1)
            attack_seqs.scatter_(1, target_pos, self.target_item)

        return attack_seqs

