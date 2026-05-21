从 MoT 入口的两路 token 开始：

x_v  : [B, S_v,  D_v]   D_v=3072
x_a  : [B, T_act, D_a]   D_a=1024
ctx_v: [B, L,    D_v]    # video pre_dit 里 text_embedding 后的 context
ctx_a: [B, L,    D_a]    # action pre_dit 里同理
freqs_v: [S_v, 1, rope_dim]
freqs_a: [T_act, 1, rope_dim]
t_mod_v, t_mod_a         # 各层 time modulation
mask: [S_v+T_act, S_v+T_act]  # joint self-attn 掩码

────────────────────────────────────────

MoT 外层循环（30 层，每层结构相同）

```python
for layer in range(30):
    block_v = video_expert.blocks[layer]   # 参数空间 D_v=3072
    block_a = action_expert.blocks[layer]  # 参数空间 D_a=1024
    # ── Step 1: 各自准备 Q/K/V（还没互相看见）──
    # video 支
    shift, scale, gate_msa, ..., gate_mlp = split_modulation(block_v, t_mod_v)
    h_v = modulate(LayerNorm(x_v), shift, scale)          # [B, S_v, D_v]
    Q_v = RoPE(RMSNorm(block_v.self_attn.q(h_v)))         # [B, S_v, 3072]
    K_v = RoPE(RMSNorm(block_v.self_attn.k(h_v)))         # [B, S_v, 3072]
    V_v =      block_v.self_attn.v(h_v)                    # [B, S_v, 3072]
    # 3072 = num_heads(24) × attn_head_dim(128)，两边相同
    # action 支（完全对称，只是 D_a=1024 作为 block 输入维）
    h_a = modulate(LayerNorm(x_a), ...)
    Q_a = RoPE(RMSNorm(block_a.self_attn.q(h_a)))         # [B, T_act, 3072]
    K_a = RoPE(RMSNorm(block_a.self_attn.k(h_a)))         # [B, T_act, 3072]
    V_a =      block_a.self_attn.v(h_a)                    # [B, T_act, 3072]
    # ── Step 2: concat → 一次 joint self-attention ──
    Q = cat([Q_v, Q_a], dim=1)   # [B, S_v+T_act, 3072]
    K = cat([K_v, K_a], dim=1)
    V = cat([V_v, V_a], dim=1)
    mixed = flash_attention(Q, K, V, mask=mask)  # [B, S_v+T_act, 3072]
    mixed_v = mixed[:, :S_v,     :]   # [B, S_v,  3072]
    mixed_a = mixed[:, S_v:,      :]   # [B, T_act, 3072]
    # ── Step 3: 切回两支，各自做 post-block（不再 joint）──
    # video post
    x_v = gate(x_v, gate_msa, block_v.self_attn.o(mixed_v))     # o: 3072→D_v, [B,S_v,D_v]
    x_v = x_v + block_v.cross_attn(LayerNorm(x_v), ctx_v, ...)  # [B,S_v,D_v]
    x_v = gate(x_v, gate_mlp, block_v.ffn(modulate(LayerNorm(x_v), ...)))
    # ffn: D_v(3072) → 14336 → D_v
    # action post（同一逻辑，维度换成 D_a）
    x_a = gate(x_a, gate_msa, block_a.self_attn.o(mixed_a))     # o: 3072→D_a, [B,T_act,D_a]
    x_a = x_a + block_a.cross_attn(LayerNorm(x_a), ctx_a, ...)  # [B,T_act,D_a]
    x_a = gate(x_a, gate_mlp, block_a.ffn(...))                  # D_a(1024)→4096→D_a
# 30 层后
tokens_out_v = x_v   # [B, S_v,  D_v]
tokens_out_a = x_a   # [B, T_act, D_a]
```

────────────────────────────────────────

mask 控制 joint attn 谁能看谁

mask = zeros(S_v + T_act, S_v + T_act)
# video ↔ video：按 first_frame_causal 等规则（video 内部）
mask[:S_v, :S_v] = video_to_video_mask
# action ↔ action：全互看
mask[S_v:, S_v:] = True
# action → video：只能看首帧 video tokens
mask[S_v:, :tokens_per_frame] = True
# video → action：False（video 看不到 action）

────────────────────────────────────────

MoT 出口 → post_dit

# video
pred_v_latent = video_expert.post_dit(tokens_out_v, video_pre)
#   head:   [B, S_v, D_v] → [B, S_v, C*patch_vol]
#   unpatchify → [B, C, T, H, W]
# action
pred_action = action_expert.post_dit(tokens_out_a, action_pre)
#   head: [B, T_act, D_a] → [B, T_act, A]

────────────────────────────────────────

形状变化一览

[B,S_v,3072] ─┐
              ├─ cat ─→ joint attn ─→ split ─→ o_proj ─→ [B,S_v,3072]
[B,T_act,3072]┘                                              ↓ cross+ffn
                                                         [B,S_v,3072] ×30层
[B,T_act,1024] 输入 block_a，Q/K/V 仍投到 3072，o_proj 投回 1024

核心： embedding 维不同（3072 vs 1024），但 Q/K/V 统一到 24×128=3072 做 joint attn；attn 输出再经各自的 o_proj 回到各自 hidden
dim，cross-attn 和 FFN 也各用各的 block 参数。

---

1. t_mod_v 是什么

扩散时间步的调制信号，告诉每一层 DiT block 当前噪声水平，用来 AdaLN 式地 shift/scale/gate。

在 video pre_dit 里（默认 fuse_vae_embedding_in_latents=true）：

timestep          # [B]，每个 sample 一个扩散 t
# 首帧 latent 已去噪(t=0)，其余帧用当前 t
token_t = ones(B, T_lat, tokens_per_frame) * timestep
token_t[:, 0, :] = 0
token_t = token_t.reshape(B, S_v)              # 每个 video token 一个 t
t = time_embedding(sinusoidal_1d(token_t))     # [B, S_v, D_v]
t_mod_v = time_projection(t).unflatten(-1, (6, D_v))  # [B, S_v, 6, D_v]
#                              ↑ 6 路：shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

MoT 每层 _split_modulation 把它拆开，用于：

h = modulate(LayerNorm(x_v), shift_msa, scale_msa)   # self-attn 前
x_v = gate(x_v, gate_msa, attn_out)                  # self-attn 后
h = modulate(LayerNorm(x_v), shift_mlp, scale_mlp)   # FFN 前
x_v = gate(x_v, gate_mlp, ffn_out)                   # FFN 后

action 侧 t_mod_a 同理，但更简单——所有 action token 共享同一个 t：

t = time_embedding(sinusoidal_1d(timestep))    # [B, D_a]
t_mod_a = time_projection(t).unflatten(1, (6, D_a))  # [B, 6, D_a]，广播到每个 token

────────────────────────────────────────

2. S_v 是什么

video token 序列长度 = patchify 后 flatten 的总 token 数：

# 输入 latent
x : [B, C, T_lat, H_lat, W_lat]
# patchify: Conv3d, patch_size=(1,2,2), stride=patch_size
x = patch_embedding(x)          # [B, D_v, f, h, w]
#   f = T_lat
#   h = H_lat / 2
#   w = W_lat / 2
tokens_per_frame = h * w
S_v = f * h * w = T_lat * tokens_per_frame
x_v = rearrange(x, "b c f h w -> b (f h w) c")   # [B, S_v, D_v]

举例：T_lat=5, H_lat=32, W_lat=32 → h=w=16, tokens_per_frame=256, S_v = 5×256 = 1280。

action 侧对应 T_act（action horizon），没有 patchify。

────────────────────────────────────────

3. freqs_v 是什么，怎么算

3D RoPE 位置编码，给每个 video token 的 Q/K 注入 (帧, 高, 宽) 位置信息。

初始化时预计算三组 1D RoPE cache（标准 RoPE 公式）：

# attn_head_dim=128，拆成 f/h/w 三段
f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(128)
# 各为 [1024, dim_part] 的复数旋转因子

pre_dit 里按 grid (f, h, w) 广播拼到每个 token：

# 对 grid 中位置 (fi, hi, wi) 的 token：
freqs_v[token_idx] = concat(
    f_freqs[fi],   # 帧维
    h_freqs[hi],   # 高维
    w_freqs[wi],   # 宽维
)  # → [1, rope_dim=128]
freqs_v = stack_all_tokens   # [S_v, 1, 128]

MoT 里对 Q/K 应用：

Q_v = rope_apply(Q_v, freqs_v, num_heads=24)   # 旋转位置编码
K_v = rope_apply(K_v, freqs_v, num_heads=24)

action 侧是 1D RoPE（只有时间步序号）：

# ActionDiT 初始化
self.freqs = precompute_freqs_cis(128, end=1024)  # [1024, 64] 复数
# pre_dit
freqs_a = self.freqs[:T_act].view(T_act, 1, -1)   # [T_act, 1, 128]

────────────────────────────────────────

4. mask 大概怎么填

MoT joint mask 形状 [S_v + T_act, S_v + T_act]，True = 允许 attend。

mask = zeros(S_v + T_act, S_v + T_act)   # 默认全 False
# ── video → video（默认 first_frame_causal）──
mask[:S_v, :S_v] = video_to_video_mask
# 首帧 token 只看首帧；其余帧 token 看全部 video
#   mask[:tpf, tpf:] = False    (tpf = tokens_per_frame)
#   mask[tpf:, :]    = True
# ── action → action：全互看 ──
mask[S_v:, S_v:] = True
# ── action → video：只看首帧 video ──
mask[S_v:, :tpf] = True
# video → action：保持 False（video 看不到 action）

示意（S_v=4 简化为 2 帧×2 token，T_act=2，tpf=2）：

         vid0 vid1 vid2 vid3 | act0 act1
vid0      T    T    F    F   |  F    F
vid1      T    T    F    F   |  F    F
vid2      T    T    T    T   |  F    F
vid3      T    T    T    T   |  F    F
─────────────────────────────────────────
act0      T    T    F    F   |  T    T     ← 只看首帧 video
act1      T    T    F    F   |  T    T

────────────────────────────────────────
