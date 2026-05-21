```python
inputs = build_inputs(sample)
# inputs["input_latents"]      [B, C, T_lat, H_lat, W_lat]   C=48, T_lat=(T_v-1)//4+1
# inputs["first_frame_latents"] [B, C, 1, H_lat, W_lat] | None
# inputs["context"]            [B, L, D]                     D=4096
# inputs["context_mask"]       [B, L]
# inputs["action"]             [B, T_act, A]
# inputs["action_is_pad"]      [B, T_act] | None
# inputs["image_is_pad"]       [B, T_v]   | None
B = inputs["input_latents"].shape[0]
# ── video 扩散 ──
noise_v = randn_like(input_latents)              # [B, C, T_lat, H_lat, W_lat]
t_v = sample_t(B)                                # [B]
latents = add_noise(input_latents, noise_v, t_v)   # [B, C, T_lat, H_lat, W_lat]
target_v = training_target(input_latents, noise_v, t_v)  # [B, C, T_lat, H_lat, W_lat]
if first_frame_latents:
    latents[:, :, 0:1] = first_frame_latents   # 首帧保持干净 latent
# ── action 扩散 ──
noise_a = randn_like(action)                   # [B, T_act, A]
t_a = sample_t(B)                                # [B]
noisy_a = add_noise(action, noise_a, t_a)      # [B, T_act, A]
target_a = training_target(action, noise_a, t_a) # [B, T_act, A]
# ── 前向 ──
v_pre = video_expert.pre_dit(latents, t_v, context, action=action, ...)
# v_pre["tokens"]        [B, S_v, D_v]          S_v=T_lat·h·w, D_v=3072
# v_pre["t_mod"]         [B, S_v, 6, D_v]
# v_pre["freqs"]         [S_v, 1, 128]
# v_pre["context"]       [B, L, D_v]
# v_pre["context_mask"]  [B, S_v, L]
a_pre = action_expert.pre_dit(noisy_a, t_a, context, ...)
# a_pre["tokens"]        [B, T_act, D_a]        D_a=1024
# a_pre["t_mod"]         [B, 6, D_a]
# a_pre["freqs"]         [T_act, 1, 128]
# a_pre["context"]       [B, L, D_a]
# a_pre["context_mask"]  [B, T_act, L]
mask = build_mot_attention_mask(v_pre, a_pre)    # [S_v+T_act, S_v+T_act]
out = mot(v_pre.tokens, a_pre.tokens, mask, freqs, context, t_mod)
# out["video"]           [B, S_v, D_v]
# out["action"]          [B, T_act, D_a]
pred_v = video_expert.post_dit(out["video"], v_pre)   # [B, C, T_lat, H_lat, W_lat]
pred_a = action_expert.post_dit(out["action"], a_pre) # [B, T_act, A]
# ── loss ──
if first_frame_latents:
    pred_v   = pred_v[:, :, 1:]                  # [B, C, T_lat-1, H_lat, W_lat]
    target_v = target_v[:, :, 1:]                # [B, C, T_lat-1, H_lat, W_lat]
loss_v_ps = masked_mse(pred_v, target_v, image_is_pad)  # [B]
loss_v = mean(training_weight(t_v) * loss_v_ps)         # scalar  (weight [B])
loss_a_token = mse(pred_a, target_a).mean(dim=-1)       # [B, T_act]
loss_a_ps = masked_mean(loss_a_token, action_is_pad)    # [B]
loss_a = mean(training_weight(t_a) * loss_a_ps)         # scalar  (weight [B])
return λ_v*loss_v + λ_a*loss_a, {loss_video, loss_action}  # scalar, dict of scalars
```

────────────────────────────────────────

10 行版本：

```python
inp = build_inputs(sample)
# 加噪 + 构造 target（video latent + action）
latents, tgt_v = diffuse(inp.latents); noisy_a, tgt_a = diffuse(inp.action)
# pre → MoT joint attn → post
v_pre, a_pre = video_expert.pre_dit(...), action_expert.pre_dit(...)
pred_v = video_expert.post_dit(mot(v_pre, a_pre).video, v_pre)
pred_a = action_expert.post_dit(mot(v_pre, a_pre).action, a_pre)
# 加权 masked MSE，合并
loss_v = weighted_masked_mse(pred_v, tgt_v, inp.image_is_pad, t_v)
loss_a = weighted_masked_mse(pred_a, tgt_a, inp.action_is_pad, t_a)
return λ_v*loss_v + λ_a*loss_a
```
