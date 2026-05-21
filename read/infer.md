```python
def infer_action(input_image, context, action_horizon=32):  # Libero 默认: H×W=224×448, T_act=32, A=7, L=128
    first = vae.encode(input_image)                         # [1, 48, 1, 28, 56]
    ctx, ctx_mask = text_embed(context)                     # [1, 128, 4096], [1, 128]
    v_pre = video_expert.pre_dit(first, t=0, ctx)           # tokens [1, 392, 3072]  (S_v=14×28)
    mask = build_mot_mask(S_v=392, T_act=32)                # [424, 424]
    kv = mot.prefill_video_cache(v_pre, mask[:392,:392])    # 30层 × {k,v:[1,392,3072]}
    a = randn([1, 32, 7])                                   # [1, T_act, A]
    for t_a in action_schedule:                             # t_a: [1]
        pred_a = mot.forward_action_with_cache(             # pred [1, 32, 7]
            a, t_a, kv, mask, video_seq_len=392)
        a = action_scheduler.step(pred_a, a)                # [1, 32, 7]
    return a[0]                                             # [32, 7]

关于 pre_dit:
input_image [1,3,H,W]
    → VAE.encode（一次，观测编码）
    → first_frame_latents [1, 48, 1, H_lat, W_lat]
    → pre_dit（patchify + time/text 嵌入 + RoPE）
    → v_pre["tokens"] [1, S_v, 3072]   # 仅首帧 token，S_v≈392
- 不需要对未来 video latent 的随机初始化 + 多步 denoise
````
